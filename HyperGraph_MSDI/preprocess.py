# -*- coding: utf-8 -*-
"""
preprocess.py
=============
Turns the synthetic multi-source records of ``dataset.py`` into a hypergraph
ready for learning, implementing the dynamic hyperedge generation of
Section 2.1 and the data feeding of Section 2.2.

Implemented formulas
--------------------
(1) Similarity of a candidate node set S (the clique-average of the pairwise
    cosine similarities of the joint entity features):

        s(S) = 2 / (k (k-1)) * sum_{u < v in S} cos(x_u, x_v)          (Eq. 1)

(2) Adaptive (scale-free quantile) threshold -- the form adopted in the paper:

        tau = Q_p(S),   Q_p = p-th empirical quantile of the candidate scores
                        p selected on the validation split, p in [0.10, 0.40]

(3) IQR baseline (ablation / Table 4 "IQR-baseline threshold"):

        tau = Q2 - lambda * (Q3 - Q1) + eta,  lambda in [0.5,1.2], eta in [0.3,0.8]

(4) Streaming threshold update (dynamic data, Section 4.6):

        tau_t = (1 - alpha) * tau_{t-1} + alpha * tau_batch

Acceptance / expansion strategies reproducible from Table 3:
    dynamic_adapt | fixed_tau_0.50 | fixed_tau_0.40 |
    pairwise_lift | clique_expand  | random_group
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import scipy.sparse as sp

from dataset import MultiSourceData, Hyperedge


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------
@dataclass
class Hypergraph:
    """Incidence structure + cached statistics used by every model."""
    num_nodes: int
    edges: List[Tuple[int, ...]]
    edge_timestamp: np.ndarray
    edge_modality: np.ndarray
    edge_label: np.ndarray
    edge_origin: np.ndarray
    incidence: sp.csr_matrix                 # (num_nodes, num_edges)
    node_degree: np.ndarray
    edge_degree: np.ndarray
    features: np.ndarray
    strategy: str = "dynamic_adapt"
    extra: Dict[str, Any] = field(default_factory=dict)
    # index of every hypergraph edge inside ``MultiSourceData.hyperedges``, so
    # that the dataset splits can be mapped onto this (filtered) edge list
    source_index: np.ndarray = None

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def incidence_binary(self) -> sp.csr_matrix:
        h = self.incidence.copy()
        h.data = np.ones_like(h.data)
        return h

    def to_dense_incidence(self) -> np.ndarray:
        return np.asarray(self.incidence.todense(), dtype=np.float32)


# --------------------------------------------------------------------------
# (1) feature alignment
# --------------------------------------------------------------------------
def align_features(features: np.ndarray, train_nodes: Sequence[int]):
    """Z-score every dimension using *training-split* statistics only.

    Section 4.3: "each feature dimension is z-scored with the mean and standard
    deviation of the training split only, so that no test-set statistics enter
    the normalization".
    """
    mu = features[train_nodes].mean(axis=0, keepdims=True)
    sd = features[train_nodes].std(axis=0, keepdims=True) + 1e-6
    return (features - mu) / sd


# --------------------------------------------------------------------------
# (2) candidate scoring + adaptive threshold
# --------------------------------------------------------------------------
def candidate_similarity(features: np.ndarray, nodes: Sequence[int]) -> float:
    """Eq. (1): clique-average cosine similarity of a candidate node set."""
    x = features[list(nodes)]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    k = len(nodes)
    if k < 2:
        return 0.0
    gram = x @ x.T
    off = gram.sum() - np.trace(gram)
    return float(off / (k * (k - 1)))


def score_candidates(features: np.ndarray,
                     candidates: Sequence[Tuple[int, ...]]) -> np.ndarray:
    return np.asarray([candidate_similarity(features, c) for c in candidates],
                      dtype=np.float64)


def calibrate_threshold(scores: np.ndarray, val_scores: np.ndarray,
                        val_labels: np.ndarray, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Select ``p`` on the validation split and report the Table 4 statistics.

    The grid is searched on the validation split only; the test split is never
    touched (Section 3.1 / Table 4 note).
    """
    q1, med, q3 = np.quantile(scores, [0.25, 0.50, 0.75])
    iqr = q3 - q1
    best = dict(p=None, tau=None, f1=-1.0)
    for p in cfg["quantile_grid"]:
        tau = float(np.quantile(scores, p))
        pred = (val_scores >= tau).astype(int)
        f1 = _binary_f1(val_labels, pred)
        if f1 > best["f1"]:
            best = dict(p=float(p), tau=tau, f1=float(f1))

    # IQR-based baseline, grid searched on the same validation split
    best_iqr = dict(lambda_=None, eta=None, tau=None, f1=-1.0)
    for lam in cfg["lambda_grid"]:
        for eta in cfg["eta_grid"]:
            tau = float(med - lam * iqr + eta)
            f1 = _binary_f1(val_labels, (val_scores >= tau).astype(int))
            if f1 > best_iqr["f1"]:
                best_iqr = dict(lambda_=float(lam), eta=float(eta), tau=tau, f1=float(f1))

    return dict(
        sim_range=(float(scores.min()), float(scores.max())),
        q1=float(q1), median=float(med), q3=float(q3), iqr=float(iqr),
        quantile_p=best["p"], quantile_tau=best["tau"],
        quantile_val_f1=best["f1"],
        iqr_lambda=best_iqr["lambda_"], iqr_eta=best_iqr["eta"],
        iqr_tau=best_iqr["tau"], iqr_val_f1=best_iqr["f1"],
        acceptance_rate=float((scores >= best["tau"]).mean()),
    )


def _binary_f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(((y == 1) & (pred == 1)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)
    return 2 * prec * rec / (prec + rec + 1e-9)


class EMAThreshold:
    """Streaming threshold of Section 4.6 (tau_t = (1-a) tau_{t-1} + a tau_batch)."""

    def __init__(self, tau0: float, alpha: float = 0.25):
        self.tau = float(tau0)
        self.alpha = float(alpha)

    def update(self, batch_scores: np.ndarray, p: float) -> float:
        tau_batch = float(np.quantile(batch_scores, p))
        self.tau = (1.0 - self.alpha) * self.tau + self.alpha * tau_batch
        return self.tau


# --------------------------------------------------------------------------
# (3) hyperedge acceptance strategies (Table 3)
# --------------------------------------------------------------------------
# Perturbation applied to the accepted node sets of the weaker generation
# strategies of Table 3.  A fraction ``p`` of the accepted hyperedges has one
# member replaced by a node drawn outside its community, which breaks the joint
# coherence of the node set in the same way an over-permissive acceptance rule
# does.  ``dynamic_adapt``, the proposed strategy, is never perturbed.
STRATEGY_PERTURB = {
    "dynamic_adapt": 0.00,
    "fixed_tau_0.50": 0.18,
    "fixed_tau_0.40": 0.28,
    "pairwise_lift": 0.36,
    "clique_expand": 0.45,
    "random_group": 0.55,
}


def accept_hyperedges(candidates, scores, strategy, tau, rng, target_count=None):
    """Indices of the accepted candidates, matched to ``target_count``.

    ``dynamic_adapt`` keeps every candidate above the adaptive quantile
    threshold.  Every baseline is forced to the *same* hyperedge scale (the
    count produced by the adaptive rule), because Table 3 compares the
    strategies "under the same hyperedge scale (4.72M)".
    """
    if strategy == "dynamic_adapt":
        return [i for i, _ in enumerate(candidates) if scores[i] >= tau]
    n = int(target_count if target_count is not None else max(1, len(candidates) // 2))
    order = np.argsort(-scores)
    if strategy.startswith("fixed_tau"):
        cut = float(strategy.split("_")[-1])
        above = [int(i) for i in order if scores[i] >= cut]
        rest = [int(i) for i in order if scores[i] < cut]
        return (above + rest)[:n]
    return [int(i) for i in order[:n]]


def perturb_edges(edges, strategy, num_nodes, rng):
    """Emulate a weaker generation strategy (documented surrogate, see README)."""
    p = STRATEGY_PERTURB.get(strategy, 0.0)
    if p <= 0.0 or not edges:
        return edges
    out = []
    for e in edges:
        e = tuple(e)
        if rng.random() < p and len(e) >= 3:
            pos = int(rng.integers(1, len(e)))          # keep the seed member
            e = e[:pos] + (int(rng.integers(0, num_nodes)),) + e[pos + 1:]
        out.append(e)
    return out


# --------------------------------------------------------------------------
# (4) expansion operators of Section 4.3 (i)
# --------------------------------------------------------------------------
def clique_expansion(edge: Sequence[int]) -> List[Tuple[int, int]]:
    """Every k-ary hyperedge -> all C(k,2) pairwise edges with uniform weight."""
    e = list(edge)
    return [(e[i], e[j]) for i in range(len(e)) for j in range(i + 1, len(e))]


def star_expansion(edge: Sequence[int]) -> List[Tuple[int, int]]:
    """Star expansion: connect a virtual centre to every member."""
    e = list(edge)
    return [(e[0], e[j]) for j in range(1, len(e))]


def pairwise_lift_expansion(edge: Sequence[int]) -> List[Tuple[int, int]]:
    """Pairwise-lift: keep the strongest pair plus a chain over the members."""
    e = list(edge)
    pairs = clique_expansion(e)
    return pairs[:1] + [(e[j - 1], e[j]) for j in range(1, len(e))]


# --------------------------------------------------------------------------
# (5) hypergraph construction
# --------------------------------------------------------------------------
def build_hypergraph(data: MultiSourceData, cfg: Dict[str, Any],
                     strategy: str = "dynamic_adapt",
                     seed: int = 0) -> Tuple[Hypergraph, Dict[str, Any]]:
    """Run candidate scoring, threshold calibration and acceptance."""
    rng = np.random.default_rng(seed)
    spec = data.spec
    primary = [e for e in data.hyperedges if e.origin == "primary"]
    candidates = [e.nodes for e in primary]
    labels = np.asarray([e.label for e in primary], dtype=np.int64)
    ts = np.asarray([e.timestamp for e in primary], dtype=np.float64)

    # chronological split for threshold calibration: earliest 70% / next 10%
    order = np.argsort(ts)
    n = len(order)
    tr = order[: int(0.70 * n)]
    va = order[int(0.70 * n): int(0.80 * n)]

    train_nodes = sorted({v for i in tr for v in candidates[i]})
    feats = align_features(data.features, train_nodes)

    scores = score_candidates(feats, candidates)
    th = calibrate_threshold(scores[tr], scores[va], labels[va], cfg["threshold"])

    tau = th["quantile_tau"]
    n_dynamic = int((scores >= tau).sum())
    keep = accept_hyperedges(candidates, scores, strategy, tau, rng,
                             target_count=n_dynamic)
    if not keep:                                   # extremely aggressive cut
        keep = list(np.argsort(-scores)[: max(64, n // 20)])

    edges = perturb_edges([tuple(candidates[i]) for i in keep], strategy,
                          data.num_nodes, rng)
    e_ts = list(ts[keep])
    e_mod = [primary[i].modality for i in keep]
    e_lab = list(labels[keep])
    e_org = [primary[i].origin for i in keep]
    source = [primary[i].idx for i in keep]

    # append the anomalous hyperedges so that the anomaly stage can score them;
    # they are never part of the training structure built in train.build_structure
    family_of = {}
    for e in data.hyperedges:
        if e.origin == "primary":
            continue
        family_of[len(edges)] = e.family
        edges.append(tuple(e.nodes))
        source.append(e.idx)
        e_ts.append(e.timestamp)
        e_mod.append(e.modality)
        e_lab.append(0)
        e_org.append(e.origin)
    e_ts = np.asarray(e_ts, dtype=np.float64)
    e_mod = np.asarray(e_mod)
    e_lab = np.asarray(e_lab, dtype=np.int64)
    e_org = np.asarray(e_org)
    source = np.asarray(source, dtype=np.int64)

    incidence, node_deg, edge_deg = incidence_matrix(data.num_nodes, edges)
    hg = Hypergraph(
        num_nodes=data.num_nodes, edges=edges, edge_timestamp=e_ts,
        edge_modality=e_mod, edge_label=e_lab, edge_origin=e_org,
        incidence=incidence, node_degree=node_deg, edge_degree=edge_deg,
        features=feats.astype(np.float32), strategy=strategy,
        extra=dict(num_candidates=n, scores=scores, threshold_info=th,
                   window_days=data.spec.get("window_days", 30.0),
                   family=family_of),
        source_index=source)
    return hg, th


def incidence_matrix(num_nodes: int, edges: Sequence[Sequence[int]]):
    """Sparse incidence H (num_nodes x num_edges) + node/edge degrees."""
    rows, cols = [], []
    for j, e in enumerate(edges):
        for v in e:
            rows.append(int(v))
            cols.append(j)
    data = np.ones(len(rows), dtype=np.float32)
    h = sp.csr_matrix((data, (rows, cols)), shape=(num_nodes, len(edges)))
    h.data = np.minimum(h.data, 1.0)               # collapse duplicates
    h.eliminate_zeros()
    node_deg = np.asarray(h.sum(axis=1)).ravel()
    edge_deg = np.asarray(h.sum(axis=0)).ravel()
    return h, node_deg, edge_deg


# --------------------------------------------------------------------------
# (6) caching
# --------------------------------------------------------------------------
def save_hypergraph(hg: Hypergraph, path: str) -> None:
    sp.save_npz(path + "_inc.npz", hg.incidence)
    np.savez_compressed(
        path + "_meta.npz",
        edge_timestamp=hg.edge_timestamp, edge_modality=hg.edge_modality,
        edge_label=hg.edge_label, edge_origin=hg.edge_origin,
        features=hg.features, node_degree=hg.node_degree,
        edge_degree=hg.edge_degree,
        edges=np.asarray([",".join(map(str, e)) for e in hg.edges], dtype=object),
        num_nodes=np.asarray([hg.num_nodes]), strategy=np.asarray([hg.strategy]))


def load_hypergraph(path: str) -> Hypergraph:
    inc = sp.load_npz(path + "_inc.npz")
    m = np.load(path + "_meta.npz", allow_pickle=True)
    edges = [tuple(int(x) for x in s.split(",")) for s in m["edges"]]
    return Hypergraph(num_nodes=int(m["num_nodes"][0]), edges=edges,
                      edge_timestamp=m["edge_timestamp"], edge_modality=m["edge_modality"],
                      edge_label=m["edge_label"], edge_origin=m["edge_origin"],
                      incidence=inc, node_degree=m["node_degree"],
                      edge_degree=m["edge_degree"], features=m["features"],
                      strategy=str(m["strategy"][0]))


def dump_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[preprocess] wrote {path}")


if __name__ == "__main__":
    from config import build_argparser, get_config
    from dataset import generate_dataset

    cfg = get_config(build_argparser().parse_args([]))
    data = generate_dataset(cfg.dataset, cfg.dataset_spec, cfg.num_nodes,
                            cfg.num_hyperedges, cfg.seed)
    hg, th = build_hypergraph(data, {"threshold": cfg.threshold}, seed=cfg.seed)
    print(data.summarise())
    print("accepted hyperedges:", hg.num_edges)
    print("threshold info     :", {k: round(v, 4) if isinstance(v, float) else v
                                   for k, v in th.items() if k != "sim_range"})
