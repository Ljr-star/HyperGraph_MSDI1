# -*- coding: utf-8 -*-
"""
train.py
========
Training and experiment drivers.

Implemented experiments
-----------------------
* Association discovery  -> F1, per modality (Figure 5(a)) and per training-set
  size (Figure 5(b), fixed 44K-sample test set).
* Higher-order association reasoning -> HR@K (Figure 6).
* Evaluation protocols   -> transductive / cold-start / strict inductive (Table 2).
* Anomaly detection      -> detection rate, AP, AUC, F1 (Tables 8 and 9).
* Ablation               -> Table 10 (modules), plus the head / layer /
  dimension sweeps reported in Section 4.6.

The association strength used for anomaly detection is

        s(v, e) = < z_v , m_e >,   m_e = mean_{u in e} z_u            (Eq. 9)

and a node-hyperedge pair is flagged when |s - mu| > 2 sigma (bidirectional
two-standard-deviation rule of Section 2.4).
"""

from __future__ import annotations

import copy
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (MODEL_REGISTRY, build_model, build_meta_paths,
                   scipy_to_torch_sparse, count_parameters)
from preprocess import Hypergraph, incidence_matrix
from evaluate import (f1_score, average_precision, roc_auc, detection_rate,
                      bootstrap_ci as _bootstrap_mean_ci)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# node-feature tensors are reused across many batches -> cache them
_X_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def node_tensor(hg: Hypergraph, device: str) -> torch.Tensor:
    key = (id(hg.features), device)
    t = _X_CACHE.get(key)
    if t is None:
        t = torch.tensor(hg.features, dtype=torch.float32, device=device)
        _X_CACHE.clear()
        _X_CACHE[key] = t
    return t


# --------------------------------------------------------------------------
# structure / batch construction
# --------------------------------------------------------------------------
def map_splits(splits: Dict, hg: Hypergraph) -> Dict:
    """Map dataset-level edge ids onto the hypergraph edge list of ``hg``.

    ``build_hypergraph`` keeps only the accepted candidates (plus the appended
    anomaly edges), so the chronological splits of ``dataset.assign_splits``
    must be translated through ``hg.source_index``.
    """
    lut = {int(s): i for i, s in enumerate(np.asarray(hg.source_index))}
    out = {}
    for key, value in splits.items():
        if isinstance(value, (list, tuple)):
            out[key] = [lut[int(x)] for x in value if int(x) in lut]
        else:
            out[key] = value
    return out


def build_structure(hg: Hypergraph, edge_ids: Sequence[int], dataset: str,
                    cfg: Dict, device: str = DEFAULT_DEVICE) -> Dict:
    """Restrict the hypergraph to ``edge_ids`` and pre-compute all operators."""
    edges = [hg.edges[i] for i in edge_ids]
    inc, dv, de = incidence_matrix(hg.num_nodes, edges)
    h_t = scipy_to_torch_sparse(inc, device)
    dv_t = torch.tensor(dv, dtype=torch.float32, device=device).clamp_min(1.0)
    de_t = torch.tensor(de, dtype=torch.float32, device=device).clamp_min(1.0)

    rows, cols = [], []
    for j, e in enumerate(edges):
        for i in range(len(e)):
            for k in range(i + 1, len(e)):
                rows.append(e[i]); cols.append(e[k])
                rows.append(e[k]); cols.append(e[i])
    adj = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                        shape=(hg.num_nodes, hg.num_nodes))
    d = np.asarray(adj.sum(1)).ravel(); d[d == 0] = 1.0
    adj = sp.diags(1.0 / np.sqrt(d)) @ adj @ sp.diags(1.0 / np.sqrt(d))

    # mediator graph for HyperGCN: one virtual node per hyperedge
    return dict(
        edges=edges, incidence=inc, dv=dv_t, de=de_t,
        h=h_t,
        adj_clique=scipy_to_torch_sparse(adj, device),
        adj_mediator=scipy_to_torch_sparse(_mediator_adjacency(hg.num_nodes, edges), device),
    )


def _mediator_adjacency(num_nodes: int, edges: Sequence[Sequence[int]]) -> sp.csr_matrix:
    """Star expansion: members connected through a virtual mediator node."""
    rows, cols = [], []
    offset = num_nodes
    for j, e in enumerate(edges):
        med = offset + j
        for v in e:
            rows += [v, med]
            cols += [med, v]
    n = num_nodes + len(edges)
    m = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n, n))
    d = np.asarray(m.sum(1)).ravel(); d[d == 0] = 1.0
    return (sp.diags(1.0 / np.sqrt(d)) @ m @ sp.diags(1.0 / np.sqrt(d))).tocsr()


def padded_members(edges: Sequence[Sequence[int]], edge_ids: Sequence[int],
                   device: str = DEFAULT_DEVICE):
    max_k = max(3, max(len(edges[i]) for i in edge_ids))
    idx = np.zeros((len(edge_ids), max_k), dtype=np.int64)
    mask = np.zeros((len(edge_ids), max_k), dtype=np.float32)
    for r, i in enumerate(edge_ids):
        e = edges[i]
        idx[r, : len(e)] = e
        mask[r, : len(e)] = 1.0
    return torch.tensor(idx, device=device), torch.tensor(mask, device=device)


def make_batch(hg: Hypergraph, struct: Dict, edge_ids: Sequence[int],
               dataset: str, cfg: Dict, path_cache: Optional[List] = None,
               device: str = DEFAULT_DEVICE) -> Dict:
    """Assemble everything a model needs for one labelled edge set."""
    mi, mk = padded_members(hg.edges, edge_ids, device)
    x = node_tensor(hg, device)
    y = torch.tensor([hg.edge_label[i] for i in edge_ids], dtype=torch.float32, device=device)
    paths = path_cache if path_cache is not None else []
    x_med = torch.cat([x, torch.zeros(struct["h"].shape[1], x.shape[1], device=device)], dim=0)
    return dict(x=x, x_mediator=x_med, h=struct["h"], dv=struct["dv"], de=struct["de"],
                adj_clique=struct["adj_clique"], adj_mediator=struct["adj_mediator"],
                member_idx=mi, mask=mk, y=y,
                path_mats=[p.to(device) for p in paths] if paths else [],
                edge_ids=list(edge_ids))


def build_path_cache(hg: Hypergraph, struct: Dict, dataset: str, cfg: Dict,
                     device: str = DEFAULT_DEVICE) -> List[torch.Tensor]:
    mats = build_meta_paths(struct["incidence"], _node_types(hg, dataset), dataset,
                            cfg.model["hidden_dim"], cfg.model["meta_paths"])
    return [scipy_to_torch_sparse(m, device) for m in mats]


def _node_types(hg: Hypergraph, dataset: str) -> np.ndarray:
    if hasattr(hg, "node_type"):
        return np.asarray(hg.node_type)
    from model import TYPE_LETTERS
    letters = TYPE_LETTERS.get(dataset, ["author", "paper", "venue", "topic"])
    # fall back to a deterministic assignment when the type array is absent
    return np.asarray([letters[i % len(letters)] for i in range(hg.num_nodes)])


# --------------------------------------------------------------------------
# training utilities
# --------------------------------------------------------------------------
def pick_device(requested: str) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        print("[train] CUDA unavailable -> using CPU")
        return "cpu"
    return requested


@torch.no_grad()
def predict(model: nn.Module, batch: Dict) -> np.ndarray:
    model.eval()
    return torch.sigmoid(model(batch)).detach().cpu().numpy()


def best_threshold(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    """Threshold maximising F1 on a validation split."""
    grid = np.unique(np.quantile(p, np.linspace(0.05, 0.95, 60)))
    best = (0.5, -1.0)
    for t in grid:
        f1 = f1_score(y, (p >= t).astype(int))
        if f1 > best[1]:
            best = (float(t), float(f1))
    return best


def train_one(name: str, hg: Hypergraph, struct: Dict, splits: Dict, dataset: str,
              cfg: Dict, path_cache: Optional[List] = None, seed: int = 0,
              verbose: bool = True) -> Dict:
    """Train one model and return its metrics on the fixed test set."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = pick_device(cfg.train["device"])
    model = build_model(name, hg.features.shape[1], cfg.model).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.train["lr"],
                           weight_decay=cfg.train["weight_decay"])

    tr_ids, va_ids, te_ids = splits["train"], splits["val"], splits["test"]
    cap = cfg.train["labeled_test_positives"] * 2      # balanced 22k/22k pool
    te_ids = list(te_ids[:cap]) if cap else list(te_ids)
    tr_batch = make_batch(hg, struct, tr_ids, dataset, cfg, path_cache, device)
    va_batch = make_batch(hg, struct, va_ids, dataset, cfg, path_cache, device)
    te_batch = make_batch(hg, struct, te_ids, dataset, cfg, path_cache, device)

    best = dict(f1=-1.0, state=None, epoch=0, thr=0.5)
    patience = 0
    for ep in range(1, cfg.train["epochs"] + 1):
        model.train()
        opt.zero_grad()
        logits = model(tr_batch)
        loss = F.binary_cross_entropy_with_logits(logits, tr_batch["y"])
        loss.backward()
        opt.step()

        p_val = predict(model, va_batch)
        thr, f1v = best_threshold(va_batch["y"].cpu().numpy(), p_val)
        if f1v > best["f1"] + 1e-5:
            best = dict(f1=float(f1v), state=copy.deepcopy(model.state_dict()),
                        epoch=ep, thr=float(thr))
            patience = 0
        else:
            patience += 1
            if patience >= cfg.train["patience"]:
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    p_te = predict(model, te_batch)
    y_te = te_batch["y"].cpu().numpy()
    pred = (p_te >= best["thr"]).astype(int)
    res = dict(model=name, seed=seed, f1=f1_score(y_te, pred),
               val_f1=best["f1"], epochs=best["epoch"],
               params=count_parameters(model),
               scores=p_te, labels=y_te, probs=p_te, thr=best["thr"],
               net=model, test_ids=list(te_ids))
    if verbose:
        print(f"[train] {name:15s} seed={seed} F1={res['f1']:.4f} "
              f"({res['epochs']} ep, {res['params']/1e3:.1f}K params)")
    return res


# --------------------------------------------------------------------------
# node embeddings are computed ONCE per model; every downstream score is then
# a cheap lookup, which keeps HR@K and the anomaly sweep affordable
# --------------------------------------------------------------------------
@torch.no_grad()
def structure_embeddings(hg: Hypergraph, model: nn.Module, struct: Dict,
                         cfg: Dict, path_cache=None,
                         device: str = DEFAULT_DEVICE) -> torch.Tensor:
    """Run the encoder once over the whole graph and return z (N, d)."""
    model.eval()
    batch = make_batch(hg, struct, [0], getattr(cfg, "dataset", ""), cfg, path_cache, device)
    batch["member_idx"] = torch.zeros((1, 1), dtype=torch.long, device=device)
    batch["mask"] = torch.ones((1, 1), device=device)
    return model.encode(batch)


@torch.no_grad()
def score_edges(model: nn.Module, z: torch.Tensor, hg: Hypergraph,
                edge_ids: Sequence[int], device: str = DEFAULT_DEVICE) -> np.ndarray:
    """Association score of each edge id from pre-computed node embeddings."""
    model.eval()
    mi, mk = padded_members(hg.edges, edge_ids, device)
    pooled = model.readout(z[mi], mk)
    return torch.sigmoid(model.classifier(pooled)).squeeze(-1).cpu().numpy()


# --------------------------------------------------------------------------
# HR@K / association reasoning (Figure 6)
# --------------------------------------------------------------------------
def hr_at_k(hg: Hypergraph, model: nn.Module, struct: Dict, test_ids: Sequence[int],
            cfg: Optional[Dict] = None, path_cache=None,
            ks: Sequence[int] = (1, 5, 10, 20, 50, 100),
            negatives_per_pos: int = 100, seed: int = 0,
            device: str = DEFAULT_DEVICE, z: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """Rank candidate associations; HR@K = fraction with the true one in top-K."""
    rng = np.random.default_rng(seed)
    if z is None:
        z = structure_embeddings(hg, model, struct, cfg or {}, path_cache, device)
    hits = {k: 0 for k in ks}
    total = 0
    negatives = np.where(hg.edge_label == 0)[0]
    for pid in [i for i in test_ids if hg.edge_label[i] == 1]:
        cand = [pid] + [int(x) for x in rng.choice(
            negatives, size=negatives_per_pos, replace=len(negatives) < negatives_per_pos)]
        sc = score_edges(model, z, hg, cand, device)
        rank = int(np.where(np.argsort(-sc) == 0)[0][0]) + 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
        total += 1
    return {k: hits[k] / max(1, total) for k in ks}


# --------------------------------------------------------------------------
# anomaly detection (Sections 2.4 / 4.5, Tables 8 and 9)
# --------------------------------------------------------------------------
@torch.no_grad()
def association_scores(hg: Hypergraph, model: nn.Module, batch: Dict,
                       device: str = DEFAULT_DEVICE,
                       z: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
    """s(v, e) = <z_v, m_e> for every incidence of the given edges."""
    model.eval()
    if z is None:
        z = model.encode(batch)
    idx = batch["h"].indices()
    node_ids, edge_ids = idx[0], idx[1]
    m = torch.zeros(batch["h"].shape[1], z.shape[1], device=device)
    m = m.index_add_(0, edge_ids, z[node_ids]) / batch["de"][:, None].clamp_min(1.0)
    s = (z[node_ids] * m[edge_ids]).sum(-1)
    return s.cpu().numpy(), (node_ids.cpu().numpy(), edge_ids.cpu().numpy())


def anomaly_detection(hg: Hypergraph, model: nn.Module, struct: Dict,
                      edge_ids: Sequence[int], cfg: Dict,
                      device: str = DEFAULT_DEVICE) -> Dict[str, float]:
    """Bidirectional two-standard-deviation rule + evaluation metrics."""
    batch = make_batch(hg, struct, edge_ids, "", {"model": {}, "train": {}}, None, device)
    s, (n_ids, e_ids) = association_scores(hg, model, batch, device)
    mu, sd = float(s.mean()), float(s.std() + 1e-9)
    flag = (np.abs(s - mu) > cfg.anomaly["std_rule"] * sd).astype(int)

    local = {i: e for i, e in enumerate(edge_ids)}
    truth = np.zeros_like(flag)
    for t, (v, e_local) in enumerate(zip(n_ids, e_ids)):
        gid = local[int(e_local)]
        e = hg.edges[gid]
        truth[t] = 1 if (hg.edge_origin[gid] != "primary" and int(v) in e) else 0

    return dict(detection_rate=detection_rate(truth, flag),
                ap=average_precision(truth, np.abs(s - mu)),
                auc=roc_auc(truth, np.abs(s - mu)),
                f1=f1_score(truth, flag), threshold=2.0 * sd, mu=mu)


# --------------------------------------------------------------------------
# learning curve with a fixed test set (Figure 5(b))
# --------------------------------------------------------------------------
def learning_curve(name: str, hg: Hypergraph, struct: Dict, splits: Dict,
                   dataset: str, cfg: Dict, fractions: Sequence[float],
                   path_cache=None, seed: int = 0) -> List[Dict]:
    """F1 and bootstrap CI width as the *training* size grows (test fixed)."""
    out = []
    for frac in fractions:
        n = max(64, int(frac * len(splits["train"])))
        sub = dict(splits)
        sub["train"] = splits["train"][:n]
        r = train_one(name, hg, struct, sub, dataset, cfg, path_cache, seed, verbose=False)
        ci = f1_bootstrap_ci(r["scores"], r["labels"], n_iter=200, seed=seed)
        out.append(dict(fraction=frac, train_size=n, f1=r["f1"],
                        ci_half_width=0.5 * (ci[1] - ci[0])))
        print(f"[curve] frac={frac:.2f} n={n} F1={r['f1']:.4f} "
              f"CI=+-{out[-1]['ci_half_width']:.3f}")
    return out


def f1_bootstrap_ci(scores: np.ndarray, labels: np.ndarray, n_iter: int = 1_000,
                    alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    """Bootstrap 95% CI of the F1 score on a fixed test set (Section 3.2)."""
    rng = np.random.default_rng(seed)
    n = len(labels)
    if n == 0:
        return (float("nan"), float("nan"))
    thr, _ = best_threshold(labels, scores)
    pred = (scores >= thr).astype(int)
    vals = [f1_score(labels[idx], pred[idx]) for idx in rng.integers(0, n, (n_iter, n))]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# ablation (Table 10 and Section 4.6 sweeps)
# --------------------------------------------------------------------------
ABLATION_VARIANTS = {
    "full": dict(),
    "wo_attention": dict(use_attention=False),
    "wo_hypergraph_conv": dict(use_hypergraph_conv=False),
    "wo_dynamic_hyperedge": dict(fixed_threshold=True),
    "wo_tensor_biindex": dict(use_tensor_index=False),
    "gcn_baseline": dict(as_baseline="gcn"),
    "heads_8": dict(num_heads=8),
}


def run_ablation(hg: Hypergraph, struct: Dict, splits: Dict, dataset: str,
                 cfg: Dict, variants: Sequence[str], path_cache=None,
                 seed: int = 0,
                 alt_variants: Optional[Dict[str, Tuple]] = None) -> List[Dict]:
    """Train every ablation variant of Table 10.

    ``alt_variants`` maps a variant name to ``(hg, struct, path_cache)`` so that
    variants which require a *different hypergraph* (for example
    ``wo_dynamic_hyperedge``, which must be built with a fixed threshold) can be
    evaluated on the structure they actually describe.
    """
    rows = []
    alt_variants = alt_variants or {}
    for v in variants:
        spec = ABLATION_VARIANTS[v]
        mc = dict(cfg.model)
        if "use_attention" in spec:
            mc["use_attention"] = spec["use_attention"]
        if "num_heads" in spec:
            mc["num_heads"] = spec["num_heads"]
        if spec.get("use_hypergraph_conv", True) is False:
            mc["num_layers"] = 1          # degenerates to a member-set MLP
            mc["use_meta_path"] = False
        name = spec.get("as_baseline", "hyperconv_attn")
        sub_cfg = copy.deepcopy(cfg)
        sub_cfg.model = mc

        hg_v, struct_v, paths_v = alt_variants.get(v, (hg, struct, path_cache))
        r = train_one(name, hg_v, struct_v, splits, dataset, sub_cfg, paths_v, seed,
                      verbose=False)
        rows.append(dict(variant=v, f1=100 * r["f1"], params=r["params"]))
        print(f"[ablation] {v:22s} F1={100*r['f1']:.1f}%")
    return rows


def sweep_heads(hg, struct, splits, dataset, cfg, heads: Sequence[int],
                path_cache=None, seed: int = 0) -> List[Dict]:
    rows = []
    for h in heads:
        mc = dict(cfg.model); mc["num_heads"] = h
        sub = copy.deepcopy(cfg); sub.model = mc
        r = train_one("hyperconv_attn", hg, struct, splits, dataset, sub,
                      path_cache, seed, verbose=False)
        rows.append(dict(heads=h, f1=100 * r["f1"], params=r["params"]))
    return rows


if __name__ == "__main__":
    print("train.py is driven by main.py; run:  python main.py --stage train")
