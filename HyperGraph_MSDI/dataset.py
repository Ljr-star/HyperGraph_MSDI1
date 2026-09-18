# -*- coding: utf-8 -*-
"""
dataset.py
==========
Synthetic multi-source heterogeneous data used to exercise the framework.

The three datasets of Section 3.1 (DBLP-OAG, MovieLens-25M, Yelp) cannot be
re-shipped, so this module *generates statistically plausible surrogates*
with exactly the properties the paper relies on:

  * three modalities (text / graph / temporal) that are aligned in a shared
    128-dimensional feature space (Section 4.3);
  * a genuine high-order label rule.  A candidate node set is a valid
    higher-order association iff the *majority* of its members carry the
    latent class ``c_v = 1``.  Majority is not recoverable from pairwise
    edges, which is what makes the hyperedge representation informative
    (this is the phenomenon quantified by delta-MI and BD-SL in Table 3);
  * the three natural-anomaly families of Table 7 and the three injected
    families of Section 4.5;
  * timestamps, so that the chronological (transductive), cold-start and
    strict-inductive protocols of Table 2 can be derived.

NOTE ON SCALE
-------------
Hyperedge construction is written as an explicit loop for readability.  At the
paper scale (millions of hyperedges) replace the loops with the vectorised
helpers marked ``# VECTORISE`` -- the semantics stay identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any

import numpy as np

from config import FEATURE_DIM

# feature-space layout: concatenated modality blocks (Section 4.3)
MODALITY_SLICES = {"text": (0, 48), "graph": (48, 88), "temporal": (88, 128)}


# --------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------
@dataclass
class Hyperedge:
    """One higher-order association (a hyperedge or a labelled candidate)."""
    idx: int
    nodes: Tuple[int, ...]
    timestamp: float
    modality: str
    label: int                       # 1 = valid association, 0 = invalid
    origin: str = "primary"          # primary | natural | injected
    family: str = ""                 # anomaly family name, '' when normal
    split: str = "train"             # train | val | test


@dataclass
class MultiSourceData:
    name: str
    display: str
    num_nodes: int
    node_type: np.ndarray            # (N,) object array of node types
    node_time: np.ndarray            # (N,) first appearance timestamp
    features: np.ndarray             # (N, D) aligned multi-modal features
    latent_class: np.ndarray         # (N,) hidden binary attribute c_v
    label_conflict: np.ndarray       # (N,) cross-source label contradiction flag
    merged_pair: Dict[int, int]      # entity-alignment collisions
    hyperedges: List[Hyperedge] = field(default_factory=list)
    spec: Dict[str, Any] = field(default_factory=dict)

    # ---------------- convenience accessors ----------------
    @property
    def num_edges(self) -> int:
        return len(self.hyperedges)

    def edges_by_split(self, split: str) -> List[Hyperedge]:
        return [e for e in self.hyperedges if e.split == split]

    def node_matrix(self) -> np.ndarray:
        return self.features

    def summarise(self) -> str:
        n_valid = sum(e.label for e in self.hyperedges)
        cards = [len(e.nodes) for e in self.hyperedges] or [0]
        return (f"[{self.display}] nodes={self.num_nodes} hyperedges={len(self.hyperedges)} "
                f"valid={n_valid} avg_cardinality={np.mean(cards):.2f}")


# --------------------------------------------------------------------------
# Helper generators
# --------------------------------------------------------------------------
def _sample_cardinality(rng: np.random.Generator, probs: List[float],
                        lo: int = 3, hi: int = 10) -> int:
    """Cardinality ~ {3, 4, 5, >=6} with the distribution reported in Sec. 3.1(d)."""
    bucket = rng.choice(4, p=np.asarray(probs, dtype=float) / np.sum(probs))
    if bucket == 0:
        return 3
    if bucket == 1:
        return 4
    if bucket == 2:
        return 5
    return int(rng.integers(6, hi + 1))


def _temporal_encoding(t: np.ndarray, dim: int, period: float) -> np.ndarray:
    """Sinusoidal encoding of timestamps -> (len(t), dim)."""
    freqs = np.arange(1, dim // 2 + 1, dtype=np.float64)
    ang = 2.0 * np.pi * (t[:, None] / period) * freqs[None, :]
    return np.concatenate([np.sin(ang), np.cos(ang)], axis=1)[:, :dim]


def _make_features(rng: np.random.Generator, latent_z: np.ndarray,
                   latent_c: np.ndarray, times: np.ndarray,
                   degrees: np.ndarray, period: float,
                   modalities: List[str]) -> np.ndarray:
    """Build the aligned 128-d feature space from modality-specific blocks."""
    n, z_dim = latent_z.shape
    out = np.zeros((n, FEATURE_DIM), dtype=np.float32)
    c_col = latent_c[:, None].astype(np.float32)

    for mod, (a, b) in MODALITY_SLICES.items():
        dim = b - a
        if mod not in modalities:
            # modality absent -> block receives a graph-style projection only
            mod = "graph"
        if mod == "text":
            w = rng.normal(0, 1.0 / np.sqrt(z_dim), size=(z_dim, dim))
            block = latent_z @ w + 0.35 * c_col * rng.normal(1.0, 0.1, size=(1, dim))
        elif mod == "graph":
            w = rng.normal(0, 1.0 / np.sqrt(z_dim), size=(z_dim, dim))
            block = latent_z @ w
            block[:, 0] = np.log1p(degrees) / 5.0            # degree signature
            block[:, 1] = c_col[:, 0] * 0.5                  # structural leakage
        else:  # temporal
            w = rng.normal(0, 1.0 / np.sqrt(z_dim), size=(z_dim, dim))
            enc = _temporal_encoding(times, dim, period)
            block = 0.7 * (latent_z @ w) + 0.6 * enc
        out[:, a:b] = block.astype(np.float32)

    out += rng.normal(0.0, 0.030, size=out.shape).astype(np.float32)
    return out


def _joint_label(latent_c: np.ndarray, nodes: Tuple[int, ...]) -> int:
    """Ground-truth higher-order rule: majority of the members carry c_v = 1."""
    votes = latent_c[list(nodes)].sum()
    return int(votes * 2 >= len(nodes))


# --------------------------------------------------------------------------
# Main generator
# --------------------------------------------------------------------------
def generate_dataset(name: str, spec: Dict[str, Any], num_nodes: int,
                     num_hyperedges: int, seed: int = 0) -> MultiSourceData:
    """Create the synthetic surrogate of ``name`` at the requested scale."""
    rng = np.random.default_rng(seed)
    modalities = list(spec["modalities"])

    # ---------------- entities ----------------
    types = spec["node_types"]
    counts = {
        "author": spec.get("num_authors", 0), "paper": spec.get("num_papers", 0),
        "venue": spec.get("num_venues", 0), "topic": spec.get("num_topics", 0),
        "institution": spec.get("num_institutions", 0),
        "user": spec.get("num_users", 0), "movie": spec.get("num_movies", 0),
        "genre": spec.get("num_genres", 0), "business": spec.get("num_businesses", 0),
        "category": spec.get("num_genres", 0),
    }
    active = [(t, max(counts.get(t, 0), 1)) for t in types]
    weights = np.asarray([c for _, c in active], dtype=float)
    weights /= weights.sum()
    node_type = rng.choice([t for t, _ in active], size=num_nodes, p=weights).astype(object)

    period = 365.0 * 18.0                                   # ~18 years of events
    node_time = np.sort(rng.uniform(0.0, period, size=num_nodes))
    node_time += rng.uniform(0.0, 30.0, size=num_nodes)     # 30-day granularity

    # hidden latent class, slightly type dependent
    p_c = np.where(node_type == "paper", 0.55, 0.48)
    latent_c = (rng.random(num_nodes) < p_c).astype(np.int64)
    latent_z = rng.normal(0, 1, size=(num_nodes, 32))

    # entity-resolution collisions (identical normalised name / DOI prefix)
    n_merged = max(4, int(0.02 * num_nodes))
    merged_pair = {int(i): int(i + 1) for i in rng.choice(
        np.arange(0, max(1, num_nodes - 1)), size=n_merged, replace=False)}

    # cross-source label contradictions (DBLP topic vs OAG field of study)
    n_conflict = max(4, int(0.03 * num_nodes))
    label_conflict = np.zeros(num_nodes, dtype=np.int64)
    label_conflict[rng.choice(num_nodes, size=n_conflict, replace=False)] = 1

    degrees = np.zeros(num_nodes, dtype=np.float64)

    # ---------------- hyperedges ----------------
    hyperedges: List[Hyperedge] = []
    n_target = num_hyperedges
    eid = 0
    # community structure keeps candidate sets locally plausible (2-hop rule)
    n_communities = max(8, num_nodes // 64)
    community = rng.integers(0, n_communities, size=num_nodes)
    # pre-compute the member list of every community (the sampling loop below
    # would otherwise rescan all nodes at every iteration)
    order = np.argsort(community, kind="stable")
    sorted_comm = community[order]
    bounds = np.searchsorted(sorted_comm, np.arange(n_communities + 1))
    comm_members = {c: order[bounds[c]:bounds[c + 1]] for c in range(n_communities)}

    while eid < n_target:
        k = _sample_cardinality(rng, spec["cardinality_probs"],
                                hi=min(10, max(3, num_nodes // 4)))
        # sample a seed node, then draw the rest from its community / neighbours
        seed_node = int(rng.integers(0, num_nodes))
        pool = comm_members[int(community[seed_node])]
        if len(pool) < k:
            pool = np.arange(num_nodes)
        members = tuple(int(x) for x in rng.choice(pool, size=k, replace=False))
        t = float(max(node_time[list(members)])) + float(rng.uniform(0, 30))
        mod = str(rng.choice(modalities))
        y = _joint_label(latent_c, members)
        hyperedges.append(Hyperedge(idx=eid, nodes=members, timestamp=t,
                                    modality=mod, label=y))
        degrees[list(members)] += 1.0
        eid += 1

    # balanced labelled pool: keep positives plus matched negatives
    positives = [e for e in hyperedges if e.label == 1]
    negatives = [e for e in hyperedges if e.label == 0]
    n_keep = min(len(positives), len(negatives), max(4_000, num_hyperedges))
    rng.shuffle(positives)
    rng.shuffle(negatives)
    keepp = positives[:n_keep]
    keepn = negatives[:n_keep]
    for e in keepn:
        e.label = 0
    hyperedges = sorted(keepp + keepn, key=lambda e: e.timestamp)
    for i, e in enumerate(hyperedges):
        e.idx = i

    # ---------------- natural anomalies (Table 7) ----------------
    per_family = max(4, int(spec.get("anomaly_per_family", 2_100)
                            * max(0.002, num_hyperedges / max(1, spec["target_hyperedges"]))))
    natural: List[Hyperedge] = []

    # (1) temporally inconsistent collaboration:
    #     co-occurrence dated before the earliest publication of any member
    for e in hyperedges[: int(per_family)]:
        natural.append(Hyperedge(idx=-1, nodes=e.nodes,
                                 timestamp=float(np.min(node_time[list(e.nodes)]) - rng.uniform(30, 400)),
                                 modality=e.modality, label=0, origin="natural",
                                 family="temporally_inconsistent_collaboration"))

    # (2) cross-source semantic inconsistency:
    #     members linked only by a contradicting entity-resolution link
    conflicted = np.where(label_conflict == 1)[0]
    for _ in range(int(per_family)):
        k = _sample_cardinality(rng, spec["cardinality_probs"], hi=8)
        members = tuple(int(x) for x in rng.choice(conflicted, size=min(k, len(conflicted)),
                                                   replace=False))
        natural.append(Hyperedge(idx=-1, nodes=members,
                                 timestamp=float(rng.uniform(0, period)),
                                 modality="cross_modal", label=0, origin="natural",
                                 family="cross_source_semantic_inconsistency"))

    # (3) attribute confusion caused by entity-alignment conflict
    for i, (a, b) in enumerate(list(merged_pair.items())[: int(per_family)]):
        extra = rng.choice(num_nodes, size=max(0, _sample_cardinality(rng, spec["cardinality_probs"], hi=8) - 2),
                           replace=False)
        members = tuple(int(x) for x in np.concatenate([[a, b], extra]))
        natural.append(Hyperedge(idx=-1, nodes=members,
                                 timestamp=float(rng.uniform(0, period)),
                                 modality="graph", label=0, origin="natural",
                                 family="attribute_confusion_entity_alignment"))

    # ---------------- injected anomalies (Section 4.5) ----------------
    n_inj = max(6, int(spec.get("anomaly_per_dataset", 4_700)
                       * max(0.002, num_hyperedges
                             / max(1, spec["target_hyperedges"]))))
    injected: List[Hyperedge] = []
    fams = ["node", "event", "cross_modal"]
    for i in range(n_inj):
        fam = fams[i % 3]
        k = _sample_cardinality(rng, spec["cardinality_probs"], hi=8)
        members = tuple(int(x) for x in rng.choice(num_nodes, size=k, replace=False))
        if fam == "node":
            # swap entity attributes across domains
            idx = list(members)
            latent_z[idx] = latent_z[idx][::-1]
        elif fam == "event":
            # insert an out-of-sequence event inside a temporal window
            t = float(np.min(node_time[list(members)]) - rng.uniform(1, 20))
        else:
            # break the text/graph/temporal mapping of one member
            v = int(members[0])
            latent_z[v] = rng.normal(0, 1, size=latent_z.shape[1])
        injected.append(Hyperedge(idx=-1, nodes=members,
                                  timestamp=float(rng.uniform(0, period)),
                                  modality="cross_modal" if fam == "cross_modal" else str(rng.choice(modalities)),
                                  label=1, origin="injected", family=fam))

    all_edges = hyperedges + natural + injected
    for i, e in enumerate(all_edges):
        e.idx = i

    # ---------------- features ----------------
    features = _make_features(rng, latent_z, latent_c, node_time, degrees,
                              period, modalities)
    # propagate the injected node-level corruption into the feature space
    for e in injected:
        if e.family == "node":
            v = int(e.nodes[0])
            a, b = MODALITY_SLICES["text"]
            features[v, a:b] = features[v, a:b][::-1]
        elif e.family == "cross_modal":
            a, b = MODALITY_SLICES["temporal"]
            features[int(e.nodes[0]), a:b] += 2.0

    data = MultiSourceData(
        name=name, display=spec["display"], num_nodes=num_nodes,
        node_type=node_type, node_time=node_time, features=features,
        latent_class=latent_c, label_conflict=label_conflict,
        merged_pair=merged_pair, hyperedges=all_edges, spec=spec)
    return data


# --------------------------------------------------------------------------
# Splits (Section 3.1: transductive / cold-start / strict inductive)
# --------------------------------------------------------------------------
def assign_splits(data: MultiSourceData, ratios: Tuple[float, float, float],
                  seed: int = 0) -> Dict[str, Dict[str, Any]]:
    """Chronological 7:1:2 split; returns the derived protocol index sets.

    transductive      : hyperedges and time disjoint, nodes may be shared
    cold_start        : test hyperedges holding >=1 node unseen during training
    strict_inductive  : additionally drop every training hyperedge that shares
                        a node with any test hyperedge
    """
    rng = np.random.default_rng(seed + 777)
    primary = [e for e in data.hyperedges if e.origin == "primary"]
    primary.sort(key=lambda e: e.timestamp)
    n = len(primary)
    n_tr = int(ratios[0] * n)
    n_va = int(ratios[1] * n)

    for i, e in enumerate(primary):
        e.split = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")

    train_nodes = set()
    for e in primary[:n_tr]:
        train_nodes.update(e.nodes)

    test_edges = primary[n_tr + n_va:]
    cold = [e for e in test_edges if any(v not in train_nodes for v in e.nodes)]
    strict_test = set(id(e) for e in test_edges)
    strict_train = [e for e in primary[:n_tr]
                    if not (set(e.nodes) & set(v for te in test_edges for v in te.nodes))]

    idx = dict(
        train=[e.idx for e in primary[:n_tr]],
        val=[e.idx for e in primary[n_tr:n_tr + n_va]],
        test=[e.idx for e in test_edges],
        cold_start=[e.idx for e in cold],
        strict_inductive_train=[e.idx for e in strict_train],
        strict_inductive_test=[e.idx for e in test_edges],
    )
    idx["meta"] = dict(num_train=len(idx["train"]), num_val=len(idx["val"]),
                       num_test=len(idx["test"]), num_cold_start=len(idx["cold_start"]),
                       cold_start_fraction=len(cold) / max(1, len(test_edges)),
                       num_strict_train=len(strict_train))
    return idx


def get_association_labels(data: MultiSourceData, edge_ids: List[int]) -> np.ndarray:
    """Binary validity labels used by the association-discovery task."""
    by_idx = {e.idx: e for e in data.hyperedges}
    return np.asarray([by_idx[i].label for i in edge_ids], dtype=np.int64)


if __name__ == "__main__":
    from config import build_argparser, get_config
    cfg = get_config(build_argparser().parse_args([]))
    d = generate_dataset(cfg.dataset, cfg.dataset_spec, cfg.num_nodes,
                         cfg.num_hyperedges, seed=cfg.seed)
    print(d.summarise())
    print("splits:", assign_splits(d, cfg.dataset_spec["split_ratio"])["meta"])
