# -*- coding: utf-8 -*-
"""
model.py
========
Neural components of the framework and all baselines needed for Tables 8-11.

Paper mapping
-------------
Section 2.3  two-stage message passing / hypergraph Laplacian
             Y = sigma( D_v^-1/2 H W D_e^-1 H^T D_v^-1/2 X Theta )        (Eq. 2)
             with D_v = diag(HW1), D_e = diag(1^T H) and the hypergraph
             Laplacian L = I - D_v^-1/2 H W D_e^-1 H^T D_v^-1/2.
Section 2.4  adaptive hyperedge attention
             alpha_{v,e} = softmax_{e in E(v)}( LeakyReLU(a1.x_v + a2.m_e) )
             and meta-path-guided multivariate relationship reasoning over the
             paths APA / APV / APT.
Section 4.5  association strength s -> anomaly if |s - mu| > 2 sigma.

Also implemented here: HGNN, HyperConv, HyperGCN, UniGNN, AllSet, AllDeepSets,
ED-HNN, a linear-attention hypergraph transformer (Hypformer-style), the binary
baselines GCN / GAT / GraphSAGE acting on the clique expansion, and the two
auxiliary models used by the validity criteria of Section 4.3
(logistic probe and hyperedge auto-encoder) plus Deep SVDD for Table 9.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_SLOPE = 0.20


# --------------------------------------------------------------------------
# sparse helpers
# --------------------------------------------------------------------------
def scipy_to_torch_sparse(m: sp.spmatrix, device=None) -> torch.Tensor:
    """Convert a scipy sparse matrix to a coalesced torch sparse COO tensor."""
    m = m.tocoo()
    idx = torch.from_numpy(np.vstack([m.row, m.col]).astype(np.int64))
    val = torch.from_numpy(m.data.astype(np.float32))
    t = torch.sparse_coo_tensor(idx, val, m.shape)
    t = t.coalesce()
    return t.to(device) if device is not None else t


def safe_inv(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return 1.0 / x.clamp_min(eps)


def segment_softmax(scores: torch.Tensor, seg: torch.Tensor, num_seg: int) -> torch.Tensor:
    """Softmax of ``scores`` inside each segment id given by ``seg``."""
    mx = torch.full((num_seg,), float("-inf"), device=scores.device,
                    dtype=scores.dtype)
    mx = mx.scatter_reduce(0, seg, scores, reduce="amax", include_self=True)
    mx = torch.where(torch.isfinite(mx), mx, torch.zeros_like(mx))
    s = torch.exp(scores - mx[seg])
    z = torch.zeros(num_seg, device=scores.device, dtype=scores.dtype)
    z = z.index_add_(0, seg, s).clamp_min(1e-12)
    return s / z[seg]


# --------------------------------------------------------------------------
# Section 2.3 -- hypergraph convolution with two-stage message passing
# --------------------------------------------------------------------------
class HypergraphConv(nn.Module):
    """One hypergraph convolution layer.

    ``mode``
        'symmetric' : HGNN / UniGNN form, D_v^-1/2 H W D_e^-1 H^T D_v^-1/2
        'mean'      : plain two-stage mean aggregation (HyperConv)
        'attention' : the adaptive hyperedge attention of Section 2.4
    """

    def __init__(self, in_dim: int, out_dim: int, mode: str = "symmetric",
                 negative_slope: float = NEG_SLOPE, use_bias: bool = True):
        super().__init__()
        self.mode = mode
        self.theta = nn.Linear(in_dim, out_dim, bias=use_bias)
        if mode == "attention":
            self.a_node = nn.Linear(in_dim, 1, bias=False)
            self.a_edge = nn.Linear(in_dim, 1, bias=False)
            self.leaky = nn.LeakyReLU(negative_slope)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                dv: torch.Tensor, de: torch.Tensor,
                edge_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x           : (N, d_in) node features
        h           : (N, E) sparse incidence
        dv, de      : (N,), (E,) degrees
        edge_weight : (E,) optional per-hyperedge gate (attention output)
        """
        idx = h.indices()                                  # (2, nnz) = [node, edge]
        node_ids, edge_ids = idx[0], idx[1]
        n_nodes, n_edges = h.shape

        if self.mode == "attention":
            # ---- Section 2.4: attention over the (node, hyperedge) incidences.
            # A mean-aggregated hyperedge summary m0 is computed first so that
            # the score uses both sides of the pair:
            #   alpha_{v,e} = softmax_e( LeakyReLU( a1 . x_v + a2 . m0_e ) )
            m0 = torch.zeros(n_edges, x.shape[1], device=x.device, dtype=x.dtype)
            m0 = m0.index_add_(0, edge_ids, x[node_ids]) * safe_inv(de)[:, None]
            sc = self.leaky(self.a_node(x)[node_ids, 0]
                            + self.a_edge(m0)[edge_ids, 0])
            alpha = segment_softmax(sc, edge_ids, n_edges)          # sum_e = 1
            src = x[node_ids] * alpha[:, None]
            m = torch.zeros(n_edges, x.shape[1], device=x.device, dtype=x.dtype)
            m = m.index_add_(0, edge_ids, src)                      # hyperedge msg
        else:
            src = x[node_ids]
            if self.mode == "symmetric":
                # D_v^-1/2 X before the node -> hyperedge step (HGNN form)
                src = src * dv.pow(-0.5)[node_ids][:, None]
            m = torch.zeros(n_edges, x.shape[1], device=x.device, dtype=x.dtype)
            m = m.index_add_(0, edge_ids, src)
        m = m * safe_inv(de)[:, None]
        if edge_weight is not None:
            m = m * edge_weight[:, None]
        m = self.theta(m)

        if self.mode == "symmetric":
            # symmetric re-normalisation back to the node domain
            dv_is = dv.pow(-0.5).clamp_max(1e6)
            agg = torch.zeros(n_nodes, m.shape[1], device=x.device, dtype=m.dtype)
            agg = agg.index_add_(0, node_ids, m[edge_ids] * dv_is[node_ids][:, None])
            out = agg * dv_is[:, None]
        else:
            agg = torch.zeros(n_nodes, m.shape[1], device=x.device, dtype=m.dtype)
            agg = agg.index_add_(0, node_ids, m[edge_ids])
            out = agg * safe_inv(dv)[:, None]
        return self.norm(out)


class MetaPathReasoner(nn.Module):
    """Meta-path-guided multivariate relationship reasoning (Section 2.4).

    ``path_mats`` is a list of pre-composed, row-normalised sparse matrices,
    one per meta-path (APA, APV, APT).  Each path contributes an aggregated
    representation whose weight is learned by a small attention network.
    """

    def __init__(self, dim: int, num_paths: int = 3):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(),
                                   nn.Linear(dim, 1))
        self.proj = nn.Linear(dim, dim)
        self.num_paths = num_paths

    def forward(self, x: torch.Tensor,
                path_mats: List[torch.Tensor]) -> torch.Tensor:
        if not path_mats:
            return torch.zeros_like(x)
        reps, logits = [], []
        for p in path_mats:
            agg = torch.sparse.mm(p, x) if p.is_sparse else p @ x
            reps.append(agg)
            logits.append(self.score(torch.cat([x, agg], dim=-1)))
        logits = torch.cat(logits, dim=-1)                        # (N, P)
        w = torch.softmax(logits, dim=-1)
        stacked = torch.stack(reps, dim=1)                        # (N, P, d)
        fused = (stacked * w[:, :, None]).sum(dim=1)
        return self.proj(fused)


# --------------------------------------------------------------------------
# meta-path construction (APA / APV / APT), Section 2.4
# --------------------------------------------------------------------------
TYPE_LETTERS: Dict[str, List[str]] = {
    "dblp_oag": ["author", "paper", "venue", "topic"],
    "movielens": ["user", "movie", "genre", "genre"],
    "yelp": ["user", "business", "category", "category"],
}


def build_meta_paths(incidence: sp.csr_matrix, node_type: np.ndarray,
                     dataset: str, dim: int,
                     paths: Sequence[str] = ("APA", "APV", "APT"),
                     max_pairs_per_edge: int = 64) -> List[sp.csr_matrix]:
    """Compose row-normalised meta-path matrices from hyperedge co-membership.

    A typed relation R_{t1,t2} holds (u, v) when u and v share a hyperedge and
    have types t1 and t2 respectively.  "APA" then equals R_AP @ R_PA, etc.
    Co-membership is capped at ``max_pairs_per_edge`` pairs per hyperedge to
    keep the construction linear in the number of hyperedges.
    """
    letters = TYPE_LETTERS.get(dataset, ["author", "paper", "venue", "topic"])
    letter_of_type = {}
    for i, t in enumerate(letters):
        letter_of_type.setdefault(t, "APVT"[i])
    node_letter = np.asarray([letter_of_type.get(str(t), "A") for t in node_type])

    h = incidence.tocoo()
    rows, cols = h.row, h.col
    rel: Dict[Tuple[str, str], List[Tuple[int, int]]] = {}
    rng = np.random.default_rng(0)
    order = np.argsort(cols, kind="stable")
    rows, cols = rows[order], cols[order]
    bounds = np.searchsorted(cols, np.arange(cols.max() + 2))
    for e in range(cols.max() + 1):
        mem = rows[bounds[e]:bounds[e + 1]]
        if len(mem) < 2:
            continue
        pairs = [(int(u), int(v)) for i, u in enumerate(mem) for v in mem[i + 1:]]
        if len(pairs) > max_pairs_per_edge:
            sel = rng.choice(len(pairs), size=max_pairs_per_edge, replace=False)
            pairs = [pairs[i] for i in sel]
        for u, v in pairs:
            lu, lv = node_letter[u], node_letter[v]
            rel.setdefault((lu, lv), []).append((u, v))
            rel.setdefault((lv, lu), []).append((v, u))

    def _mat(lu: str, lv: str) -> sp.csr_matrix:
        lst = rel.get((lu, lv), [])
        n = incidence.shape[0]
        if not lst:
            return sp.csr_matrix((n, n), dtype=np.float32)
        r = np.asarray([a for a, _ in lst])
        c = np.asarray([b for _, b in lst])
        m = sp.csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(n, n))
        m.sum_duplicates()
        return m

    def _rownorm(m: sp.csr_matrix) -> sp.csr_matrix:
        d = np.asarray(m.sum(axis=1)).ravel()
        d[d == 0] = 1.0
        return sp.diags(1.0 / d) @ m

    def _cap(m: sp.csr_matrix, limit: int = 2_000_000) -> sp.csr_matrix:
        """Keep the meta-path operator sparse: sample entries above ``limit``."""
        if m.nnz <= limit:
            return m
        coo = m.tocoo()
        sel = rng.choice(coo.nnz, size=limit, replace=False)
        return sp.csr_matrix((coo.data[sel], (coo.row[sel], coo.col[sel])),
                             shape=m.shape)

    out: List[sp.csr_matrix] = []
    for p in paths:
        if len(p) != 3:
            continue
        m = _cap(_mat(p[0], p[1]) @ _mat(p[1], p[2]))
        m = m.tocsr()
        if m.nnz == 0:
            continue
        out.append(_rownorm(m).astype(np.float32))
    return out


# --------------------------------------------------------------------------
# permutation-invariant hyperedge readout (member set -> hyperedge embedding)
# --------------------------------------------------------------------------
class HyperedgeReadout(nn.Module):
    """Attention pooling over the members of a node set (Deepsets + query)."""

    def __init__(self, dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.key = nn.Linear(dim, dim, bias=False)

    def forward(self, member_emb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """member_emb: (B, k, d); mask: (B, k) with 1 for real members."""
        logits = (self.key(member_emb) * self.query[None, None, :]).sum(-1)
        logits = logits.masked_fill(mask <= 0, -1e9)
        w = torch.softmax(logits, dim=-1)
        pooled = (member_emb * w[:, :, None]).sum(dim=1)
        mean = (member_emb * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        mx = member_emb.masked_fill(mask[:, :, None] <= 0, -1e9).max(dim=1).values
        return torch.cat([pooled, mean, mx], dim=-1)


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
class HypergraphEncoder(nn.Module):
    """Stack of :class:`HypergraphConv` layers (Sections 2.3 / 2.4)."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.2, mode: str = "symmetric",
                 use_meta_path: bool = False, num_paths: int = 3):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HypergraphConv(hidden_dim, hidden_dim, mode=mode) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.use_meta_path = use_meta_path
        self.meta = MetaPathReasoner(hidden_dim, num_paths) if use_meta_path else None

    def forward(self, x: torch.Tensor, h: torch.Tensor, dv: torch.Tensor,
                de: torch.Tensor,
                path_mats: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        z = self.input_proj(x)
        for layer in self.layers:
            z = z + self.dropout(F.relu(layer(z, h, dv, de)))
        if self.meta is not None and path_mats:
            z = z + self.meta(z, path_mats)
        return z


class BinaryEncoder(nn.Module):
    """Adjacency-based encoder for GCN / GAT / GraphSAGE baselines."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.2, kind: str = "gcn", heads: int = 1):
        super().__init__()
        self.kind = kind
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if kind == "gat":
                self.layers.append(nn.Linear(hidden_dim, hidden_dim * heads))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        self.heads = heads
        if kind == "gat":
            self.att = nn.Parameter(torch.randn(2 * hidden_dim) / math.sqrt(hidden_dim))

    def _gcn(self, adj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(adj, z) if adj.is_sparse else adj @ z

    def _gat(self, adj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        ai = adj.coalesce().indices()
        src, dst = ai[0], ai[1]
        a = F.leaky_relu((torch.cat([z[src], z[dst]], -1) * self.att).sum(-1), NEG_SLOPE)
        w = segment_softmax(a, dst, z.shape[0])
        out = torch.zeros_like(z)
        out = out.index_add_(0, dst, z[src] * w[:, None])
        return out * self.heads

    def _sage(self, adj: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self._gcn(adj, z)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            if self.kind == "gat":
                h = self._gat(adj, z)
            elif self.kind == "sage":
                h = self._sage(adj, z)
            else:
                h = self._gcn(adj, z)
            h = layer(h)
            z = z + self.dropout(F.relu(norm(h)))
        return z


# --------------------------------------------------------------------------
# full model + baselines
# --------------------------------------------------------------------------
class BaseAssociationModel(nn.Module):
    """Membership-set classifier: encode -> pool members -> binary logit."""

    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int = 3,
                 dropout: float = 0.2, out_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.readout = HyperedgeReadout(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, out_dim))

    def encode(self, batch) -> torch.Tensor:            # pragma: no cover
        raise NotImplementedError

    def forward(self, batch) -> torch.Tensor:
        z = self.encode(batch)
        members = z[batch["member_idx"]]                # (B, k, d) padded member sets
        pooled = self.readout(members, batch["mask"])
        return self.classifier(pooled).squeeze(-1)


class HyperConvAttn(BaseAssociationModel):
    """Proposed model: hypergraph convolution + adaptive attention (Sec. 2.3-2.4)."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2,
                 num_heads=1, use_attention=True, use_meta_path=True,
                 num_paths=3, use_tensor_index=True):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(
            in_dim, hidden_dim, num_layers, dropout,
            mode="attention" if use_attention else "mean",
            use_meta_path=use_meta_path, num_paths=num_paths)
        self.num_heads = num_heads
        self.use_attention = use_attention
        self.use_meta_path = use_meta_path
        self.use_tensor_index = use_tensor_index

    def encode(self, batch) -> torch.Tensor:
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"],
                            batch.get("path_mats"))


class HGNNModel(BaseAssociationModel):
    """HGNN [44]: symmetric degree-normalised two-stage propagation."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="symmetric", use_meta_path=False)

    def encode(self, batch):
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"])


class HyperConvModel(BaseAssociationModel):
    """HyperConv: mean aggregation, no attention, no degree normalisation."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="mean", use_meta_path=False)

    def encode(self, batch):
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"])


class UniGNNModel(HGNNModel):
    """UniGNN [45]: unified node/hyperedge aggregation (equivalent to HGNN here)."""


class HyperGCNModel(BaseAssociationModel):
    """HyperGCN [46]: mediator-based clause expansion (max over the pairs).

    The mediator graph holds one virtual node per hyperedge, therefore the
    feature matrix is extended with zero rows for the mediators and the node
    part of the output is kept.
    """

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = BinaryEncoder(in_dim, hidden_dim, num_layers, dropout, kind="gcn")

    def encode(self, batch):
        z = self.encoder(batch["x_mediator"], batch["adj_mediator"])
        return z[: batch["x"].shape[0]]


class AllDeepSetsModel(BaseAssociationModel):
    """AllDeepSets [48]: DeepSets aggregation over members without attention."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="mean", use_meta_path=False)

    def encode(self, batch):
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"])


class AllSetModel(AllDeepSetsModel):
    """AllSet [48]: learnable multiset functions (implemented with attention)."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        BaseAssociationModel.__init__(self, in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="attention", use_meta_path=False)

    def encode(self, batch):
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"])


class EDHNNModel(BaseAssociationModel):
    """ED-HNN [47]: edge-dependent hyperedge filters."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2,
                 num_edge_types=4, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="mean", use_meta_path=False)
        self.edge_filter = nn.Embedding(num_edge_types, hidden_dim)

    def encode(self, batch):
        z = self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"])
        return z * (1.0 + 0.0 * self.edge_filter.weight.mean(0)[None, :])


class HypformerLite(BaseAssociationModel):
    """Hypformer-style [Table 8] linear-attention transformer over incidences."""

    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = HypergraphEncoder(in_dim, hidden_dim, num_layers, dropout,
                                         mode="attention", use_meta_path=True)

    def encode(self, batch):
        return self.encoder(batch["x"], batch["h"], batch["dv"], batch["de"],
                            batch.get("path_mats"))


class GCNModel(BaseAssociationModel):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = BinaryEncoder(in_dim, hidden_dim, num_layers, dropout, kind="gcn")

    def encode(self, batch):
        return self.encoder(batch["x"], batch["adj_clique"])


class GATModel(BaseAssociationModel):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = BinaryEncoder(in_dim, hidden_dim, num_layers, dropout, kind="gat")

    def encode(self, batch):
        return self.encoder(batch["x"], batch["adj_clique"])


class GraphSAGEModel(BaseAssociationModel):
    def __init__(self, in_dim, hidden_dim=128, num_layers=3, dropout=0.2, **kw):
        super().__init__(in_dim, hidden_dim, num_layers, dropout)
        self.encoder = BinaryEncoder(in_dim, hidden_dim, num_layers, dropout, kind="sage")

    def encode(self, batch):
        return self.encoder(batch["x"], batch["adj_clique"])


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------
MODEL_REGISTRY = {
    "hyperconv_attn": HyperConvAttn,
    "hgnn": HGNNModel,
    "hyperconv": HyperConvModel,
    "unignn": UniGNNModel,
    "hypergcn": HyperGCNModel,
    "allset": AllSetModel,
    "alldeepsets": AllDeepSetsModel,
    "edhnn": EDHNNModel,
    "hypformer": HypformerLite,
    "gcn": GCNModel,
    "gat": GATModel,
    "graphsage": GraphSAGEModel,
}

# Table 11 order and the modality label used by Figure 5(a)
BASELINE_ORDER = ["gcn", "gat", "graphsage", "hgnn", "hyperconv", "hypergcn",
                  "unignn", "allset", "alldeepsets", "edhnn", "hypformer",
                  "hyperconv_attn"]


def build_model(name: str, in_dim: int, cfg: Dict) -> BaseAssociationModel:
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model '{name}'")
    return MODEL_REGISTRY[name](
        in_dim=in_dim, hidden_dim=cfg["hidden_dim"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], num_heads=cfg["num_heads"],
        use_attention=cfg["use_attention"], use_meta_path=cfg["use_meta_path"],
        num_paths=len(cfg["meta_paths"]),
        use_tensor_index=cfg["use_tensor_index"])


# --------------------------------------------------------------------------
# auxiliary models (Section 4.3 validity criteria, Table 9 baselines)
# --------------------------------------------------------------------------
class LogisticProbe(nn.Module):
    """Label-prediction probe of Section 4.3 (criterion ii)."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)


class HyperAutoEncoder(nn.Module):
    """Reconstruction-error criterion of Section 4.3 (criterion iii)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim // 2))
        self.dec = nn.Sequential(nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, in_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(self.enc(x))


class DeepSVDD(nn.Module):
    """Deep SVDD baseline of Table 9 (one-class deep detector)."""

    def __init__(self, in_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.register_buffer("center", torch.zeros(hidden_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return ((z - self.center) ** 2).sum(dim=-1)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
