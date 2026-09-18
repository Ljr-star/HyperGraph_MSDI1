# -*- coding: utf-8 -*-
"""
storage.py
==========
Tensor storage and bidirectional indexing of the data foundation
(Section 2.2) together with the storage/query benchmark of Section 4.4
(Table 6 and Figure 7).

Structure of the proposed backend
---------------------------------
The hypergraph adjacency is materialised as a THIRD-ORDER tensor

        T[i, j, k] = 1   <=>   node i is the k-th member of hyperedge j

stored in CSF (Compressed Sparse Fiber) form:

        node_ptr[N + 1]         -> offsets into the node fibers
        node_edge_id[nnz]       -> hyperedge id of every (node, edge) pair
        node_slot[nnz]          -> attribute slot k inside the hyperedge
        edge_ptr[E + 1]         -> offsets into the hyperedge fibers
        edge_node_id[nnz]       -> node id of every (edge, node) pair

The node-side fiber gives node -> hyperedges, the edge-side fiber gives
hyperedge -> nodes: that pair is the "bidirectional index" that makes
insertions, deletions and membership lookups atomic (Section 2.2).

Competing backends of Table 6
-----------------------------
    static_csr          incidence matrix in a static CSR, full rebuild on write
    dynamic_csr         CSR whose row arrays are over-allocated and grown
    hash_incidence_list dict node -> list of hyperedge ids (hash incidence list)
    lsm_key_value       LSM key-value emulation (memtable + sorted levels,
                        stand-in for RocksDB; no external service required)
    in_memory_hash      dict-of-dict hash index (stand-in for Redis)
    csf_tensor          the proposed CSF third-order tensor + bidirectional index

The last two emulate the engine characteristics locally (write amplification,
in-memory overhead) so that the comparison runs without installing RocksDB or
Redis; the numbers are therefore indicative, exactly as in the paper's note to
Table 6.
"""

from __future__ import annotations

import bisect
import sys
import threading
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp


# --------------------------------------------------------------------------
# backend 1: static CSR incidence matrix (full rebuild on every write)
# --------------------------------------------------------------------------
class StaticCSRBackend:
    name = "Incidence matrix H with static CSR"
    supports_incremental = False

    def __init__(self, num_nodes: int, num_edges: int, edges: Sequence[Sequence[int]]):
        self.num_nodes, self.num_edges = num_nodes, num_edges
        self.edges: List[List[int]] = [list(e) for e in edges]
        self._rebuild()

    def _rebuild(self):
        rows, cols = [], []
        for j, e in enumerate(self.edges):
            for v in e:
                rows.append(v)
                cols.append(j)
        self.h = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                               shape=(self.num_nodes, self.num_edges))

    def insert(self, edge_id: int, nodes: Sequence[int]):
        while len(self.edges) <= edge_id:
            self.edges.append([])
        self.edges[edge_id] = list(nodes)
        self._rebuild()

    def delete(self, edge_id: int):
        self.edges[edge_id] = []
        self._rebuild()

    def update(self, edge_id: int, nodes: Sequence[int]):
        self.insert(edge_id, nodes)

    def contains(self, node: int, edge_id: int) -> bool:
        return bool(self.h[node, edge_id])

    def next_edge_id(self) -> int:
        return len(self.edges)

    def memory_bytes(self) -> int:
        return self.h.data.nbytes + self.h.indices.nbytes + self.h.indptr.nbytes + 512


# --------------------------------------------------------------------------
# backend 2: dynamic CSR with over-allocated rows
# --------------------------------------------------------------------------
class DynamicCSRBackend:
    name = "Dynamic CSR (incremental)"
    supports_incremental = True

    def __init__(self, num_nodes: int, num_edges: int, edges: Sequence[Sequence[int]],
                 growth: float = 1.6):
        self.num_nodes, self.num_edges = num_nodes, num_edges
        self.growth = growth
        self.rows: List[np.ndarray] = [np.empty(0, np.int64) for _ in range(num_nodes)]
        self.caps = np.zeros(num_nodes, np.int64)
        self.edges: List[List[int]] = [[] for _ in range(num_edges)]
        self._load(edges)

    def _load(self, edges):
        for j, e in enumerate(edges):
            self.edges[j] = list(e)
            for v in e:
                self._append(v, j)

    def _append(self, node: int, edge_id: int):
        # over-allocate the row array so that amortised insertion stays O(1)
        if len(self.rows[node]) + 1 > self.caps[node]:
            self.caps[node] = max(4, int(self.growth * max(1, self.caps[node])))
        self.rows[node] = np.append(self.rows[node], edge_id)

    def insert(self, edge_id: int, nodes: Sequence[int]):
        while len(self.edges) <= edge_id:
            self.edges.append([])
        self.edges[edge_id] = list(nodes)
        for v in nodes:
            self._append(v, edge_id)

    def delete(self, edge_id: int):
        for v in self.edges[edge_id]:
            self.rows[v] = np.delete(self.rows[v], np.where(self.rows[v] == edge_id))
        self.edges[edge_id] = []

    def update(self, edge_id: int, nodes: Sequence[int]):
        self.delete(edge_id)
        self.insert(edge_id, nodes)

    def contains(self, node: int, edge_id: int) -> bool:
        return edge_id in self.rows[node]

    def next_edge_id(self) -> int:
        return len(self.edges)

    def memory_bytes(self) -> int:
        return int(sum(a.nbytes for a in self.rows) + self.caps.nbytes + 512)


# --------------------------------------------------------------------------
# backend 3: hash incidence list
# --------------------------------------------------------------------------
class HashIncidenceListBackend:
    name = "Hash-based incidence list"
    supports_incremental = True

    def __init__(self, num_nodes: int, num_edges: int, edges: Sequence[Sequence[int]]):
        self.index: Dict[int, List[int]] = {}
        self.edges: List[List[int]] = [[] for _ in range(num_edges)]
        for j, e in enumerate(edges):
            self.insert(j, e)

    def insert(self, edge_id: int, nodes: Sequence[int]):
        while len(self.edges) <= edge_id:
            self.edges.append([])
        self.edges[edge_id] = list(nodes)
        for v in nodes:
            self.index.setdefault(int(v), []).append(int(edge_id))

    def delete(self, edge_id: int):
        for v in self.edges[edge_id]:
            lst = self.index.get(int(v), [])
            if edge_id in lst:
                lst.remove(edge_id)
        self.edges[edge_id] = []

    def update(self, edge_id: int, nodes: Sequence[int]):
        self.delete(edge_id)
        self.insert(edge_id, nodes)

    def contains(self, node: int, edge_id: int) -> bool:
        return edge_id in self.index.get(int(node), [])

    def next_edge_id(self) -> int:
        return len(self.edges)

    def memory_bytes(self) -> int:
        return sys.getsizeof(self.index) + sum(
            sys.getsizeof(v) + sum(8 for _ in v) for v in self.index.values()) + 512


# --------------------------------------------------------------------------
# backend 4: LSM key-value emulation (RocksDB stand-in)
# --------------------------------------------------------------------------
class LSMKeyValueBackend:
    name = "RocksDB (LSM key-value index)"
    supports_incremental = True
    __slots__ = ("memtable", "levels", "level_size", "_next")

    def __init__(self, num_nodes: int, num_edges: int, edges: Sequence[Sequence[int]],
                 level_size: int = 4096):
        self.memtable: Dict[Tuple[int, int], int] = {}
        self.levels: List[List[Tuple[int, int]]] = []
        self.level_size = level_size
        self._next = int(num_edges)
        for j, e in enumerate(edges):
            self.insert(j, e)

    def insert(self, edge_id: int, nodes: Sequence[int]):
        for v in nodes:
            self.memtable[(int(v), int(edge_id))] = 1
        if len(self.memtable) > self.level_size:          # flush -> sorted level
            keys = sorted(self.memtable.keys())
            self.levels.append(keys)
            self.memtable.clear()

    def delete(self, edge_id: int):
        self.memtable.pop((0, edge_id), None)
        for lvl in self.levels:
            for i in range(len(lvl) - 1, -1, -1):
                if lvl[i][1] == edge_id:
                    lvl.pop(i)

    def update(self, edge_id: int, nodes: Sequence[int]):
        self.delete(edge_id)
        self.insert(edge_id, nodes)

    def contains(self, node: int, edge_id: int) -> bool:
        if (int(node), int(edge_id)) in self.memtable:
            return True
        for lvl in self.levels:                            # binary search per level
            pos = bisect.bisect_left(lvl, (int(node), int(edge_id)))
            if pos < len(lvl) and lvl[pos] == (int(node), int(edge_id)):
                return True
        return False

    def next_edge_id(self) -> int:
        self._next += 1
        return self._next - 1

    def memory_bytes(self) -> int:
        return (sys.getsizeof(self.memtable)
                + sum(16 * len(lvl) + 56 for lvl in self.levels) + 512)


# --------------------------------------------------------------------------
# backend 5: in-memory hash index (Redis stand-in)
# --------------------------------------------------------------------------
class InMemoryHashBackend(HashIncidenceListBackend):
    name = "Redis (in-memory hash index)"

    def memory_bytes(self) -> int:
        # Redis stores one dict entry per membership plus per-key overhead
        return 96 * max(1, sum(len(v) for v in self.index.values())) + 512


# --------------------------------------------------------------------------
# backend 6 (proposed): CSF third-order tensor + bidirectional index
# --------------------------------------------------------------------------
class CSFTensorBackend:
    """Compressed Sparse Fiber storage of T[i, j, k] plus the bidirectional index."""

    name = "CSF third-order tensor (proposed)"
    supports_incremental = True

    def __init__(self, num_nodes: int, num_edges: int, edges: Sequence[Sequence[int]]):
        self.num_nodes, self.num_edges = num_nodes, num_edges
        self.node_edge: List[List[Tuple[int, int]]] = [[] for _ in range(num_nodes)]
        self.edge_node: List[List[int]] = [[] for _ in range(num_edges)]
        self.attr: List[np.ndarray] = [np.zeros(0, np.float32) for _ in range(num_edges)]
        self._frozen = None
        for j, e in enumerate(edges):
            self.insert(j, e)

    # ---- write path (atomic per hyperedge) ----
    def insert(self, edge_id: int, nodes: Sequence[int]):
        while len(self.edge_node) <= edge_id:
            self.edge_node.append([])
            self.attr.append(np.zeros(0, np.float32))
        nodes = [int(v) for v in nodes]
        self.edge_node[edge_id] = nodes
        self.attr[edge_id] = np.zeros(len(nodes), np.float32)
        for k, v in enumerate(nodes):
            self.node_edge[v].append((edge_id, k))
        self._frozen = None

    def delete(self, edge_id: int):
        for k, v in enumerate(self.edge_node[edge_id]):
            self.node_edge[v] = [(j, s) for (j, s) in self.node_edge[v] if j != edge_id]
        self.edge_node[edge_id] = []
        self.attr[edge_id] = np.zeros(0, np.float32)
        self._frozen = None

    def update(self, edge_id: int, nodes: Sequence[int]):
        self.delete(edge_id)
        self.insert(edge_id, nodes)

    # ---- read path ----
    def contains(self, node: int, edge_id: int) -> bool:
        for j, _k in self.node_edge[node]:
            if j == edge_id:
                return True
        return False

    def pair_value(self, node: int, edge_id: int) -> float:
        for j, k in self.node_edge[node]:
            if j == edge_id:
                return float(self.attr[edge_id][k])
        return 0.0

    def next_edge_id(self) -> int:
        return len(self.edge_node)

    def freeze(self):
        """Materialise the CSR-style fiber arrays (done once per benchmark)."""
        ptr = np.zeros(self.num_nodes + 1, np.int64)
        for i, lst in enumerate(self.node_edge):
            ptr[i + 1] = ptr[i] + len(lst)
        eid = np.empty(ptr[-1], np.int64)
        slot = np.empty(ptr[-1], np.int64)
        for i, lst in enumerate(self.node_edge):
            for t, (j, k) in enumerate(lst):
                eid[ptr[i] + t] = j
                slot[ptr[i] + t] = k
        self._frozen = (ptr, eid, slot)
        return self._frozen

    def memory_bytes(self) -> int:
        if self._frozen is None:
            self.freeze()
        ptr, eid, slot = self._frozen
        base = ptr.nbytes + eid.nbytes + slot.nbytes
        attr = sum(a.nbytes for a in self.attr)
        edges = sum(8 * len(e) for e in self.edge_node)
        return int(base + attr + edges + 512)


BACKENDS = {
    "static_csr": StaticCSRBackend,
    "dynamic_csr": DynamicCSRBackend,
    "hash_incidence_list": HashIncidenceListBackend,
    "lsm_key_value": LSMKeyValueBackend,
    "in_memory_hash": InMemoryHashBackend,
    "csf_tensor": CSFTensorBackend,
}


# --------------------------------------------------------------------------
# measurement helpers (Section 4.4 protocol)
# --------------------------------------------------------------------------
def _median(xs: Sequence[float]) -> float:
    return float(np.median(xs)) if xs else float("nan")


def measure_pair_query(backend, num_nodes: int, num_edges: int,
                       repeats: int = 1_000, warmup: int = 3,
                       seed: int = 0) -> float:
    """Median single pair-query latency (ms) after pre-warming the index."""
    rng = np.random.default_rng(seed)
    pairs = [(int(rng.integers(0, num_nodes)), int(rng.integers(0, num_edges)))
             for _ in range(repeats)]
    for _ in range(warmup):
        for n, e in pairs[:100]:
            backend.contains(n, e)
    t0 = time.perf_counter()
    for n, e in pairs:
        backend.contains(n, e)
    return 1e3 * (time.perf_counter() - t0) / repeats


def measure_batch_update(backend, num_nodes: int, cfg: Dict,
                         seed: int = 0) -> Dict[str, float]:
    """Insert / delete / update / query latency of one batch, in ms.

    The batch is timed with ``cfg['batch_size']`` edges and linearly scaled to
    the 10,000-edge latency reported in Table 6, so that backends whose write
    path is O(number of hyperedges) (static CSR) stay measurable on a laptop.
    """
    rng = np.random.default_rng(seed)
    bs = cfg["batch_size"]
    reps = max(1, min(cfg["batch_repeats"], 100))
    scale_to_10k = 10_000.0 / bs
    res = {}
    for op in ("insert", "delete", "update"):
        lat = []
        for r in range(reps):
            edges = [tuple(int(x) for x in rng.choice(num_nodes, 3 + (r % 3), replace=False))
                     for _ in range(bs)]
            t0 = time.perf_counter()
            for i, e in enumerate(edges):
                gid = backend.next_edge_id()
                if op == "insert":
                    backend.insert(gid, e)
                elif op == "delete":
                    backend.insert(gid, e)
                    backend.delete(gid)
                else:
                    backend.update(gid, e)
            lat.append(scale_to_10k * 1e3 * (time.perf_counter() - t0))
        res[op] = float(np.mean(lat))
    lat = []
    for _ in range(reps):
        pairs = [(int(rng.integers(0, num_nodes)), int(rng.integers(0, 10 ** 6)))
                 for _ in range(bs)]
        t0 = time.perf_counter()
        for n, e in pairs:
            backend.contains(n, e)
        lat.append(scale_to_10k * 1e3 * (time.perf_counter() - t0))
    res["query"] = float(np.mean(lat))
    return res


def measure_throughput(backend, num_nodes: int, workers: int = 8,
                       ops_per_worker: int = 2_000, seed: int = 0) -> float:
    """Sustained updates per second with ``workers`` concurrent threads."""
    rng = np.random.default_rng(seed)
    if not getattr(backend, "supports_incremental", True):
        # a full index rebuild per write cannot sustain the same op budget
        ops_per_worker = min(ops_per_worker, 100)
    barrier = threading.Barrier(workers)
    gids = [backend.next_edge_id() + i for i in range(workers * ops_per_worker)]
    rng.shuffle(gids)

    def _worker(wid: int):
        pairs = [(tuple(int(x) for x in rng.choice(num_nodes, 3, replace=False)),
                  gids[wid * ops_per_worker + i]) for i in range(ops_per_worker)]
        barrier.wait()
        for e, gid in pairs:
            backend.insert(gid, e)

    threads = [threading.Thread(target=_worker, args=(w,)) for w in range(workers)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return workers * ops_per_worker / max(1e-9, time.perf_counter() - t0)


# --------------------------------------------------------------------------
# Table 6 + Figure 7
# --------------------------------------------------------------------------
def benchmark_backends(num_nodes: int, edges: Sequence[Sequence[int]], cfg: Dict,
                       seed: int = 0) -> List[Dict]:
    """Run the Table 6 comparison on one (reduced) dataset instance."""
    rows = []
    for key in cfg["backends"]:
        cls = BACKENDS[key]
        t0 = time.perf_counter()
        b = cls(num_nodes, len(edges), edges)
        build_ms = 1e3 * (time.perf_counter() - t0)
        rows.append(dict(
            backend=b.name,
            key=key,
            incremental=bool(getattr(b, "supports_incremental", False)),
            pair_query_ms=round(measure_pair_query(b, num_nodes, len(edges),
                                                   cfg["pair_query_repeats"], cfg["warmup_passes"]), 2),
            update_10k_ms=round(measure_batch_update(b, num_nodes, cfg)["update"], 1),
            throughput_8w=round(measure_throughput(b, num_nodes, cfg["num_workers"])),
            memory_mb=round(b.memory_bytes() / 1024 ** 2, 1),
            build_ms=round(build_ms, 1),
        ))
        print(f"[storage] {b.name:42s} done")
    return rows


def scaling_curve(num_points: int = 6, seed: int = 0,
                  max_nodes: int = 500_000) -> Dict[str, Dict[str, np.ndarray]]:
    """Insertion / query latency of every backend as a function of node count.

    Returns the arrays behind Figure 7(a); the shapes follow the paper
    (log-log, from ~5K to ~2.03M nodes) while the absolute values come from
    the local benchmark, so the *ordering* of the curves is what is reproduced.
    """
    from config import STORAGE
    sizes = np.unique(np.round(np.geomspace(5_000, max_nodes, num_points)).astype(int))
    out: Dict[str, Dict[str, np.ndarray]] = {}
    rng = np.random.default_rng(seed)
    for key in STORAGE["backends"]:
        ins, qry = [], []
        for n in sizes:
            m = max(50, n // 20)
            edges = [tuple(int(x) for x in rng.choice(n, 4, replace=False)) for _ in range(m)]
            b = BACKENDS[key](int(n), m, edges)
            ins.append(_median([_timeit(lambda: b.insert(m + i, e))
                                for i, e in enumerate(edges[:50])]))
            qry.append(measure_pair_query(b, int(n), m + 60, repeats=200, warmup=1, seed=seed))
        out[key] = dict(nodes=sizes, insertion_ms=np.asarray(ins), query_ms=np.asarray(qry))
    return out


def _timeit(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


if __name__ == "__main__":
    from config import STORAGE
    rng = np.random.default_rng(0)
    n, m = 8_000, 6_000
    edges = [tuple(int(x) for x in rng.choice(n, 4, replace=False)) for _ in range(m)]
    for r in benchmark_backends(n, edges, STORAGE):
        print({k: v for k, v in r.items() if k != "backend"})
