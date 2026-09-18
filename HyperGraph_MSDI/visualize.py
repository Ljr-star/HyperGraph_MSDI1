# -*- coding: utf-8 -*-
"""
visualize.py
============
Reproduction of the paper's figures.

    Figure 5(a) F1 of six methods across the three data modalities
    Figure 5(b) F1 vs. number of training hyperedges (fixed test set)
    Figure 6(a) HR@10 across single-modal / multi-modal / cross-temporal
    Figure 6(b) HR@K curves in the multi-modal scenario
    Figure 7(a) Insertion and query latency vs. node count (log-log)
    Figure 7(b) Batch operation latency distributions (violin + box)

Every figure is written to results/figures/<dataset>/<scale>/ in PNG and PDF.
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib
matplotlib.use("Agg")                       # headless-safe for servers / CI
import matplotlib.pyplot as plt
import numpy as np

from config import PLOT

plt.rcParams.update({
    "font.size": PLOT["font_size"],
    "axes.grid": True,
    "grid.alpha": 0.30,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
})

METHOD_LABELS = {
    "hyperconv_attn": "HyperConv-Attn (Ours)", "hgnn": "HGNN",
    "hyperconv": "HyperConv", "gat": "GAT", "gcn": "GCN",
    "graphsage": "GraphSAGE", "hypergcn": "HyperGCN", "unignn": "UniGNN",
    "allset": "AllSet", "alldeepsets": "AllDeepSets", "edhnn": "ED-HNN",
    "hypformer": "Hypformer",
}
MODALITY_LABELS = ["Text modality", "Time-series modality", "Cross-modal data"]


def _save(fig, path_no_ext: str) -> None:
    os.makedirs(os.path.dirname(path_no_ext), exist_ok=True)
    for fmt in PLOT["save_formats"]:
        fig.savefig(f"{path_no_ext}.{fmt}", dpi=PLOT["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[visualize] wrote {path_no_ext}.{{{','.join(PLOT['save_formats'])}}}")


# --------------------------------------------------------------------------
# Figure 5
# --------------------------------------------------------------------------
def figure5a(f1_by_method_modality: Dict[str, Sequence[float]], out_dir: str) -> None:
    """Grouped bar chart: F1 per method per modality (Figure 5(a))."""
    methods = list(f1_by_method_modality.keys())
    x = np.arange(len(MODALITY_LABELS))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, m in enumerate(methods):
        vals = f1_by_method_modality[m]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=METHOD_LABELS.get(m, m), color=PLOT["palette"][i % len(PLOT["palette"])])
    ax.set_xticks(x)
    ax.set_xticklabels(MODALITY_LABELS)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.75, 1.0)
    ax.set_title("F1 of different methods under different data modalities")
    ax.legend(ncol=2, fontsize=8)
    _save(fig, os.path.join(out_dir, "fig5a_modality_f1"))


def figure5b(curve: List[Dict], out_dir: str) -> None:
    """F1 (+ CI band) against the size of the training hyperedge set (Figure 5(b))."""
    x = np.asarray([c["train_size"] for c in curve], dtype=float)
    f1 = np.asarray([c["f1"] for c in curve], dtype=float)
    ci = np.asarray([c["ci_half_width"] for c in curve], dtype=float)

    fig, ax = plt.subplots(figsize=PLOT["figsize"])
    ax.plot(x, f1, "o-", color=PLOT["palette"][0], label="HyperConv-Attn")
    ax.fill_between(x, f1 - ci, f1 + ci, color=PLOT["palette"][0], alpha=0.18,
                    label="95% bootstrap CI")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training hyperedges")
    ax.set_ylabel("F1 score")
    ax.set_title("Impact of the number of training hyperedges")
    ax.legend()
    _save(fig, os.path.join(out_dir, "fig5b_learning_curve"))


# --------------------------------------------------------------------------
# Figure 6
# --------------------------------------------------------------------------
def figure6a(hr10_by_scenario: Dict[str, Dict[str, float]], out_dir: str) -> None:
    """HR@10 of several methods in single-modal / multi-modal / cross-temporal."""
    scenarios = list(next(iter(hr10_by_scenario.values())).keys())
    methods = list(hr10_by_scenario.keys())
    x = np.arange(len(scenarios))
    width = 0.8 / max(1, len(methods))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for i, m in enumerate(methods):
        vals = [hr10_by_scenario[m][s] for s in scenarios]
        ax.bar(x + i * width - 0.4 + width / 2, vals, width,
               label=METHOD_LABELS.get(m, m), color=PLOT["palette"][i % len(PLOT["palette"])])
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace("_", " ").title() for s in scenarios])
    ax.set_ylabel("HR@10")
    ax.set_title("HR@10 of different methods in different scenarios")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, "fig6a_hr10_scenarios"))


def figure6b(hr_curves: Dict[str, Dict[int, float]], out_dir: str) -> None:
    """HR@K curves in the multi-modal scenario (Figure 6(b))."""
    fig, ax = plt.subplots(figsize=PLOT["figsize"])
    for i, (m, curve) in enumerate(hr_curves.items()):
        ks = sorted(curve.keys())
        ax.plot(ks, [curve[k] for k in ks], "o-",
                label=METHOD_LABELS.get(m, m),
                color=PLOT["palette"][i % len(PLOT["palette"])], markersize=4)
    ax.set_xscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("HR@K")
    ax.set_title("HR@K variation curve in the multi-modal scenario")
    ax.legend(fontsize=8)
    _save(fig, os.path.join(out_dir, "fig6b_hrk_curves"))


# --------------------------------------------------------------------------
# Figure 7
# --------------------------------------------------------------------------
def figure7a(scaling: Dict[str, Dict[str, np.ndarray]], out_dir: str) -> None:
    """Insertion (solid) and query (dashed) latency vs node scale (Figure 7(a))."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for i, (key, d) in enumerate(scaling.items()):
        c = PLOT["palette"][i % len(PLOT["palette"])]
        ax.plot(d["nodes"], d["insertion_ms"], "-o", color=c, markersize=3,
                label=f"{_backend_label(key)} - insertion")
        ax.plot(d["nodes"], d["query_ms"], "--s", color=c, markersize=3,
                label=f"{_backend_label(key)} - query")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Number of nodes")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Insertion and query latency vs. node size")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, os.path.join(out_dir, "fig7a_storage_scaling"))


def _backend_label(key: str) -> str:
    return {
        "static_csr": "Static CSR", "dynamic_csr": "Dynamic CSR",
        "hash_incidence_list": "Hash incidence", "lsm_key_value": "LSM KV",
        "in_memory_hash": "In-memory hash", "csf_tensor": "Tensor+BiIndex (Ours)",
    }.get(key, key)


def figure7b(batch_latency: Dict[str, np.ndarray], out_dir: str) -> None:
    """Violin + box plot of batch insert / delete / update / query latency."""
    ops = list(batch_latency.keys())
    data = [np.asarray(batch_latency[o]) for o in ops]

    fig, ax = plt.subplots(figsize=PLOT["figsize"])
    parts = ax.violinplot(data, showmeans=True, showextrema=False, widths=0.75)
    for pc in parts["bodies"]:
        pc.set_facecolor(PLOT["palette"][0]); pc.set_alpha(0.35)
    ax.boxplot(data, widths=0.18, showfliers=False, patch_artist=False)
    ax.set_xticks(np.arange(1, len(ops) + 1))
    ax.set_xticklabels([o.replace("_", " ").title() for o in ops])
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Batch operation latency distribution")
    _save(fig, os.path.join(out_dir, "fig7b_batch_latency"))


if __name__ == "__main__":
    from config import build_argparser, get_config
    cfg = get_config(build_argparser().parse_args([]))
    demo = {"hyperconv_attn": [0.914, 0.900, 0.926], "hgnn": [0.871, 0.858, 0.883],
            "gcn": [0.834, 0.821, 0.834]}
    figure5a(demo, cfg.fig_dir)
    print("demo figures written to", cfg.fig_dir)
