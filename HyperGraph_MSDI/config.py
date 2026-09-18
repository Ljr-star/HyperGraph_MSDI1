# -*- coding: utf-8 -*-
"""
config.py
=========
Central configuration for the reference implementation of

    "Complex Multi-Source Data Association Mining and Data Foundation
     Construction Based on HyperGraph"

Every hyper-parameter used by the paper (Section 2 = method, Section 3 =
experimental setup, Section 4 = results) is declared here, so the whole
pipeline can be reproduced by editing this single file.

Run scaling
-----------
The paper reports a full-scale run with ~2.03M nodes and ~4.72M hyperedges.
Generating/training that on a laptop is impractical, therefore three presets
are provided and can be selected with ``--scale`` on main.py:

    small   : smoke test, seconds to a couple of minutes  (default)
    medium  : moderate run
    full    : paper scale (millions of hyperedges, needs a GPU)

All numbers that appear in the paper's tables are produced by
``evaluate.py`` / ``validity.py`` / ``storage.py`` and written to results/.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")
FIGURE_DIR = os.path.join(RESULTS_DIR, "figures")
TABLE_DIR = os.path.join(RESULTS_DIR, "tables")
for _d in (DATA_DIR, RESULTS_DIR, FIGURE_DIR, TABLE_DIR):
    os.makedirs(_d, exist_ok=True)

# --------------------------------------------------------------------------
# Scale presets.  "full" matches the magnitudes reported in the paper.
# --------------------------------------------------------------------------
SCALE_PRESETS: Dict[str, Dict[str, Any]] = {
    "small":  dict(node_scale=0.010, hyperedge_scale=0.008, epochs=40,
                   storage_points=6, learning_curve_points=5),
    "medium": dict(node_scale=0.050, hyperedge_scale=0.040, epochs=80,
                   storage_points=8, learning_curve_points=6),
    "full":   dict(node_scale=1.000, hyperedge_scale=1.000, epochs=150,
                   storage_points=10, learning_curve_points=7),
}

# --------------------------------------------------------------------------
# Datasets (Section 3.1).  Feature dimension, modality set and the
# natural-anomaly budget are taken from the paper.
# --------------------------------------------------------------------------
FEATURE_DIM = 128           # "128-dimensional aligned feature space" (Sec. 4.3)

DATASETS: Dict[str, Dict[str, Any]] = {
    "dblp_oag": dict(
        key="dblp_oag",
        display="DBLP-OAG",
        node_types=["author", "paper", "venue", "topic", "institution"],
        num_authors=1_180_000, num_papers=560_000, num_venues=18_000,
        num_topics=4_200, num_institutions=62_000,
        modalities=["text", "graph", "temporal"],
        target_hyperedges=4_720_000,
        target_nodes=2_030_000,
        cardinality_probs=[0.489, 0.223, 0.114, 0.174],   # 3 / 4 / 5 / >=6
        window_days=30,
        anomaly_per_family=2_100,
        # Sections 3.1(d): 1.42M / 0.20M / 0.41M nodes, 3.30M / 0.47M / 0.95M edges
        split_ratio=(0.70, 0.10, 0.20),
    ),
    "movielens": dict(
        key="movielens",
        display="MovieLens",
        node_types=["user", "movie", "genre"],
        num_users=162_000, num_movies=59_000, num_genres=20,
        num_venues=0, num_topics=0, num_institutions=0,
        num_authors=0, num_papers=0,
        modalities=["temporal", "graph"],
        target_hyperedges=1_350_000,
        target_nodes=221_000,
        cardinality_probs=[0.462, 0.241, 0.120, 0.177],
        window_days=30,
        anomaly_per_family=2_100,
        split_ratio=(0.70, 0.10, 0.20),
    ),
    "yelp": dict(
        key="yelp",
        display="Yelp (multi-modal)",
        node_types=["user", "business", "category"],
        num_users=780_000, num_businesses=150_000, num_genres=1_100,
        num_authors=0, num_papers=0, num_venues=0, num_topics=0,
        num_institutions=0,
        modalities=["text", "graph", "temporal"],
        target_hyperedges=2_300_000,
        target_nodes=1_100_000,
        cardinality_probs=[0.495, 0.228, 0.109, 0.168],
        window_days=30,
        anomaly_per_family=2_100,
        split_ratio=(0.70, 0.10, 0.20),
    ),
}

# --------------------------------------------------------------------------
# Dynamic hyperedge generation (Section 2.1)
#   quantile form  : tau = Q_p,  p selected on the validation split
#   IQR baseline   : tau = Q2 - lambda * (Q3 - Q1) + eta  (ablation only)
# --------------------------------------------------------------------------
THRESHOLD = dict(
    quantile_grid=[0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40],
    lambda_grid=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2],
    eta_grid=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    fixed_taus=[0.40, 0.50],          # Fixed-tau baselines of Table 3
    ema_alpha=0.25,                   # streaming threshold EMA (Sec. 4.6)
    min_cardinality=3,
    max_cardinality=10,
    neighbourhood_hops=2,             # 2-hop constraint of Sec. 3.1(a)
    # Table 4 reference values (papers' calibration on the training split)
    table4_reference={
        "dblp_oag": dict(sim_range=(0.11, 0.97), q1=0.41, med=0.55, q3=0.68,
                         iqr=0.27, p=0.22, tau=0.35,
                         iqr_lambda=0.82, iqr_eta=0.56, iqr_tau=0.89,
                         acceptance=0.784),
        "movielens": dict(sim_range=(0.08, 0.95), q1=0.38, med=0.52, q3=0.64,
                          iqr=0.26, p=0.25, tau=0.38,
                          iqr_lambda=0.85, iqr_eta=0.53, iqr_tau=0.83,
                          acceptance=0.749),
        "yelp": dict(sim_range=(0.13, 0.98), q1=0.43, med=0.57, q3=0.71,
                     iqr=0.28, p=0.20, tau=0.39,
                     iqr_lambda=0.80, iqr_eta=0.59, iqr_tau=0.94,
                     acceptance=0.796),
    },
)

# --------------------------------------------------------------------------
# Model (Sections 2.2 - 2.4)
# --------------------------------------------------------------------------
MODEL = dict(
    hidden_dim=128,          # feature dimension of the aligned space
    num_layers=3,            # paper: F1 peaks at 3 layers (Sec. 4.6)
    num_heads=1,             # single-head attention is the reported setting
    head_ablation=[1, 4, 8, 16],
    dropout=0.20,
    negative_slope=0.20,     # LeakyReLU in the attention score
    symmetric_norm=True,     # D_v^-1/2 H W D_e^-1 H^T D_v^-1/2  (HGNN form)
    meta_paths=["APA", "APV", "APT"],   # author-paper-author / -venue / -topic
    use_attention=True,
    use_meta_path=True,
    use_tensor_index=True,   # "w/o Tensor + BiIndex" ablation of Table 10
)

# --------------------------------------------------------------------------
# Training (Section 3.2 / 3.3)
# --------------------------------------------------------------------------
TRAIN = dict(
    epochs=40,
    lr=1.0e-3,
    weight_decay=1.0e-4,
    batch_size=512,
    patience=10,
    seeds=[0, 1, 2, 3, 4],      # 5 independent runs (Table 8, Table 12)
    train_ratio=0.70,
    val_ratio=0.10,
    test_ratio=0.20,
    labeled_test_positives=22_000,   # Sec. 3.1: 22k positive / 22k negative
    labeled_test_negatives=22_000,
    probe_folds=5,
    lr_curve_train_fractions=[0.10, 0.25, 0.50, 0.75, 1.00],
    device="cuda",               # falls back to cpu automatically
)

# --------------------------------------------------------------------------
# Evaluation protocols (Section 3.1 / Table 2)
# --------------------------------------------------------------------------
PROTOCOLS = dict(
    transductive=dict(
        label="transductive",
        node_sharing=True,
        split_by="timestamp",
    ),
    cold_start=dict(
        label="cold-start subset",
        node_sharing=False,
        fraction_of_test=0.05,      # 5% of the test hyperedges, 2,200 samples
        subset_of_test=True,
    ),
    strict_inductive=dict(
        label="strict inductive",
        node_sharing=False,
        remove_all_node_shared_edges=True,
    ),
)

# Headline numbers of the paper under the transductive DBLP-OAG setting
PAPER_HEADLINE = dict(protocol="transductive", dataset="dblp_oag",
                      f1=0.926, hr10=0.812, ap=0.918)

# --------------------------------------------------------------------------
# Hyperedge validity judgement (Section 4.3)
#   (i)   MI gain criterion : relative gain >= 5%
#   (ii)  logistic probe    : F1 gain >= 1.0 percentage point
#   (iii) auto-encoder      : reconstruction error >= 8% lower
#   conjunctive rule        : at least 2 of 3 criteria must hold
# --------------------------------------------------------------------------
VALIDITY = dict(
    knn_mi_k=5,                 # k-nearest-neighbour MI estimator, k = 5
    knn_mi_metric="chebyshev",  # maximum-norm distance
    # KSG cost grows quadratically with the sample count; the paper uses the
    # 22,000 positive hyperedges per dataset, the local default subsamples.
    knn_mi_max_samples=2_000,
    group_size=256,             # 1 hyperedge + 255 nearest neighbours
    judge_budget=24,            # hyperedges judged per generation strategy
    group_min_baseline_mi=1.0e-3,   # negligible-baseline guard (Sec. 4.3)
    mi_relative_gain=0.05,
    probe_f1_gain=0.010,
    ae_error_gain=0.08,
    conjunctive_min_votes=2,
    manual_check_size=2_000,    # manual verification subset per dataset
    ae_hidden_dim=64,
    ae_epochs=30,
    probe_max_iter=200,
    threshold_sweep=0.20,       # +-20% threshold sensitivity (Table 5 note)
)

# --------------------------------------------------------------------------
# Anomaly detection (Section 4.5 / Table 7 - Table 9)
# --------------------------------------------------------------------------
ANOMALY = dict(
    std_rule=2.0,               # bidirectional two-standard-deviation rule
    injected_per_dataset=4_700,# rule-based anomalies, 1:1:1 over three families
    injected_ratio_target=0.05, # overall anomaly rate ~5% of all test pairs
    injected_families=["node", "event", "cross_modal"],
    natural_families=[
        "temporally_inconsistent_collaboration",
        "cross_source_semantic_inconsistency",
        "attribute_confusion_entity_alignment",
    ],
    anomaly_rate_sweep=[0.01, 0.03, 0.05, 0.10],
)

# --------------------------------------------------------------------------
# Storage / indexing benchmark (Sections 2.2, 4.4 / Table 6 / Figure 7)
# --------------------------------------------------------------------------
STORAGE = dict(
    backends=["static_csr", "dynamic_csr", "hash_incidence_list",
              "lsm_key_value", "in_memory_hash", "csf_tensor"],
    num_workers=8,              # throughput measured with 8 concurrent workers
    pair_query_repeats=1_000,   # median over 1,000 pre-warmed queries
    # the local benchmark uses a 1,000-edge batch and scales it to the paper's
    # "10K-update latency" column; set batch_size = 10_000 for the real thing
    batch_size=1_000,
    bench_max_edges=5_000,      # size of the instance used for Table 6
    batch_repeats=3,            # average over repeated batches (100 in the paper)
    warmup_passes=3,
    node_sizes=[5_000, 20_000, 80_000, 320_000, 1_000_000, 2_030_000],
    # Table 6 reference values for the proposed CSF tensor
    table6_reference=dict(pair_query_ms=2.1, update_10k_ms=55.1,
                          throughput_8w=14_600, memory_mb=412),
)

# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
PLOT = dict(
    dpi=300,
    figsize=(6.0, 4.2),
    font_size=10,
    palette=["#2E5AAC", "#E4572E", "#17BECF", "#7F7F7F", "#BCBD22", "#9467BD"],
    save_formats=["png", "pdf"],
)


# --------------------------------------------------------------------------
# Run configuration helpers
# --------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Resolved configuration for one run of main.py."""

    dataset: str = "dblp_oag"
    scale: str = "small"
    seed: int = 0
    output_tag: str = "default"

    node_scale: float = 0.01
    hyperedge_scale: float = 0.008
    epochs: int = 40
    storage_points: int = 6
    learning_curve_points: int = 5

    model: Dict[str, Any] = field(default_factory=lambda: dict(MODEL))
    train: Dict[str, Any] = field(default_factory=lambda: dict(TRAIN))
    threshold: Dict[str, Any] = field(default_factory=lambda: dict(THRESHOLD))
    validity: Dict[str, Any] = field(default_factory=lambda: dict(VALIDITY))
    anomaly: Dict[str, Any] = field(default_factory=lambda: dict(ANOMALY))
    storage: Dict[str, Any] = field(default_factory=lambda: dict(STORAGE))
    dataset_spec: Dict[str, Any] = field(default_factory=dict)

    # ---------------- derived quantities ----------------
    @property
    def num_nodes(self) -> int:
        return max(2_000, int(self.dataset_spec.get("target_nodes", 20_000) * self.node_scale))

    @property
    def num_hyperedges(self) -> int:
        return max(4_000, int(self.dataset_spec.get("target_hyperedges", 50_000)
                              * self.hyperedge_scale))

    @property
    def feature_dim(self) -> int:
        return FEATURE_DIM

    @property
    def fig_dir(self) -> str:
        return os.path.join(FIGURE_DIR, self.dataset, self.scale)

    @property
    def table_dir(self) -> str:
        return os.path.join(TABLE_DIR, self.dataset, self.scale)

    @property
    def data_dir(self) -> str:
        return os.path.join(DATA_DIR, self.dataset, self.scale)

    def ensure_dirs(self) -> None:
        for d in (self.fig_dir, self.table_dir, self.data_dir):
            os.makedirs(d, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def get_config(args: argparse.Namespace) -> RunConfig:
    """Build a :class:`RunConfig` from parsed command line arguments."""
    preset = SCALE_PRESETS[args.scale]
    cfg = RunConfig(
        dataset=args.dataset,
        scale=args.scale,
        seed=args.seed,
        output_tag=args.tag,
        node_scale=preset["node_scale"],
        hyperedge_scale=preset["hyperedge_scale"],
        epochs=args.epochs if args.epochs is not None else preset["epochs"],
        storage_points=preset["storage_points"],
        learning_curve_points=preset["learning_curve_points"],
        dataset_spec=dict(DATASETS[args.dataset]),
    )
    cfg.train = dict(TRAIN)
    cfg.train["epochs"] = cfg.epochs
    if args.device is not None:
        cfg.train["device"] = args.device
    cfg.ensure_dirs()
    return cfg


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="HyperGraph multi-source association mining - reference implementation")
    p.add_argument("--dataset", default="dblp_oag", choices=sorted(DATASETS.keys()),
                   help="which synthetic multi-source dataset to build")
    p.add_argument("--scale", default="small", choices=sorted(SCALE_PRESETS.keys()),
                   help="scale preset; 'full' reproduces the paper magnitudes")
    p.add_argument("--seed", type=int, default=0, help="random seed")
    p.add_argument("--epochs", type=int, default=None, help="override epoch budget")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    p.add_argument("--tag", default="default", help="suffix for output folders")
    p.add_argument("--stage", default="all",
                   choices=["all", "data", "train", "ablation", "protocols",
                            "validity", "storage", "figures"],
                   help="which part of the pipeline to execute")
    p.add_argument("--models", default="all",
                   help="comma separated model names, or 'all'")
    return p


if __name__ == "__main__":
    a = build_argparser().parse_args([])
    c = get_config(a)
    print("dataset          :", c.dataset)
    print("scale            :", c.scale)
    print("nodes            :", c.num_nodes)
    print("hyperedges       :", c.num_hyperedges)
    print("feature dim      :", c.feature_dim)
    print("results (figure) :", c.fig_dir)
    print("results (table)  :", c.table_dir)
