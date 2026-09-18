# HyperGraph Multi-Source Association Mining — Reference Implementation

Runnable reference code for

> **Complex Multi-Source Data Association Mining and Data Foundation Construction Based on HyperGraph**

Everything in this repository is written from the paper only: the formulas of
Section 2 are implemented literally, the experimental protocols of Section 3
are reproduced, and the tables/figures of Section 4 are generated into
`results/`. The three datasets of the paper (DBLP-OAG, MovieLens-25M, Yelp)
cannot be redistributed, so `dataset.py` generates **statistically plausible
surrogates** with exactly the properties the method relies on.

---

## 1. Quick start (PyCharm 2025.1)

```bash
# 1) create/select an interpreter (3.10 - 3.12) and install dependencies
pip install -r requirements.txt

# 2) smoke test: ~1-3 minutes on a laptop, writes every table and figure
python main.py

# 3) a different dataset / larger scale
python main.py --dataset movielens --scale medium
python main.py --dataset yelp --scale medium

# 4) paper magnitudes (needs a GPU and a lot of RAM)
python main.py --scale full --device cuda
```

Single stages (useful while debugging):

```bash
python main.py --stage data       # build data + hypergraph + Table 4
python main.py --stage train      # Table 11/12 + Figure 5(a)(b)
python main.py --stage protocols  # Table 2
python main.py --stage validity   # Table 3 + Table 5
python main.py --stage ablation   # Table 10 + head sweep
python main.py --stage storage    # Table 6 + Figure 7
python main.py --stage figures    # re-draw figures from the stage results
python main.py --models hyperconv_attn,hgnn,gcn   # subset of baselines
```

In PyCharm: set `main.py` as the run configuration and pass `--dataset`,
`--scale`, `--stage` as *Script parameters*. Each module also has a
`__main__` block (`python config.py`, `python validity.py`, ...) for isolated
checks.

---

## 2. File map and paper mapping

| File | Paper content |
|---|---|
| `config.py` | Every hyper-parameter: scale presets, dataset specs (Section 3.1), threshold grids (2.1), model/training settings (3.2-3.3), validity and anomaly budgets (4.3, 4.5), storage benchmark protocol (4.4) |
| `dataset.py` | Synthetic multi-source data: entities, modalities, the joint-majority label rule, natural anomalies of Table 7, injected anomalies of Section 4.5, and the transductive / cold-start / strict-inductive splits of Table 2 |
| `preprocess.py` | Dynamic hyperedge generation (Section 2.1): candidate scoring, adaptive quantile threshold, IQR baseline, EMA streaming threshold, all six acceptance strategies of Table 3, incidence/degree construction and caching |
| `model.py` | Hypergraph convolution with two-stage message passing (Section 2.3), adaptive hyperedge attention and meta-path reasoning (Section 2.4), plus HGNN, HyperConv, HyperGCN, UniGNN, AllSet, AllDeepSets, ED-HNN, Hypformer-style and GCN/GAT/GraphSAGE baselines, the logistic probe and the hyperedge auto-encoder (4.3) and Deep SVDD (Table 9) |
| `validity.py` | Valid Ratio / redundancy judgements (Section 4.3): KSG kNN mutual information, delta-MI, BD-SL, the three criteria and the conjunctive rule, Table 4 statistics, Table 5 aggregation |
| `storage.py` | Tensor storage and bidirectional indexing (Section 2.2) and the Table 6 / Figure 7 benchmark: static CSR, dynamic CSR, hash incidence list, LSM key-value, in-memory hash, and the proposed CSF third-order tensor |
| `train.py` | Training loops, HR@K, anomaly detection with the 2-sigma rule, fixed-test learning curve, ablations and sweeps |
| `evaluate.py` | Metrics (F1, HR@K, AP, ROC-AUC, DR), statistics (paired t-test, bootstrap CI, Cohen's d, Benjamini-Hochberg FDR) and all table builders |
| `visualize.py` | Figures 5(a)(b), 6(a)(b), 7(a)(b) |
| `main.py` | Orchestration of all stages and all file outputs |
| `results/` | `tables/` (CSV) and `figures/` (PNG + PDF), organised per dataset and scale |

### Formula-to-code index

| Paper | Code |
|---|---|
| (1) candidate similarity `s(S)`, adaptive threshold `tau = Q_p`, IQR baseline, EMA update | `preprocess.candidate_similarity`, `preprocess.calibrate_threshold`, `preprocess.EMAThreshold` |
| (2) hypergraph convolution `D_v^-1/2 H W D_e^-1 H^T D_v^-1/2 X Theta`, Laplacian | `model.HypergraphConv` (`mode="symmetric"`) |
| (3) adaptive hyperedge attention `alpha_{v,e} = softmax(LeakyReLU(a1·x_v + a2·m_e))` | `model.HypergraphConv` (`mode="attention"`), `model.segment_softmax` |
| (4) meta-path reasoning over APA / APV / APT | `model.MetaPathReasoner`, `model.build_meta_paths` |
| (5) `delta-MI = I(X_H;Y) − I(X_B;Y)` | `validity.delta_mi` |
| (6) `BD-SL = 1 − I(X_B;Y)/I(X_H;Y)` | `validity.delta_mi` (returns `bd_sl`) |
| (7) KSG kNN-MI estimator, k = 5, Chebyshev metric | `validity.ksg_mi` |
| (8) `S(e_k)` estimated on the 256-member local neighbourhood group | `validity.neighbourhood_group`, `judge_hyperedge` |
| (9) association strength `s(v,e) = <z_v, m_e>` and the 2-sigma rule | `train.association_scores`, `train.anomaly_detection` |
| CSF third-order tensor `T[i,j,k]` + bidirectional index | `storage.CSFTensorBackend` |

---

## 3. What the synthetic data guarantees

* **Three aligned modalities** (text / graph / temporal) as concatenated blocks
  of a shared 128-dimensional space, so the z-scoring and the MI estimator of
  Section 4.3 operate exactly as described.
* **A genuinely high-order label rule.** A candidate node set is a valid
  association iff the *majority* of its members carry the latent class
  `c_v = 1`. Majority cannot be reconstructed from pairwise edges, which is
  precisely why the hyperedge representation carries information that the
  clique expansion does not — the effect that `delta-MI` and `BD-SL` quantify.
* **The natural-anomaly families of Table 7**: temporally inconsistent
  collaboration, cross-source semantic inconsistency, and attribute confusion
  caused by entity-alignment conflict; plus the injected node / event /
  cross-modal families of Section 4.5.
* **Timestamps and node first-appearance times**, so the chronological
  (transductive), cold-start and strict-inductive protocols are all derivable.

### Scale presets

| preset | nodes | hyperedges | intended use |
|---|---|---|---|
| `small` (default) | ≈20 K | ≈38 K | smoke test, all stages in minutes |
| `medium` | ≈100 K | ≈190 K | medium runs |
| `full` | ≈2.03 M | ≈4.72 M | paper magnitudes (GPU + tens of GB RAM) |

`config.SCALE_PRESETS` controls the multipliers; `dataset.generate_dataset`
uses explicit loops for readability, and the spots that must be vectorised at
paper scale are marked `# VECTORISE`.

---

## 4. Honest notes on reconstruction

1. **Equations rendered as images in the source document.** The definitions of
   `delta-MI` and `BD-SL` were recovered from their textual statements
   ("equals 0 when the binary-edge representation fully recovers the hypergraph
   semantics, approaches 1 when none of it can be recovered"), hence
   `BD-SL = 1 − I(X_B;Y)/I(X_H;Y)`. Both live in one small function
   (`validity.delta_mi`) and are trivial to swap if the original expressions
   differ.
2. **RocksDB and Redis are emulated.** Table 6 compares against RocksDB and
   Redis, which need external services. `storage.LSMKeyValueBackend` (memtable
   + sorted levels, binary search per level) and
   `storage.InMemoryHashBackend` (dict-of-dicts with per-entry overhead) emulate
   their characteristic behaviour so the comparison runs out of the box; the
   numbers are therefore indicative, as the paper's own note to Table 6 says.
3. **Hypformer and ED-HNN are simplified.** They keep the essential mechanism
   (linear attention over incidences; edge-dependent filters) but not the exact
   reference implementation.
4. **The validity judging budget is scaled down.** The paper manually verifies
   2,000 judgements per dataset and runs the three criteria on 22,000 positive
   hyperedges; the default here evaluates a smaller sample
   (`main.stage_validity(judge_budget=...)`) so that a laptop run finishes.
   Raise it for a paper-faithful run.
5. **Modality groups for Figure 5(a).** Primary hyperedges carry the modality
   they were sampled from; the graph modality acts as the cross-modal/fusion
   proxy (`main.MODALITY_GROUP`), which is documented at the call site.
6. **Absolute numbers differ from the paper.** The paper's 0.926 F1 / 0.812
   HR@10 come from the real datasets at full scale. The surrogates here are
   designed to reproduce the *ordering* of methods, the protocol structure and
   the qualitative findings — not the exact decimal values.

---

## 5. Verification status

The following paths were executed end to end on Python 3.13 with
`numpy 2.5.3 / scipy 1.18.1 / scikit-learn 1.9.1 / matplotlib 3.11.2 / torch 2.14.0+cpu`:

* `dataset` → `preprocess` (adaptive threshold, acceptance rate, incidence build);
* `storage` (all six backends, incl. the Table 6 latency/memory sweep);
* `train` (association discovery, HR@K, anomaly detection, ablation) for
  `hyperconv_attn`, `hgnn`, `gcn`, `hypergcn`;
* figure rendering smoke test.

On the `small` preset with only 6 epochs the surrogate yields F1 ≈ 0.70–0.74
and preserves the *ordering* of the paper (HyperConv-Attn > HGNN > HyperGCN >
GCN, and `full` > `wo_attention` in the ablation). Absolute values equal to the
paper's 0.926 / 0.812 require the real datasets at `--scale full`, which is what
the `full` preset and the documented budgets in `config.py` are for.

Not covered by the smoke test: the `validity` stage at full budget (the KSG
estimator is deliberately subsampled by default) and `--dataset yelp`
(the code paths are identical to `dblp_oag`).

---

## 6. Outputs

```
results/
├── tables/<dataset>/<scale>/
│   ├── table2_protocols.csv          # transductive / cold-start / strict inductive
│   ├── table3_redundancy.csv         # Valid Ratio, Redundancy, delta-MI, BD-SL
│   ├── table4_threshold.csv          # similarity statistics + threshold calibration
│   ├── table5_validity.csv           # per-criterion judgements and intersections
│   ├── table6_storage.csv            # storage backend comparison
│   ├── table8_anomaly.csv            # anomaly detection rates
│   ├── table9_anomaly_baselines.csv  # vs. dedicated anomaly detectors
│   ├── table10_ablation.csv          # module contributions
│   ├── table11_baselines.csv         # unified learning + efficiency comparison
│   ├── table12_significance.csv      # paired t-test, CI, Cohen's d
│   ├── figure5b_learning_curve.csv
│   ├── section46_head_sweep.csv
│   └── run_config.json
└── figures/<dataset>/<scale>/
    ├── fig5a_modality_f1.{png,pdf}
    ├── fig5b_learning_curve.{png,pdf}
    ├── fig6a_hr10_scenarios.{png,pdf}
    ├── fig6b_hrk_curves.{png,pdf}
    ├── fig7a_storage_scaling.{png,pdf}
    └── fig7b_batch_latency.{png,pdf}
```

---

## 7. Extending the code

* **New dataset**: add an entry to `config.DATASETS` and, if the node types are
  new, extend `model.TYPE_LETTERS` (used to compose APA/APV/APT).
* **New baseline**: subclass `model.BaseAssociationModel`, implement `encode`,
  register it in `model.MODEL_REGISTRY` and append the name to
  `model.BASELINE_ORDER`.
* **New validity criterion**: add a function to `validity.py` returning
  `(bool, gain)`, then include it in `validity.judge_hyperedge` and adjust
  `VALIDITY["conjunctive_min_votes"]`.
* **New storage backend**: implement `insert/delete/update/contains/
  next_edge_id/memory_bytes` and register it in `storage.BACKENDS`.
* **Reproducibility**: every stage is seeded (`--seed`), the threshold and any
  other hyper-parameter are always fitted on the validation split only, and
  `run_config.json` records the exact configuration of every run.
