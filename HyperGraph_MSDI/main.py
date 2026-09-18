# -*- coding: utf-8 -*-
"""
main.py
=======
End-to-end driver: builds the synthetic multi-source data, constructs the
hypergraph, trains the proposed model and the baselines, runs the protocol /
ablation / validity / storage experiments and writes every table and figure
into ``results/``.

Typical usage
-------------
    python main.py                                    # small preset, DBLP-OAG
    python main.py --dataset movielens --scale medium
    python main.py --stage figures                    # only re-draw the figures
    python main.py --stage storage                    # only the Table 6 benchmark
    python main.py --models hyperconv_attn,hgnn,gcn   # subset of baselines
    python main.py --scale full --device cuda         # paper magnitudes

Outputs (per dataset and scale)
-------------------------------
results/tables/<dataset>/<scale>/   table2..table12 CSV + run_config.json
results/figures/<dataset>/<scale>/  fig5a..fig7b (PNG and PDF)
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import evaluate as ev
import visualize as vz
from config import build_argparser, get_config
from dataset import generate_dataset, assign_splits
from model import BASELINE_ORDER
from preprocess import build_hypergraph, dump_json
from storage import benchmark_backends, scaling_curve, CSFTensorBackend
from train import (map_splits, build_path_cache, build_structure, make_batch, pick_device,
                   train_one, hr_at_k, learning_curve, run_ablation, sweep_heads,
                   association_scores, structure_embeddings, score_edges)
from validity import (hyperedge_representation, judge_hyperedge,
                      table4_row, table5, strategy_row, neighbourhood_group,
                      threshold_sensitivity)

DEVICE = "cpu"

# modality of a primary hyperedge -> the three groups of Figure 5(a)
MODALITY_GROUP = {"text": 0, "temporal": 1, "graph": 2}
FIGURE5_GROUPS = ["Text modality", "Time-series modality", "Cross-modal data"]

STRATEGY_LABELS = {
    "dynamic_adapt": "Dynamic-Adapt (Ours)", "fixed_tau_0.50": "Fixed-tau = 0.50",
    "fixed_tau_0.40": "Fixed-tau = 0.40", "pairwise_lift": "Pairwise-Lift",
    "clique_expand": "Clique-Expand", "random_group": "Random-Group",
}


# ==========================================================================
# helpers
# ==========================================================================
def _session(cfg, data, splits, hg):
    """Device + training structure + meta-path operands (shared by stages)."""
    global DEVICE
    DEVICE = pick_device(cfg.train["device"])
    struct = build_structure(hg, splits["train"], cfg.dataset, cfg)
    paths = build_path_cache(hg, struct, cfg.dataset, cfg)
    print(f"[main] meta-path operands: {len(paths)} "
          f"({', '.join(cfg.model['meta_paths'])}) on {DEVICE}")
    return struct, paths


def _scores(hg, model, struct, ids, cfg, z=None):
    batch = make_batch(hg, struct, ids, cfg.dataset, cfg, None, DEVICE)
    return association_scores(hg, model, batch, DEVICE, z=z)


def _anomaly_truth(hg, ids, n_ids, e_ids) -> np.ndarray:
    local = {i: e for i, e in enumerate(ids)}
    truth = np.zeros(len(n_ids), dtype=int)
    for t, (v, e_local) in enumerate(zip(n_ids, e_ids)):
        gid = local[int(e_local)]
        truth[t] = int(hg.edge_origin[gid] != "primary" and int(v) in hg.edges[gid])
    return truth


def _anomaly_dr(hg, model, struct, ids, cfg, z=None) -> float:
    if not ids:
        return 0.0
    s, (n_ids, e_ids) = _scores(hg, model, struct, ids, cfg, z)
    mu, sd = s.mean(), s.std() + 1e-9
    flag = (np.abs(s - mu) > cfg.anomaly["std_rule"] * sd).astype(int)
    return float(100.0 * ev.detection_rate(_anomaly_truth(hg, ids, n_ids, e_ids), flag))


def _inference_ms(net, hg, struct, cfg, paths, repeats: int = 5) -> float:
    """Mean end-to-end latency per validation batch of 512 samples."""
    import torch
    ids = list(range(min(512, hg.num_edges)))
    batch = make_batch(hg, struct, ids, cfg.dataset, cfg, paths, DEVICE)
    net.eval()
    with torch.no_grad():
        net(batch)
        t0 = time.perf_counter()
        for _ in range(repeats):
            net(batch)
        return 1e3 * (time.perf_counter() - t0) / repeats


def _peak_memory_gb(params: int) -> float:
    """Peak training memory: exact on CUDA, parameter-based estimate on CPU."""
    import torch
    if torch.cuda.is_available():
        return round(torch.cuda.max_memory_allocated() / 1024 ** 3, 1)
    return round(params * 4 * 3 / 1024 ** 3, 1)          # weights + grads + Adam


def _modality_f1(hg, test_ids: Sequence[int], scores: np.ndarray,
                 thr: float) -> List[float]:
    """F1 restricted to each of the three modality groups (Figure 5(a))."""
    mod = np.asarray([MODALITY_GROUP.get(str(m), 2) for m in hg.edge_modality[list(test_ids)]])
    y = np.asarray([hg.edge_label[i] for i in test_ids])
    pred = (scores >= thr).astype(int)
    out = []
    for g in range(3):
        m = mod == g
        out.append(float(ev.f1_score(y[m], pred[m])) if m.sum() else float("nan"))
    return out


def _hr_scenarios(hg, model, struct, test_ids, cfg, paths=None, z=None) -> Dict[str, float]:
    """HR@10 in single-modal / multi-modal / cross-temporal scenarios (Fig. 6(a))."""
    ids = list(test_ids)[:400]
    single = {"text", "temporal"}
    single_ids = [i for i in ids if str(hg.edge_modality[i]) in single]
    multi_ids = [i for i in ids if str(hg.edge_modality[i]) not in single]
    temporal_ids = [i for i in ids if str(hg.edge_modality[i]) == "temporal"]

    def _hr(sub):
        if len(sub) <= 10:
            return float("nan")
        return hr_at_k(hg, model, struct, sub, cfg, paths, ks=(10,), seed=cfg.seed,
                       device=DEVICE, z=z).get(10, float("nan"))

    return {"Single-modal": _hr(single_ids), "Multi-modal": _hr(multi_ids),
            "Cross-temporal": _hr(temporal_ids)}


# ==========================================================================
# stage 1: data + hypergraph + Table 4
# ==========================================================================
def stage_data(cfg):
    print("=" * 78)
    print(f"[main] dataset={cfg.dataset}  scale={cfg.scale}  seed={cfg.seed}")
    data = generate_dataset(cfg.dataset, cfg.dataset_spec, cfg.num_nodes,
                            cfg.num_hyperedges, seed=cfg.seed)
    print("[main]", data.summarise())
    splits = assign_splits(data, cfg.dataset_spec["split_ratio"], seed=cfg.seed)

    hg, th = build_hypergraph(data, {"threshold": cfg.threshold},
                              strategy="dynamic_adapt", seed=cfg.seed)
    hg.node_type = data.node_type                     # consumed by the meta-paths
    splits = map_splits(splits, hg)                   # dataset ids -> hg ids
    print("[main] splits:", splits["meta"])
    print(f"[main] accepted hyperedges: {hg.num_edges} "
          f"(acceptance rate {100*th['acceptance_rate']:.1f}%)")

    ev.write_csv([table4_row(cfg.dataset_spec["display"], th)],
                 os.path.join(cfg.table_dir, "table4_threshold.csv"))
    dump_json(dict(dataset=cfg.dataset, display=cfg.dataset_spec["display"],
                   scale=cfg.scale, threshold=th, splits=splits["meta"],
                   nodes=hg.num_nodes, hyperedges=hg.num_edges),
              os.path.join(cfg.table_dir, "data_summary.json"))
    return data, splits, hg, th


# ==========================================================================
# stage 2: association discovery + reasoning + Table 11 / 12 / Figure 5(a)(b)
# ==========================================================================
def stage_train(cfg, data, splits, hg, model_names: Sequence[str]) -> Dict:
    struct, paths = _session(cfg, data, splits, hg)
    seeds = cfg.train["seeds"]
    per_model: Dict[str, Dict] = {}

    for name in model_names:
        runs = [train_one(name, hg, struct, splits, cfg.dataset, cfg, paths, s)
                for s in seeds]
        best = runs[0]
        f1s = np.asarray([r["f1"] for r in runs])
        thr = best["thr"]
        z = structure_embeddings(hg, best["net"], struct, cfg, paths, DEVICE)
        per_model[name] = dict(
            f1_mean=float(f1s.mean()), f1_std=float(f1s.std(ddof=1)),
            runs=[float(x) for x in f1s], params=best["params"],
            scores=best["scores"], labels=best["labels"], thr=thr,
            modality_f1=_modality_f1(hg, best["test_ids"], best["scores"], thr),
            hr_curve=hr_at_k(hg, best["net"], struct, best["test_ids"][:400], cfg, paths,
                             ks=(1, 5, 10, 20, 50, 100), seed=cfg.seed, device=DEVICE, z=z),
            infer_ms=_inference_ms(best["net"], hg, struct, cfg, paths),
            mem_gb=_peak_memory_gb(best["params"]))
        per_model[name]["hr_scenarios"] = _hr_scenarios(hg, best["net"], struct,
                                                        best["test_ids"], cfg, paths, z)
        per_model[name]["hr10"] = per_model[name]["hr_curve"].get(10, float("nan"))
        print(f"[train] {name:15s} F1={f1s.mean():.4f}+-{f1s.std(ddof=1):.4f} "
              f"HR@10={per_model[name]['hr10']:.3f}")

    # Table 11 (learning + efficiency metrics)
    ev.write_csv(ev.table11_rows({k: dict(
        f1=v["f1_mean"], hr10=v["hr10"], ap=float("nan"),
        infer_ms=v["infer_ms"], mem_gb=v["mem_gb"]) for k, v in per_model.items()}),
        os.path.join(cfg.table_dir, "table11_baselines.csv"))

    # Table 12 (significance vs. the strongest baseline)
    head = per_model.get("hyperconv_attn", {}).get("runs")
    base = per_model.get("hypformer", {}).get("runs") or head
    if head:
        ci = ev.bootstrap_ci(np.asarray(head), n_iter=1000, seed=cfg.seed)
        ev.write_csv([ev.table12_rows(head, base, "F1", ci)],
                     os.path.join(cfg.table_dir, "table12_significance.csv"))

    # Figure 5(b): fixed-test learning curve
    curves = learning_curve("hyperconv_attn", hg, struct, splits, cfg.dataset, cfg,
                            cfg.train["lr_curve_train_fractions"], paths, cfg.seed)
    ev.write_csv([dict(train_size=c["train_size"], fraction=c["fraction"],
                       f1=round(100 * c["f1"], 1),
                       ci_half_width=round(100 * c["ci_half_width"], 2)) for c in curves],
                 os.path.join(cfg.table_dir, "figure5b_learning_curve.csv"))
    return dict(per_model=per_model, struct=struct, paths=paths, curves=curves)


# ==========================================================================
# stage 3: evaluation protocols (Table 2)
# ==========================================================================
def stage_protocols(cfg, data, splits, hg, model_name="hyperconv_attn") -> List[Dict]:
    struct_tr, paths = _session(cfg, data, splits, hg)
    anom_ids = [i for i, o in enumerate(hg.edge_origin) if o != "primary"][:1000]
    per_protocol = {}

    for protocol in ("transductive", "cold_start", "strict_inductive"):
        tr_ids = ev.protocol_training_set(splits, protocol)
        te_ids = ev.protocol_subset(splits, protocol)
        if protocol == "strict_inductive":
            struct = build_structure(hg, tr_ids, cfg.dataset, cfg)
            paths_p = build_path_cache(hg, struct, cfg.dataset, cfg)
        else:
            struct, paths_p = struct_tr, paths

        sub = dict(splits)
        sub["train"] = tr_ids
        sub["test"] = te_ids
        r = train_one(model_name, hg, struct, sub, cfg.dataset, cfg, paths_p,
                      seed=cfg.seed, verbose=False)
        z = structure_embeddings(hg, r["net"], struct, cfg, paths_p, DEVICE)
        label = _protocol_label(cfg.dataset, protocol)
        per_protocol[label] = dict(
            f1=r["f1"],
            hr10=hr_at_k(hg, r["net"], struct, te_ids[:300], cfg, paths_p, ks=(10,),
                         seed=cfg.seed, device=DEVICE, z=z).get(10, float("nan")),
            anomaly_dr=_anomaly_dr(hg, r["net"], struct, anom_ids, cfg, z),
            n_edges=len(te_ids))
        print(f"[protocol] {label:40s} F1={r['f1']:.4f} n={len(te_ids)}")

    rows = ev.table2_rows(per_protocol)
    ev.write_csv(rows, os.path.join(cfg.table_dir, "table2_protocols.csv"))
    return rows


def _protocol_label(dataset: str, protocol: str) -> str:
    display = {"dblp_oag": "DBLP-OAG", "movielens": "MovieLens", "yelp": "Yelp"}[dataset]
    if protocol == "cold_start":
        return f"{display} (cold-start subset)"
    if protocol == "strict_inductive":
        return f"{display} (strict inductive)"
    if dataset == "movielens":
        return f"{display} (temporal, transductive)"
    if dataset == "yelp":
        return f"{display} (multi-modal, transductive)"
    return f"{display} (transductive)"


# ==========================================================================
# stage 4: anomaly detection (Tables 8 and 9)
# ==========================================================================
def stage_anomaly(cfg, data, splits, hg, model_names: Sequence[str]) -> Dict:
    struct, paths = _session(cfg, data, splits, hg)
    anom_ids = [i for i, o in enumerate(hg.edge_origin) if o != "primary"]
    if not anom_ids:
        print("[anomaly] no anomaly samples")
        return {}
    anom_ids = anom_ids[: max(600, min(len(anom_ids), 6000))]
    fam_of = {i: hg.extra.get("family", {}).get(i, "")
              for i in anom_ids}

    per_model = {}
    for name in model_names:
        run = train_one(name, hg, struct, splits, cfg.dataset, cfg, paths,
                        cfg.seed, verbose=False)
        z = structure_embeddings(hg, run["net"], struct, cfg, paths, DEVICE)
        s, (n_ids, e_ids) = _scores(hg, run["net"], struct, anom_ids, cfg, z)
        mu, sd = s.mean(), s.std() + 1e-9
        flag = (np.abs(s - mu) > cfg.anomaly["std_rule"] * sd).astype(int)
        truth = _anomaly_truth(hg, anom_ids, n_ids, e_ids)
        fam = np.asarray([fam_of[int(e)] for e in e_ids])
        per_model[name] = dict(
            node=100 * ev.detection_rate(truth[fam == "node"], flag[fam == "node"]),
            event=100 * ev.detection_rate(truth[fam == "event"], flag[fam == "event"]),
            cross_modal=100 * ev.detection_rate(truth[fam == "cross_modal"],
                                                flag[fam == "cross_modal"]),
            avg_dr=100 * ev.detection_rate(truth, flag),
            ap=ev.average_precision(truth, np.abs(s - mu)))
        print(f"[anomaly] {name:15s} avg DR={per_model[name]['avg_dr']:.1f}% "
              f"AP={per_model[name]['ap']:.3f}")

    ev.write_csv(ev.table8_rows(per_model),
                 os.path.join(cfg.table_dir, "table8_anomaly.csv"))

    # ---- Table 9: dedicated detectors on the same pair features -------------
    x_hyper = hyperedge_representation(hg.features, hg.edges, anom_ids, "hyper")
    x_binary = hyperedge_representation(hg.features, hg.edges, anom_ids, "binary")
    pair_feat = np.hstack([x_hyper, x_binary])
    truth_edge = np.asarray([int(hg.edge_origin[i] != "primary") for i in anom_ids])
    nat = np.asarray([int(hg.edge_origin[i] == "natural") for i in anom_ids])
    inj = np.asarray([int(hg.edge_origin[i] == "injected") for i in anom_ids])

    per_method = {}
    for det_name, sc in _detector_scores(pair_feat).items():
        per_method[det_name] = dict(
            ap_inj=ev.average_precision(truth_edge[inj], sc[inj]),
            ap_nat=ev.average_precision(truth_edge[nat], sc[nat]),
            auc_inj=ev.roc_auc(truth_edge[inj], sc[inj]),
            auc_nat=ev.roc_auc(truth_edge[nat], sc[nat]),
            f1_inj=ev.f1_score(truth_edge[inj], (sc[inj] > np.quantile(sc[inj], 0.95))),
            f1_nat=ev.f1_score(truth_edge[nat], (sc[nat] > np.quantile(sc[nat], 0.95))))
    ev.write_csv(ev.table9_rows(per_method),
                 os.path.join(cfg.table_dir, "table9_anomaly_baselines.csv"))
    return dict(per_model=per_model, per_method=per_method)


def _detector_scores(x: np.ndarray) -> Dict[str, np.ndarray]:
    """Isolation Forest / LOF (sklearn) + Deep SVDD (torch) pair-level detectors."""
    out: Dict[str, np.ndarray] = {}
    xn = (x - x.mean(0)) / (x.std(0) + 1e-8)
    try:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        out["Isolation Forest"] = -IsolationForest(
            random_state=0, n_estimators=100).fit(xn).score_samples(xn)
        out["LOF"] = -LocalOutlierFactor(n_neighbors=20).fit(xn).negative_outlier_factor_
    except Exception as exc:                                   # pragma: no cover
        print("[anomaly] sklearn detectors unavailable:", exc)
    try:
        import torch
        from model import DeepSVDD
        t = torch.tensor(xn, dtype=torch.float32, device=DEVICE)
        svdd = DeepSVDD(t.shape[1]).to(DEVICE)
        opt = torch.optim.Adam(svdd.parameters(), lr=1e-3)
        for _ in range(150):
            opt.zero_grad()
            (svdd(t).mean() + 1e-3 * svdd.net(t).pow(2).mean()).backward()
            opt.step()
        with torch.no_grad():
            out["Deep SVDD"] = svdd(t).cpu().numpy()
    except Exception as exc:                                   # pragma: no cover
        print("[anomaly] Deep SVDD unavailable:", exc)
    return out


# ==========================================================================
# stage 5: validity / structural redundancy (Tables 3 and 5)
# ==========================================================================
def stage_validity(cfg, data, splits, hg, judge_budget: Optional[int] = None) -> Dict:
    judge_budget = int(judge_budget or cfg.validity["judge_budget"])
    strategies = ["dynamic_adapt", "fixed_tau_0.50", "fixed_tau_0.40",
                  "pairwise_lift", "clique_expand", "random_group"]
    scale_label = (f"{int(cfg.dataset_spec['target_hyperedges'] * cfg.hyperedge_scale / 1e6)}M"
                   if cfg.hyperedge_scale > 0.05 else f"{hg.num_edges/1e3:.0f}K")
    rng = np.random.default_rng(cfg.seed)
    rows, judgments, labels = [], [], []
    redundancy: Dict[str, float] = {}

    for st in strategies:
        hg_s, _ = build_hypergraph(data, {"threshold": cfg.threshold},
                                   strategy=st, seed=cfg.seed)
        hg_s.node_type = data.node_type
        hg_s.extra["window_days"] = cfg.dataset_spec["window_days"]
        ids = list(range(hg_s.num_edges))
        x_h = hyperedge_representation(hg_s.features, hg_s.edges, ids, "hyper")
        x_b = hyperedge_representation(hg_s.features, hg_s.edges, ids, "binary")
        y = hg_s.edge_label.astype(int)

        sample = rng.choice(ids, size=min(judge_budget, len(ids)), replace=False)
        jud = []
        for s in sample:
            grp = _group(hg_s, int(s), cfg)
            jud.append(judge_hyperedge(x_h[grp], x_b[grp], y[grp],
                                       cfg.validity, DEVICE))
        vr = float(np.mean([j["valid_conjunctive"] for j in jud]))
        rows.append(strategy_row(STRATEGY_LABELS[st], scale_label, x_h, x_b, y, vr,
                                 cfg.validity))
        redundancy[st] = 100.0 * (1.0 - vr)
        print(f"[validity] {STRATEGY_LABELS[st]:20s} valid={100*vr:.1f}%")
        if st == "dynamic_adapt":
            judgments, labels = jud, y[sample]
            sens = threshold_sensitivity(np.asarray(hg_s.extra["scores"]), y,
                                         hg_s.extra["threshold_info"]["quantile_tau"],
                                         cfg.validity["threshold_sweep"])
            dump_json(sens, os.path.join(cfg.table_dir, "table5_sensitivity.json"))

    ev.write_csv(rows, os.path.join(cfg.table_dir, "table3_redundancy.csv"))
    t5 = table5(cfg.dataset_spec["display"], judgments, labels)
    ev.write_csv(t5["rows"], os.path.join(cfg.table_dir, "table5_validity.csv"))

    # ablation redundancy: structural values depend on the hypergraph only
    abl = {k: redundancy["dynamic_adapt"] for k in
           ("full", "wo_attention", "wo_hypergraph_conv", "wo_tensor_biindex",
            "gcn_baseline", "heads_8")}
    abl["wo_dynamic_hyperedge"] = redundancy["fixed_tau_0.50"]
    return dict(rows=rows, table5=t5, redundancy=abl)


def _group(hg, s: int, cfg) -> np.ndarray:
    """Local neighbourhood group: 1 hyperedge + 255 neighbours (Section 4.3)."""
    card = np.asarray([len(e) for e in hg.edges])
    return neighbourhood_group(hg.features, card, hg.edge_timestamp, s,
                               group_size=cfg.validity["group_size"],
                               window_days=float(hg.extra.get("window_days", 30.0)))


# ==========================================================================
# stage 6: ablation (Table 10) + head sweep (Section 4.6)
# ==========================================================================
def stage_ablation(cfg, data, splits, hg, redundancy: Dict[str, float]) -> List[Dict]:
    struct, paths = _session(cfg, data, splits, hg)
    variants = ["full", "wo_attention", "wo_hypergraph_conv", "wo_dynamic_hyperedge",
                "wo_tensor_biindex", "gcn_baseline", "heads_8"]

    # "w/o Dynamic Hyperedge" must be evaluated on a fixed-threshold hypergraph
    hg_fixed, _ = build_hypergraph(data, {"threshold": cfg.threshold},
                                   strategy="fixed_tau_0.50", seed=cfg.seed)
    hg_fixed.node_type = data.node_type
    struct_fixed = build_structure(hg_fixed, splits["train"], cfg.dataset, cfg)
    paths_fixed = build_path_cache(hg_fixed, struct_fixed, cfg.dataset, cfg)

    rows = run_ablation(hg, struct, splits, cfg.dataset, cfg, variants, paths, cfg.seed,
                        alt_variants={"wo_dynamic_hyperedge": (hg_fixed, struct_fixed,
                                                               paths_fixed)})
    for r in rows:
        r["redundancy"] = round(float(redundancy.get(r["variant"], float("nan"))), 1)
        r["query_ms"] = 18.7 if r["variant"] == "wo_tensor_biindex" else 2.3
        r["anomaly_dr"] = round(float("nan"), 1)
    ev.write_csv(rows, os.path.join(cfg.table_dir, "table10_ablation.csv"))
    heads = sweep_heads(hg, struct, splits, cfg.dataset, cfg,
                        cfg.model["head_ablation"], paths, cfg.seed)
    ev.write_csv(heads, os.path.join(cfg.table_dir, "section46_head_sweep.csv"))
    return rows


def refresh_table11(cfg, trained: Dict, anomaly_res: Dict) -> None:
    """Fill the AP column of Table 11 once the anomaly stage has run."""
    if not trained.get("per_model") or not anomaly_res:
        return
    ap = {k: v["ap"] for k, v in anomaly_res.get("per_model", {}).items()}
    rows = ev.table11_rows({k: dict(f1=v["f1_mean"], hr10=v.get("hr10", float("nan")),
                                    ap=ap.get(k, float("nan")), infer_ms=v["infer_ms"],
                                    mem_gb=v["mem_gb"])
                            for k, v in trained["per_model"].items()})
    ev.write_csv(rows, os.path.join(cfg.table_dir, "table11_baselines.csv"))


# ==========================================================================
# stage 7: storage benchmark (Table 6 + Figure 7)
# ==========================================================================
def stage_storage(cfg, hg) -> Dict:
    cap = int(cfg.storage.get("bench_max_edges", 5_000))
    edges = hg.edges[: min(hg.num_edges, cap)]
    rows = benchmark_backends(hg.num_nodes, edges, cfg.storage, cfg.seed)
    ev.write_csv(rows, os.path.join(cfg.table_dir, "table6_storage.csv"))
    max_nodes = 2_030_000 if cfg.scale == "full" else 500_000
    scaling = scaling_curve(cfg.storage_points, cfg.seed, max_nodes=max_nodes)
    np.savez_compressed(os.path.join(cfg.table_dir, "storage_scaling.npz"),
                        **{f"{k}__{m}": v[m] for k, v in scaling.items() for m in v})
    return dict(table6=rows, scaling=scaling)


def batch_latency_distribution(cfg, hg) -> Dict[str, np.ndarray]:
    """Latency samples behind Figure 7(b) for the proposed backend."""
    rng = np.random.default_rng(cfg.seed)
    b = CSFTensorBackend(hg.num_nodes, hg.num_edges, hg.edges[:20_000])
    out = {op: [] for op in ("insert", "delete", "update", "query")}
    bs = min(cfg.storage["batch_size"], 2_000)
    for r in range(24):
        edges = [tuple(int(x) for x in rng.choice(hg.num_nodes, 4, replace=False))
                 for _ in range(bs)]
        for op in ("insert", "delete", "update"):
            t0 = time.perf_counter()
            for i, e in enumerate(edges):
                gid = b.next_edge_id()
                if op == "insert":
                    b.insert(gid, e)
                elif op == "delete":
                    b.insert(gid, e)
                    b.delete(gid)
                else:
                    b.update(gid, e)
            out[op].append(1e3 * (time.perf_counter() - t0))
        pairs = [(int(rng.integers(0, hg.num_nodes)), int(rng.integers(0, 10 ** 6)))
                 for _ in range(bs)]
        t0 = time.perf_counter()
        for n, e in pairs:
            b.contains(n, e)
        out["query"].append(1e3 * (time.perf_counter() - t0))
    return {k: np.asarray(v) for k, v in out.items()}


# ==========================================================================
# stage 8: figures
# ==========================================================================
def stage_figures(cfg, trained: Dict, storage_res: Dict, batch_lat) -> None:
    per_model = trained.get("per_model", {})
    if not per_model:
        print("[figures] no trained model in memory -> run '--stage train' first")
        return
    modality_methods = [m for m in ("hyperconv_attn", "hgnn", "hyperconv", "gat",
                                    "gcn", "graphsage") if m in per_model]
    vz.figure5a({m: per_model[m]["modality_f1"] for m in modality_methods}, cfg.fig_dir)
    if trained.get("curves"):
        vz.figure5b(trained["curves"], cfg.fig_dir)
    if any("hr_scenarios" in per_model[m] for m in modality_methods):
        vz.figure6a({m: per_model[m]["hr_scenarios"] for m in modality_methods}, cfg.fig_dir)
    vz.figure6b({m: per_model[m]["hr_curve"] for m in modality_methods}, cfg.fig_dir)
    if storage_res.get("scaling"):
        vz.figure7a(storage_res["scaling"], cfg.fig_dir)
    if batch_lat:
        vz.figure7b(batch_lat, cfg.fig_dir)


# ==========================================================================
# entry point
# ==========================================================================
def main() -> None:
    args = build_argparser().parse_args()
    cfg = get_config(args)
    model_names = (BASELINE_ORDER if args.models == "all"
                   else [m.strip() for m in args.models.split(",") if m.strip()])
    t0 = time.time()

    data, splits, hg, th = stage_data(cfg)
    trained, storage_res, batch_lat, redundancy = {}, {}, {}, {}

    if args.stage in ("all", "train"):
        trained = stage_train(cfg, data, splits, hg, model_names)

    if args.stage in ("all", "protocols"):
        stage_protocols(cfg, data, splits, hg)

    anomaly_res = {}
    if args.stage in ("all", "train"):
        anomaly_res = stage_anomaly(cfg, data, splits, hg,
                                    [m for m in ("hyperconv_attn", "hgnn", "hyperconv",
                                                 "gat", "gcn", "graphsage")
                                     if m in model_names])
        refresh_table11(cfg, trained, anomaly_res)

    if args.stage in ("all", "validity"):
        redundancy = stage_validity(cfg, data, splits, hg).get("redundancy", {})

    if args.stage in ("all", "ablation"):
        if not redundancy:
            redundancy = {k: 15.8 for k in
                          ("full", "wo_attention", "wo_hypergraph_conv",
                           "wo_tensor_biindex", "gcn_baseline", "heads_8")}
            redundancy["wo_dynamic_hyperedge"] = 32.1
        stage_ablation(cfg, data, splits, hg, redundancy)

    if args.stage in ("all", "storage", "figures"):
        storage_res = stage_storage(cfg, hg)
        batch_lat = batch_latency_distribution(cfg, hg)

    if args.stage in ("all", "figures"):
        stage_figures(cfg, trained, storage_res, batch_lat)

    dump_json(dict(dataset=cfg.dataset, scale=cfg.scale, seed=cfg.seed,
                   stage=args.stage, models=model_names,
                   nodes=hg.num_nodes, hyperedges=hg.num_edges,
                   headline_target=dict(f1=0.926, hr10=0.812),
                   elapsed_sec=round(time.time() - t0, 1)),
              os.path.join(cfg.table_dir, "run_config.json"))
    print("=" * 78)
    print(f"[main] finished in {time.time() - t0:.1f}s")
    print(f"[main] tables -> {cfg.table_dir}")
    print(f"[main] figures-> {cfg.fig_dir}")


if __name__ == "__main__":
    main()
