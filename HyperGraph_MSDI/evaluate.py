# -*- coding: utf-8 -*-
"""
evaluate.py
===========
Metrics, statistical tests and table assembly (Sections 3.2 and 4).

All functions here are pure numpy: the experiment drivers live in train.py,
which imports this module, so there is no circular dependency.

Metrics
-------
F1, HR@K, AP (the area under the precision-recall curve, the metric used for
the ~5% anomaly rate), ROC-AUC and the anomaly detection rate.

Statistics (Section 3.2 / Table 12)
-----------------------------------
paired t-test over the 5 independent runs, bootstrap 95% confidence intervals,
Cohen's d effect size (>= 0.8 = large) and Benjamini-Hochberg FDR correction
across all pairwise comparisons.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def confusion(y: np.ndarray, pred: np.ndarray) -> Tuple[float, float, float, float]:
    y = np.asarray(y); pred = np.asarray(pred)
    tp = float(((y == 1) & (pred == 1)).sum())
    fp = float(((y == 0) & (pred == 1)).sum())
    fn = float(((y == 1) & (pred == 0)).sum())
    tn = float(((y == 0) & (pred == 0)).sum())
    return tp, fp, fn, tn


def f1_score(y: np.ndarray, pred: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(y, pred)
    if tp == 0:
        return 0.0
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def precision(y, pred) -> float:
    tp, fp, _, _ = confusion(y, pred)
    return tp / (tp + fp + 1e-9)


def recall(y, pred) -> float:
    tp, _, fn, _ = confusion(y, pred)
    return tp / (tp + fn + 1e-9)


def detection_rate(truth: np.ndarray, flag: np.ndarray) -> float:
    """DR = recall of the flagged anomalous node-hyperedge pairs."""
    return recall(truth, flag)


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve (step-wise, ties handled)."""
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    if y.sum() == 0:
        return 0.0
    order = np.argsort(-score)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / y.sum()
    ap = 0.0
    prev_r = 0.0
    for p, r in zip(prec, rec):
        ap += p * (r - prev_r)
        prev_r = r
    return float(ap)


def roc_auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based ROC-AUC (equivalent to the Mann-Whitney U statistic)."""
    y = np.asarray(y); score = np.asarray(score, dtype=np.float64)
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.0
    all_scores = np.concatenate([pos, neg])
    order = np.argsort(all_scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(all_scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts)); np.add.at(sums, inv, ranks)
    avg = {i: sums[i] / counts[i] for i in range(len(counts))}
    ranks = np.asarray([avg[i] for i in inv])
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def hr_at_k(ranks: Sequence[int], ks: Sequence[int] = (10,)) -> Dict[int, float]:
    ranks = np.asarray(ranks)
    return {k: float((ranks <= k).mean()) for k in ks}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(values: np.ndarray, n_iter: int = 1_000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    draws = rng.choice(values, size=(n_iter, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired t-test p-value (normal approximation of the t law)."""
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = a - b
    n = len(d)
    if n < 2 or d.std(ddof=1) == 0:
        return 1.0
    t = d.mean() / (d.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df=n - 1))
    except Exception:
        from math import erf, sqrt
        return float(2 * (1 - 0.5 * (1 + erf(abs(t) / sqrt(2)))))


def cohens_d(a: Sequence[float], b: Sequence[float]) -> float:
    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / max(1, na + nb - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
    """BH-FDR corrected p-values (Section 3.2)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(1, n + 1))
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return [float(x) for x in out]


def summarise_runs(values: Sequence[float], n_iter: int = 1_000,
                   seed: int = 0) -> Dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    lo, hi = bootstrap_ci(v, n_iter=n_iter, seed=seed)
    return dict(mean=float(v.mean()), std=float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                ci_low=lo, ci_high=hi, n=len(v))


# --------------------------------------------------------------------------
# protocol helpers (Table 2)
# --------------------------------------------------------------------------
PROTOCOL_ROWS = [
    ("transductive", "DBLP-OAG (transductive)"),
    ("transductive", "MovieLens (temporal, transductive)"),
    ("transductive", "Yelp (multi-modal, transductive)"),
    ("cold_start", "DBLP-OAG (cold-start subset)"),
    ("strict_inductive", "DBLP-OAG (strict inductive)"),
    ("strict_inductive", "MovieLens (temporal, strict inductive)"),
]


def protocol_subset(splits: Dict, protocol: str) -> List[int]:
    """Return the evaluation edge ids of a protocol."""
    if protocol == "transductive":
        return splits["test"]
    if protocol == "cold_start":
        return splits["cold_start"]
    if protocol == "strict_inductive":
        return splits["strict_inductive_test"]
    raise KeyError(protocol)


def protocol_training_set(splits: Dict, protocol: str) -> List[int]:
    if protocol == "strict_inductive":
        return splits["strict_inductive_train"]
    return splits["train"]


# --------------------------------------------------------------------------
# table assembly
# --------------------------------------------------------------------------
def table2_rows(per_protocol: Dict[str, Dict[str, float]]) -> List[Dict]:
    rows = []
    for protocol, dataset_label in PROTOCOL_ROWS:
        m = per_protocol.get(dataset_label)
        if m is None:
            continue
        rows.append(dict(evaluation_protocol=_protocol_text(protocol),
                         dataset=dataset_label, f1=round(m["f1"], 3),
                         hr10=round(m["hr10"], 3),
                         anomaly_detection_rate=round(m["anomaly_dr"], 1)))
    return rows


def _protocol_text(protocol: str) -> str:
    return {
        "transductive": "Transductive (nodes may be shared; hyperedges and time disjoint)",
        "cold_start": "Inductive, cold-start (at least one unseen node per test hyperedge)",
        "strict_inductive": "Inductive, strict (no node shared with training)",
    }[protocol]


def table8_rows(per_model: Dict[str, Dict[str, float]]) -> List[Dict]:
    return [dict(method=k,
                 node_anomaly=round(v["node"], 1),
                 event_anomaly=round(v["event"], 1),
                 cross_modal_anomaly=round(v["cross_modal"], 1),
                 avg_dr=round(v["avg_dr"], 1),
                 ap=round(v["ap"], 3))
            for k, v in per_model.items()]


def table9_rows(per_method: Dict[str, Dict[str, float]]) -> List[Dict]:
    return [dict(method=k,
                 ap_injected=round(v["ap_inj"], 3), ap_natural=round(v["ap_nat"], 3),
                 auc_injected=round(v["auc_inj"], 3), auc_natural=round(v["auc_nat"], 3),
                 f1_injected=f"{100*v['f1_inj']:.1f}%", f1_natural=f"{100*v['f1_nat']:.1f}%")
            for k, v in per_method.items()]


def table10_rows(ablation: List[Dict], redundancy: Dict[str, float],
                 query_ms: Dict[str, float]) -> List[Dict]:
    rows = []
    for r in ablation:
        v = r["variant"]
        rows.append(dict(model_variant=v, f1=round(r["f1"], 1),
                         hr10=round(r.get("hr10", float("nan")), 1),
                         redundancy=round(redundancy.get(v, float("nan")), 1),
                         query_ms=round(query_ms.get(v, float("nan")), 1),
                         anomaly_dr=round(r.get("anomaly_dr", float("nan")), 1)))
    return rows


def table11_rows(per_model: Dict[str, Dict[str, float]]) -> List[Dict]:
    return [dict(model=k, f1=round(v["f1"], 3), hr10=round(v["hr10"], 3),
                 ap=round(v["ap"], 3), inference_ms=round(v["infer_ms"], 1),
                 peak_memory_gb=round(v["mem_gb"], 1))
            for k, v in per_model.items()]


def table12_rows(proposed: Sequence[float], baseline: Sequence[float],
                 metric: str, ci: Tuple[float, float]) -> Dict:
    p = paired_ttest(proposed, baseline)
    return dict(metric=metric, proposed=round(float(np.mean(proposed)), 3),
                baseline=round(float(np.mean(baseline)), 3),
                difference=round(float(np.mean(proposed) - np.mean(baseline)), 3),
                ci95=f"[{ci[0]:.3f}, {ci[1]:.3f}]", p_value=round(p, 3),
                cohens_d=round(cohens_d(proposed, baseline), 1))


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def write_csv(rows: Sequence[Dict], path: str) -> None:
    if not rows:
        print(f"[evaluate] nothing to write for {path}")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[evaluate] wrote {path}")


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=float)
    print(f"[evaluate] wrote {path}")


def markdown_table(rows: Sequence[Dict]) -> str:
    if not rows:
        return ""
    keys = list(rows[0].keys())
    out = ["| " + " | ".join(keys) + " |",
           "| " + " | ".join("---" for _ in keys) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    y = np.array([1, 1, 0, 0, 1, 0])
    s = np.array([0.9, 0.7, 0.4, 0.2, 0.8, 0.6])
    print("F1     :", round(f1_score(y, (s >= 0.5).astype(int)), 3))
    print("AP     :", round(average_precision(y, s), 3))
    print("AUC    :", round(roc_auc(y, s), 3))
    print("BH-FDR :", [round(x, 3) for x in benjamini_hochberg([0.01, 0.04, 0.20])])
