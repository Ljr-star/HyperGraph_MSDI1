# -*- coding: utf-8 -*-
"""
validity.py
===========
Hyperedge validity judgement and the structural-redundancy metrics of
Section 4.3 (Tables 3, 4 and 5).

Implemented definitions
-----------------------
(5) Mutual-information gain
        delta-MI(e) = I(X_H ; Y) - I(X_B ; Y)                          (Eq. 5)
    where X_H is the hyperedge representation, X_B its clique-expanded
    (binary) counterpart and Y the ground-truth association label.

(6) Binary-Decomposition Semantic Loss
        BD-SL = 1 - I(X_B ; Y) / I(X_H ; Y)                            (Eq. 6)
    -> 0 when the binary decomposition fully recovers the hypergraph
    semantics, -> 1 when none of it can be recovered.  This is the metric
    formerly named "Semantic Loss Rate".

(7) kNN mutual-information estimator (Kraskov-Stogbauer-Grassberger, KSG1)
        I = psi(k) - < psi(n_x + 1) + psi(n_y + 1) > + psi(N)          (Eq. 7)
    with k = 5 and the maximum-norm (Chebyshev) distance, computed on the
    128-dimensional aligned feature space.

(8) Per-hyperedge effectiveness score S(e_k), estimated on the LOCAL
    NEIGHBOURHOOD GROUP of e_k: the hyperedge itself plus its 255 nearest
    neighbours in the aligned feature space (cosine distance), restricted to
    the same cardinality class and the same 30-day sliding window, i.e. a
    group of 256 members.  Mutual information cannot be estimated from a
    single instance, which is why the group formulation is used.

(9) Conjunctive validity rule (this revision): a hyperedge is valid only when
    AT LEAST TWO of the three criteria hold
        (i)   relative MI gain  >= 5%     (denominator = I(X_B ; Y))
        (ii)  logistic-probe F1 gain >= 1.0 percentage point
        (iii) auto-encoder reconstruction error >= 8% lower.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import digamma
from scipy.spatial import cKDTree

from model import HyperAutoEncoder


# --------------------------------------------------------------------------
# (7) kNN mutual-information estimator
# --------------------------------------------------------------------------
def _chebyshev_p(metric: str):
    return np.inf if metric == "chebyshev" else 2


def ksg_mi(x: np.ndarray, y: np.ndarray, k: int = 5,
           metric: str = "chebyshev", max_samples: Optional[int] = None,
           seed: int = 0) -> float:
    """KSG1 estimator of I(X ; Y) with the maximum-norm distance.

    ``x`` : (N, d) representation, ``y`` : (N,) label vector.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    n = len(x)
    if max_samples is not None and n > max_samples:
        idx = np.random.default_rng(seed).choice(n, size=max_samples, replace=False)
        x, y = x[idx], y[idx]
        n = len(x)
    if n < k + 5:
        return 0.0

    p = _chebyshev_p(metric)
    joint = np.hstack([x, y])
    tv = cKDTree(joint).query(joint, k=k + 1, p=p)[0][:, k]
    eps = np.maximum(tv - 1e-9, 0.0)

    tree_x, tree_y = cKDTree(x), cKDTree(y)
    nx = np.empty(n)
    ny = np.empty(n)
    for i in range(n):
        nx[i] = len(tree_x.query_ball_point(x[i], eps[i], p=p)) - 1
        ny[i] = len(tree_y.query_ball_point(y[i], eps[i], p=p)) - 1

    return float(digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(n))


# --------------------------------------------------------------------------
# (5)(6) delta-MI and BD-SL
# --------------------------------------------------------------------------
def delta_mi(x_hyper: np.ndarray, x_binary: np.ndarray, y: np.ndarray,
             k: int = 5, metric: str = "chebyshev") -> Dict[str, float]:
    i_h = ksg_mi(x_hyper, y, k=k, metric=metric)
    i_b = ksg_mi(x_binary, y, k=k, metric=metric)
    i_h = max(i_h, 1e-9)
    i_b = max(i_b, 0.0)
    return dict(I_hyper=i_h, I_binary=i_b, delta_mi=i_h - i_b,
                bd_sl=float(1.0 - i_b / i_h))


# --------------------------------------------------------------------------
# (8) local neighbourhood group
# --------------------------------------------------------------------------
def neighbourhood_group(features: np.ndarray, cardinality: np.ndarray,
                        timestamp: np.ndarray, target: int,
                        group_size: int = 256, window_days: float = 30.0) -> np.ndarray:
    """Members of the local group of hyperedge ``target`` (definition (8))."""
    same_card = np.where(cardinality == cardinality[target])[0]
    same_win = same_card[np.abs(timestamp[same_card] - timestamp[target]) <= window_days]
    if len(same_win) < 4:
        same_win = same_card
    f = features[same_win]
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-8)
    sim = f @ f[target_pos := int(np.where(same_win == target)[0][0])]
    order = np.argsort(-sim)
    return same_win[order[: min(group_size, len(order))]]


# --------------------------------------------------------------------------
# (9) per-hyperedge criteria
# --------------------------------------------------------------------------
def mi_criterion(x_hyper, x_binary, y, cfg) -> Tuple[bool, float]:
    st = delta_mi(x_hyper, x_binary, y, k=cfg["knn_mi_k"], metric=cfg["knn_mi_metric"])
    if st["I_binary"] < cfg["group_min_baseline_mi"]:
        return False, 0.0                       # negligible baseline -> excluded
    rel = st["delta_mi"] / st["I_binary"]
    return bool(rel >= cfg["mi_relative_gain"]), float(rel)


def probe_criterion(x_hyper, x_binary, y, cfg) -> Tuple[bool, float]:
    """Logistic-probe F1 gain (5-fold cross-validation, identical folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    gain = _cv_f1_gain(x_hyper, x_binary, y, LogisticRegression,
                       dict(max_iter=cfg["probe_max_iter"], solver="lbfgs"),
                       n_splits=cfg["probe_folds"])
    return bool(gain >= cfg["probe_f1_gain"]), float(gain)


def _cv_f1_gain(xh, xb, y, estimator_cls, kwargs, n_splits=5) -> float:
    y = np.asarray(y)
    if len(np.unique(y)) < 2 or len(y) < 4 * n_splits:
        return 0.0
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    f1h, f1b = [], []
    for tr, te in skf.split(xh, y):
        for x, store in ((xh, f1h), (xb, f1b)):
            clf = estimator_cls(**kwargs).fit(x[tr], y[tr])
            store.append(_f1(y[te], clf.predict(x[te])))
    return float(np.mean(f1h) - np.mean(f1b))


def _f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    if tp == 0:
        return 0.0
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


def autoencoder_criterion(x_hyper, x_binary, y, cfg, device="cpu") -> Tuple[bool, float]:
    """Relative reduction of the reconstruction error (criterion iii)."""
    err_h = _ae_error(x_hyper, cfg, device)
    err_b = _ae_error(x_binary, cfg, device)
    if err_h <= 0:
        return False, 0.0
    gain = (err_b - err_h) / err_b if err_b > 0 else 0.0
    return bool(gain >= cfg["ae_error_gain"]), float(gain)


def _ae_error(x: np.ndarray, cfg: Dict, device="cpu") -> float:
    x = np.asarray(x, dtype=np.float32)
    if len(x) < 16:
        return 0.0
    t = torch.tensor(x, device=device)
    mu, sd = t.mean(0, keepdim=True), t.std(0, keepdim=True).clamp_min(1e-6)
    t = (t - mu) / sd
    ae = HyperAutoEncoder(t.shape[1], cfg["ae_hidden_dim"]).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    for _ in range(cfg["ae_epochs"]):
        opt.zero_grad()
        loss = F.mse_loss(ae(t), t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        return float(F.mse_loss(ae(t), t).item())


def judge_hyperedge(x_hyper, x_binary, y, cfg, device="cpu") -> Dict[str, float]:
    """Evaluate the three criteria + the conjunctive rule for one hyperedge group."""
    c1, g1 = mi_criterion(x_hyper, x_binary, y, cfg)
    c2, g2 = probe_criterion(x_hyper, x_binary, y, cfg)
    c3, g3 = autoencoder_criterion(x_hyper, x_binary, y, cfg, device)
    votes = int(c1) + int(c2) + int(c3)
    return dict(mi_only=int(c1), probe_only=int(c2), ae_only=int(c3),
                mi_gain=g1, probe_gain=g2, ae_gain=g3, votes=votes,
                valid_conjunctive=int(votes >= cfg["conjunctive_min_votes"]),
                valid_disjunctive=int(votes >= 1))


# --------------------------------------------------------------------------
# Table 4 - candidate-similarity statistics and threshold calibration
# --------------------------------------------------------------------------
def table4_row(dataset: str, calib: Dict) -> Dict:
    """Assemble one row of Table 4 from a calibration record."""
    return dict(
        dataset=dataset,
        similarity_range=f"[{calib['sim_range'][0]:.2f}, {calib['sim_range'][1]:.2f}]",
        q1_median_q3=f"{calib['q1']:.2f} / {calib['median']:.2f} / {calib['q3']:.2f}",
        iqr=f"{calib['iqr']:.2f}",
        quantile_threshold=f"p = {calib['quantile_p']:.2f}, tau = {calib['quantile_tau']:.2f}",
        iqr_baseline_threshold=(f"lambda = {calib['iqr_lambda']:.2f}, "
                                f"eta = {calib['iqr_eta']:.2f}, tau = {calib['iqr_tau']:.2f}"),
        acceptance_rate=f"{100.0 * calib['acceptance_rate']:.1f}%")


# --------------------------------------------------------------------------
# Table 5 - per-criterion judgements, intersections, agreement with manual
# --------------------------------------------------------------------------
def table5(dataset: str, judgments: List[Dict], manual_labels: np.ndarray) -> Dict:
    """Aggregate per-hyperedge judgements into the Table 5 row set."""
    n = len(judgments)
    if n == 0:
        return dict(dataset=dataset, rows=[])
    j = {k: np.asarray([d[k] for d in judgments]) for k in
         ("mi_only", "probe_only", "ae_only", "valid_conjunctive")}
    union = ((j["mi_only"] + j["probe_only"] + j["ae_only"]) >= 1).astype(int)

    def pct(a):
        return round(100.0 * float(a.mean()), 1)

    rows = [
        ("MI gain only", pct(j["mi_only"]), j["mi_only"]),
        ("Logistic probe only", pct(j["probe_only"]), j["probe_only"]),
        ("Auto-encoder error only", pct(j["ae_only"]), j["ae_only"]),
        ("Union of the three (previous rule)", pct(union), union),
        ("MI intersection with probe", pct(j["mi_only"] & j["probe_only"]),
         j["mi_only"] & j["probe_only"]),
        ("MI intersection with auto-encoder", pct(j["mi_only"] & j["ae_only"]),
         j["mi_only"] & j["ae_only"]),
        ("At least two of three (adopted)", pct(j["valid_conjunctive"]), j["valid_conjunctive"]),
    ]
    out = []
    for name, value, pred in rows:
        prec, rec = _precision_recall(manual_labels, pred)
        out.append(dict(judgment_rule=name, value_pct=value,
                        manual_precision=round(prec, 3), manual_recall=round(rec, 3)))
    return dict(dataset=dataset, rows=out)


def _precision_recall(manual: np.ndarray, pred: np.ndarray) -> Tuple[float, float]:
    manual = np.asarray(manual)
    tp = float(((manual == 1) & (pred == 1)).sum())
    fp = float(((manual == 0) & (pred == 1)).sum())
    fn = float(((manual == 1) & (pred == 0)).sum())
    return tp / (tp + fp + 1e-9), tp / (tp + fn + 1e-9)


def threshold_sensitivity(scores: np.ndarray, labels: np.ndarray, tau: float,
                          sweep: float = 0.20) -> Dict[str, float]:
    """Table 5 note: sweeping every threshold by +-20% changes the Valid Ratio
    by less than 1.8 percentage points."""
    vals = []
    for mult in (1.0 - sweep, 1.0, 1.0 + sweep):
        t = tau * mult
        vals.append(float((scores >= t).mean()))
    return dict(base=vals[1], low=vals[0], high=vals[2],
                max_abs_change_pp=100.0 * max(abs(vals[0] - vals[1]), abs(vals[2] - vals[1])))


# --------------------------------------------------------------------------
# Table 3 - structural redundancy of one generation strategy
# --------------------------------------------------------------------------
def strategy_row(strategy: str, scale_label: str, x_hyper: np.ndarray,
                 x_binary: np.ndarray, y: np.ndarray, valid_ratio: float,
                 cfg: Dict) -> Dict:
    st = delta_mi(x_hyper, x_binary, y, k=cfg["knn_mi_k"], metric=cfg["knn_mi_metric"])
    return dict(method=strategy, hyperedge_scale=scale_label,
                valid_ratio=round(100.0 * valid_ratio, 1),
                redundancy_rate=round(100.0 * (1.0 - valid_ratio), 1),
                delta_mi=round(st["delta_mi"], 3),
                bd_sl=round(100.0 * st["bd_sl"], 1))


def hyperedge_representation(features: np.ndarray, edges: Sequence[Sequence[int]],
                             edge_ids: Sequence[int], mode: str = "hyper",
                             max_pairs: int = 32, seed: int = 0) -> np.ndarray:
    """Feature representation of a hyperedge set.

    mode = 'hyper'  : the hyperedge attribute vector h_e (mean of the member
                      features), i.e. X_H of Eq. (5);
    mode = 'binary' : the clique-expanded counterpart X_B, obtained by averaging
                      the midpoint features of all C(k,2) pairwise edges;
    mode = 'pairwise_lift' / 'star' : the alternative expansions of Section 4.3(i).
    """
    rng = np.random.default_rng(seed)
    out = np.zeros((len(edge_ids), features.shape[1]), dtype=np.float64)
    for r, eid in enumerate(edge_ids):
        e = list(edges[eid])
        if not e:
            continue
        if mode == "hyper":
            out[r] = features[e].mean(axis=0)
            continue
        if mode == "binary":
            pairs = [(e[i], e[j]) for i in range(len(e)) for j in range(i + 1, len(e))]
        elif mode == "pairwise_lift":
            pairs = [(e[i - 1], e[i]) for i in range(1, len(e))] or [(e[0], e[0])]
        else:                                             # star expansion
            pairs = [(e[0], e[j]) for j in range(1, len(e))] or [(e[0], e[0])]
        if len(pairs) > max_pairs:
            sel = rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[i] for i in sel]
        acc = np.zeros(features.shape[1], dtype=np.float64)
        for u, v in pairs:
            acc += 0.5 * (features[u] + features[v])
        out[r] = acc / max(1, len(pairs))
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d = 600, 32
    y = rng.integers(0, 2, size=n)
    xh = rng.normal(0, 1, size=(n, d)) + 1.4 * y[:, None]      # informative
    xb = rng.normal(0, 1, size=(n, d)) + 0.35 * y[:, None]     # weakly informative
    print("KSG MI (hyper) :", round(ksg_mi(xh, y), 4))
    print("KSG MI (binary):", round(ksg_mi(xb, y), 4))
    print("delta-MI / BD-SL:", {k: round(v, 4) for k, v in delta_mi(xh, xb, y).items()})
