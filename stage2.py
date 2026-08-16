#!/usr/bin/env python3
"""
stage2.py -- Stage 2 primary estimand. VALIDATION ONLY. Test split stays sealed.

Pre-registered in the README on 2026-08-13, amended 2026-08-16 before any fitting.
This script implements that specification and nothing else. No falsification tests.

    residual = observed pass_success - M0x predicted probability

M0x is fitted on TRAIN and applied out of sample to VALIDATION. Every number below
is computed on validation.

Primary estimand
    Mean residual as a function of time since the press ended, on passes where
    local pressure is OFF but the segment has a pressure history, benchmarked
    against unpressed passes with no prior press in the segment.

H1  negative in the shortest elapsed bins, attenuating toward the benchmark as
    elapsed time grows. Attenuation tested as a TREND on log(1+elapsed); no
    bin-to-bin monotonicity is required.
H0  no dependence on elapsed time; indistinguishable from benchmark throughout.

Predicted-probability quintiles are cut on the pooled UNPRESSED validation sample
so the primary and benchmark samples share cut points. Note the relevant baseline
misfit for this sample is the UNPRESSED stratum (M0x max |diff| 0.0049 = 0.49 pp),
not the pressed stratum -- i.e. the baseline's own error sits right at the
practically-null floor, which is why the relevance floor is 1.0 pp.

    python stage2.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.load import load_passes, analysis_sample
from stage1_baseline import (build_design, fit_logit, predict_logit,
                             match_bucket, TRAIN_HI, VALID_HI, FEATURE_COLS)

# Frozen on 2026-08-16 from the train-split marginal distribution of the
# regressor, with no reference to any outcome.
BIN_EDGES = [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, np.inf]

FLOOR_PP = 1.0            # football-relevance floor, percentage points
PRACTICAL_NULL_PP = 0.5   # below this, reported as practically null

STAGE2_COLS = FEATURE_COLS + [
    "events_since_last_press", "time_since_last_press_spine",
    "time_since_last_press_s", "pass_ord_in_poss", "seg_press_count_spine",
]


def clustered_mean_se(r: np.ndarray, clusters: np.ndarray) -> float:
    """SE of a mean with observations clustered by match."""
    n = len(r)
    if n < 2:
        return float("nan")
    dev = r - r.mean()
    sums = pd.Series(dev).groupby(clusters).sum().to_numpy()
    return float(np.sqrt((sums ** 2).sum()) / n)


def cluster_ols(y: np.ndarray, X: np.ndarray, clusters: np.ndarray):
    """OLS with cluster-robust covariance."""
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    k = X.shape[1]
    meat = np.zeros((k, k))
    df = pd.DataFrame(X * e[:, None])
    for _, g in df.groupby(clusters, sort=False):
        sc = g.to_numpy().sum(axis=0)
        meat += np.outer(sc, sc)
    V = XtX_inv @ meat @ XtX_inv
    return b, np.sqrt(np.diag(V))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clock", default="spine", choices=["spine", "pass"],
                    help="primary uses the spine clock; 'pass' is the robustness row")
    args = ap.parse_args()

    clock_col = ("time_since_last_press_spine" if args.clock == "spine"
                 else "time_since_last_press_s")

    print("=" * 78)
    print("STAGE 2 -- PRIMARY ESTIMAND   (validation only; test sealed)")
    print(f"clock: {clock_col}")
    print("=" * 78)

    d = analysis_sample(load_passes(columns=STAGE2_COLS))
    d = d[d["pass_success"].notna()]
    b = d["match_id"].map(match_bucket)
    train = d[b < TRAIN_HI]
    valid = d[(b >= TRAIN_HI) & (b < VALID_HI)]

    Xtr, ytr = build_design(train, "M0x")
    print(f"fitting M0x on train: {len(Xtr):,} rows, {Xtr.shape[1]} columns")
    beta, _ = fit_logit(Xtr.to_numpy(), ytr)

    Xv, yv = build_design(valid, "M0x")
    Xv = Xv.reindex(columns=Xtr.columns, fill_value=0.0)
    p = predict_logit(Xv.to_numpy(), beta)
    v = valid.assign(p=p, resid=yv - p)
    print(f"validation: {len(v):,} passes, {v.match_id.nunique():,} matches")

    off = v[~v["under_pressure"].fillna(False)]
    prim = off[off["events_since_last_press"].notna() & off[clock_col].notna()]
    bench = off[off["events_since_last_press"].isna()]
    print(f"\n  pressure OFF at t          {len(off):>9,}")
    print(f"  PRIMARY  (prior press)     {len(prim):>9,}")
    print(f"  BENCHMARK (no prior press) {len(bench):>9,}")

    # quintiles cut on the pooled unpressed sample so both share cut points
    edges = np.quantile(off["p"].to_numpy(), np.linspace(0, 1, 6))
    edges[0], edges[-1] = -np.inf, np.inf
    for frame in (prim, bench):
        frame.insert(0, "q", np.digitize(frame["p"].to_numpy(), edges[1:-1]))
    prim = prim.assign(q=np.digitize(prim["p"].to_numpy(), edges[1:-1]))
    bench = bench.assign(q=np.digitize(bench["p"].to_numpy(), edges[1:-1]))

    bm = bench["resid"].to_numpy()
    bm_mean = bm.mean()
    bm_se = clustered_mean_se(bm, bench["match_id"].to_numpy())
    print(f"\n  benchmark mean residual {100*bm_mean:+.3f} pp "
          f"(clustered SE {100*bm_se:.3f})")
    bench_q = {q: g["resid"].mean() for q, g in bench.groupby("q")}

    # ---- the main view: elapsed bins x predicted-probability quintiles ----- #
    t = prim[clock_col].astype("Float64").astype(float).to_numpy()
    prim = prim.assign(bin=pd.cut(t, BIN_EDGES, right=False))

    print("\n" + "=" * 78)
    print("MEAN RESIDUAL vs BENCHMARK, in percentage points")
    print("  rows: seconds since the press ended.  columns: predicted-probability")
    print("  quintile of the same unpressed population (Q5 = easiest passes).")
    print("  Each cell subtracts that quintile's own benchmark, so baseline offset")
    print("  is differenced out within stratum.")
    print("=" * 78)
    hdr = f"  {'elapsed (s)':<12} {'n':>9} {'OVERALL':>10} {'SE':>7}  " + \
          "".join(f"{'Q'+str(k+1):>9}" for k in range(5))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for bl, g in prim.groupby("bin", observed=True):
        r = g["resid"].to_numpy()
        overall = 100 * (r.mean() - bm_mean)
        se = 100 * clustered_mean_se(r, g["match_id"].to_numpy())
        cells = []
        for q in range(5):
            gq = g[g["q"] == q]
            cells.append(100 * (gq["resid"].mean() - bench_q[q])
                         if len(gq) > 100 else np.nan)
        rows.append((str(bl), len(g), overall, se, cells))
        cs = "".join(f"{c:>+9.3f}" if np.isfinite(c) else f"{'-':>9}" for c in cells)
        print(f"  {str(bl):<12} {len(g):>9,} {overall:>+10.3f} {se:>7.3f}  {cs}")

    # ---- trend ------------------------------------------------------------ #
    print("\n" + "=" * 78)
    print("ATTENUATION TREND  (slope of residual on log(1+elapsed), ungrouped)")
    print("=" * 78)
    lt = np.log1p(prim[clock_col].astype("Float64").astype(float).to_numpy())
    X = np.column_stack([np.ones(len(lt)), lt])
    yv2 = prim["resid"].to_numpy() - bm_mean
    bb, ss = cluster_ols(yv2, X, prim["match_id"].to_numpy())
    print(f"  overall   intercept {100*bb[0]:+.3f} pp   slope {100*bb[1]:+.4f} pp "
          f"per log-second   (clustered SE {100*ss[1]:.4f}, t {bb[1]/ss[1]:+.2f})")
    print(f"\n  {'quintile':<10} {'n':>9} {'intercept':>11} {'slope':>10} {'SE':>8} {'t':>7}")
    slopes = {}
    for q in range(5):
        gq = prim[prim["q"] == q]
        if len(gq) < 500:
            continue
        l2 = np.log1p(gq[clock_col].astype("Float64").astype(float).to_numpy())
        X2 = np.column_stack([np.ones(len(l2)), l2])
        b2, s2 = cluster_ols(gq["resid"].to_numpy() - bench_q[q], X2,
                             gq["match_id"].to_numpy())
        slopes[q] = b2[1]
        print(f"  Q{q+1:<9} {len(gq):>9,} {100*b2[0]:>+11.3f} {100*b2[1]:>+10.4f}"
              f" {100*s2[1]:>8.4f} {b2[1]/s2[1]:>+7.2f}")

    # ---- pre-registered ADJUSTED estimate ---------------------------------- #
    # The pre-registration requires pass_ord_in_poss "as a control, AND results
    # stratified by it". The unadjusted table above only stratifies. It must be
    # controlled too, because the two samples are badly imbalanced on it:
    # primary mean ordinal 7.17 against benchmark 2.47, and M0x does not contain
    # pass_ord_in_poss, so the baseline is itself miscalibrated across position
    # (benchmark residual +3.57 pp at ordinal 0-1 against -1.17 pp at 2-3).
    # Without this control the contrast is partly early-possession versus
    # late-possession, not pressed-history versus not.
    print("\n" + "=" * 78)
    print("ADJUSTED ESTIMATE  (pass_ord_in_poss and quintile controlled)")
    print("  regression of residual on elapsed-bin dummies + ordinal dummies +")
    print("  quintile dummies, pooled over primary and benchmark; the benchmark is")
    print("  the omitted elapsed category, so each coefficient is that bin versus")
    print("  the benchmark holding possession position and difficulty fixed.")
    print("=" * 78)

    pool = pd.concat([
        prim.assign(_b=prim["bin"].astype(str)),
        bench.assign(_b="BENCHMARK"),
    ], ignore_index=True)
    ordc = pool["pass_ord_in_poss"].astype("Float64").astype(float).clip(0, 13)
    D_bin = pd.get_dummies(pool["_b"])
    bin_names = [c for c in D_bin.columns if c != "BENCHMARK"]
    D_bin = D_bin[bin_names]
    D_ord = pd.get_dummies(ordc.round().astype(int), prefix="o", drop_first=True)
    D_q = pd.get_dummies(pool["q"], prefix="q", drop_first=True)
    Xa = pd.concat([pd.Series(1.0, index=pool.index, name="const"),
                    D_bin, D_ord, D_q], axis=1).astype(float)
    ba, sa = cluster_ols(pool["resid"].to_numpy(), Xa.to_numpy(),
                         pool["match_id"].to_numpy())
    names = list(Xa.columns)
    order = sorted(bin_names, key=lambda s: float(s.split(",")[0].strip("[ ")))
    print(f"  {'elapsed (s)':<14} {'adjusted vs benchmark':>22} {'SE':>8} {'t':>8}")
    adj = {}
    for nm in order:
        i = names.index(nm)
        adj[nm] = 100 * ba[i]
        print(f"  {nm:<14} {100*ba[i]:>+22.3f} {100*sa[i]:>8.3f} {ba[i]/sa[i]:>+8.2f}")

    # ---- survivorship stratification -------------------------------------- #
    print("\n" + "=" * 78)
    print("SURVIVORSHIP: same estimand stratified by pass_ord_in_poss")
    print("=" * 78)
    ordv = prim["pass_ord_in_poss"].astype("Float64").astype(float)
    ob = pd.cut(ordv, [-0.1, 1.5, 3.5, 6.5, np.inf],
                labels=["0-1", "2-3", "4-6", "7+"])
    bench_ord = bench.assign(ob=pd.cut(
        bench["pass_ord_in_poss"].astype("Float64").astype(float),
        [-0.1, 1.5, 3.5, 6.5, np.inf], labels=["0-1", "2-3", "4-6", "7+"]))
    bo = {k: g["resid"].mean() for k, g in bench_ord.groupby("ob", observed=True)}
    short = prim[prim[clock_col].astype("Float64").astype(float) < 2.0]
    so = short.assign(ob=pd.cut(
        short["pass_ord_in_poss"].astype("Float64").astype(float),
        [-0.1, 1.5, 3.5, 6.5, np.inf], labels=["0-1", "2-3", "4-6", "7+"]))
    print("  restricted to the two shortest elapsed bins (< 2 s since the press)")
    print(f"  {'pass_ord':<10} {'n':>9} {'vs benchmark':>14} {'SE':>8}")
    ord_est = {}
    for k, g in so.groupby("ob", observed=True):
        est = 100 * (g["resid"].mean() - bo.get(k, bm_mean))
        se = 100 * clustered_mean_se(g["resid"].to_numpy(), g["match_id"].to_numpy())
        ord_est[str(k)] = est
        print(f"  {str(k):<10} {len(g):>9,} {est:>+14.3f} {se:>8.3f}")

    # ---- pre-registered decision rule ------------------------------------- #
    print("\n" + "=" * 78)
    print("PRE-REGISTERED DECISION RULE")
    print("=" * 78)
    two_short = prim[prim[clock_col].astype("Float64").astype(float) < 2.0]
    pooled = 100 * (two_short["resid"].mean() - bm_mean)
    pooled_se = 100 * clustered_mean_se(two_short["resid"].to_numpy(),
                                        two_short["match_id"].to_numpy())
    print(f"  pooled effect, two shortest bins: {pooled:+.3f} pp "
          f"(clustered SE {pooled_se:.3f}, n={len(two_short):,})")

    q_est = {}
    for q in range(5):
        gq = two_short[two_short["q"] == q]
        if len(gq) > 100:
            q_est[q] = 100 * (gq["resid"].mean() - bench_q[q])

    c1 = abs(pooled) >= FLOOR_PP
    n_neg_q = sum(1 for e in q_est.values() if e < 0)
    n_neg_o = sum(1 for e in ord_est.values() if e < 0)
    c2 = n_neg_q >= 4 and n_neg_o >= 3
    correct = [abs(e) for e in q_est.values() if e < 0]
    c3 = bool(correct) and min(correct) >= abs(pooled) / 3
    no_q5 = two_short[two_short["q"] != 4]
    pooled_no_q5 = 100 * (no_q5["resid"].mean()
                          - np.mean([bench_q[q] for q in range(4)]))
    c4 = abs(pooled) > 1e-9 and abs(pooled_no_q5 - pooled) / abs(pooled) < 0.5

    print(f"\n  1 magnitude   |pooled| >= {FLOOR_PP} pp"
          f"                      {'PASS' if c1 else 'FAIL'}"
          f"   ({abs(pooled):.3f} pp)")
    print(f"  2 sign        negative in >=4/5 quintiles, >=3/4 ordinals"
          f"   {'PASS' if c2 else 'FAIL'}   ({n_neg_q}/5, {n_neg_o}/4)")
    print(f"  3 magnitude consistency  min|stratum| >= pooled/3"
          f"        {'PASS' if c3 else 'FAIL'}")
    print(f"  4 not Q5-driven  drop-Q5 shifts pooled by <50%"
          f"          {'PASS' if c4 else 'FAIL'}"
          f"   (excl-Q5 {pooled_no_q5:+.3f} pp)")

    verdict = all([c1, c2, c3, c4])
    print(f"\n  ALL FOUR REQUIRED -> {'H1 SUPPORTED' if verdict else 'NULL'}")
    if abs(pooled) < PRACTICAL_NULL_PP:
        print(f"  |pooled| < {PRACTICAL_NULL_PP} pp: PRACTICALLY NULL regardless of "
              f"significance.")
    elif abs(pooled) < FLOOR_PP:
        print(f"  {PRACTICAL_NULL_PP} pp <= |pooled| < {FLOOR_PP} pp: BOUNDED NULL "
              f"-- detectable, below the relevance floor.")


if __name__ == "__main__":
    main()
