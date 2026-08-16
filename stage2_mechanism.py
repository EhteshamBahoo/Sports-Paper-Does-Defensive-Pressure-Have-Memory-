#!/usr/bin/env python3
"""
stage2_mechanism.py -- can the Stage 2 residual be explained away? TWO THREATS.

VALIDATION ONLY. The test split is not read by this script.

Stage 2 found the residual of an unpressed pass depends on how long ago the press
ended: -6.4 pp adjusted at under a second, decaying to about -0.4 pp beyond 12 s.
Two mechanisms other than memory can produce exactly that, and they are not the
same threat, so each gets its own test.

THREAT A -- the press had not actually ended
    "The press ended" is defined by the last ball event carrying `under_pressure`.
    But a Pressure event is a window, not an instant: over the full corpus its
    duration runs p50 0.729 s, p90 1.764 s, p99 4.005 s, max 9.195 s. A pass
    logged 0.4 s after a pressed carry and flagged unpressed can still lie inside
    a live pressure window. That is contemporaneous pressure the event flag failed
    to carry forward -- measurement, not memory.

    Test: rebuild the pressure windows from the raw Pressure events (see
    pressures.py), independently of `related_events`, and re-run the estimand with
    every pass inside a live window removed. Threat A can only reach the shortest
    bins; it cannot reach 2-8 s, because almost no pressure window is that long.

THREAT B -- unmeasured contemporaneous defensive state
    M0x controls for pressure through the annotator flag and the linked presser's
    distance. It does not control for the defending team's shape. When a press
    ends the defence is often still compact, still advanced, still dense around
    the receiver, and that condition persists for several seconds -- precisely the
    window the effect lives in. If that is what is happening, the residual is not
    history predicting the future; it is the present, unobserved. Unlike threat A,
    this one reaches the whole elapsed range.

    Test: the 360 freeze frames give actual defender geometry at time t. Add it as
    controls and re-run. Survives -> memory in the strong sense. Vanishes -> the
    effect is contemporaneous defensive state that the annotator flag misses.

    Note the coverage arithmetic. A frame is needed at t only; the history
    variable comes from Tier 1 event data. This is the single-frame gate, not the
    frame-chain gate.

Either outcome for threat B is reportable, but they are different papers, so the
answer has to be known before anything is written.

    python stage2_mechanism.py
    python stage2_mechanism.py --clock pass     # robustness clock
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.load import load_passes, analysis_sample, assert_pre_treatment
from stage1_baseline import (build_design, fit_logit, predict_logit,
                             match_bucket, TRAIN_HI, VALID_HI, FEATURE_COLS)
from stage2 import (BIN_EDGES, FLOOR_PP, clustered_mean_se, cluster_ols)

ROOT = Path(__file__).resolve().parent
PRESSURES = ROOT / "data" / "processed" / "pressures.parquet"

FF_COLS = [
    "ff_available", "ff_n_opp_visible", "ff_n_team_visible", "ff_visible_r5",
    "ff_recv_visible_r5", "ff_lane_visible", "ff_nearest_opp_dist",
    "ff_opp_within_3", "ff_opp_within_5", "ff_lane_opp", "ff_recv_opp_within_5",
]

MECH_COLS = FEATURE_COLS + [
    "events_since_last_press", "time_since_last_press_spine",
    "time_since_last_press_s", "pass_ord_in_poss", "period", "period_seconds",
    "opponent_team_id",
] + FF_COLS

# Distance from the pass origin, in metres, inside which a still-live pressure is
# treated as plausibly acting on THIS carrier rather than on a team-mate
# elsewhere on the pitch. Reported as a ladder; the headline exclusion uses no
# radius at all, which is the strictest test.
NEAR_RADII = (5.0, 10.0, 15.0)

ORD_EDGES = [-0.1, 1.5, 3.5, 6.5, np.inf]
ORD_LABELS = ["0-1", "2-3", "4-6", "7+"]


def mirror_xy(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate into the opposing team's attacking frame. See build.mirror."""
    return 120.0 - x, 80.0 - y


# --------------------------------------------------------------------------- #
# threat A: live pressure windows
# --------------------------------------------------------------------------- #
def attach_pressure_windows(v: pd.DataFrame) -> pd.DataFrame:
    """For each pass, is an opponent Pressure window still open at kick moment?

    Two quantities per pass:
      gap_to_window_end  seconds between the pass and the LATEST end among all
                         preceding opponent pressures. <= 0 means still inside one.
      cover_dist         distance from the pass origin to the nearest covering
                         presser, mirrored into the passer's frame. NULL when no
                         window is open.
    """
    if not PRESSURES.exists():
        raise SystemExit("[mech] data/processed/pressures.parquet missing. "
                         "Run `python pressures.py` first.")
    pr = pd.read_parquet(PRESSURES, dtype_backend="numpy_nullable")
    n_raw = len(pr)
    pr = pr[pr["duration_s"].notna() & pr["x"].notna()]
    pr = pd.DataFrame({
        "match_id": pr["match_id"].astype("int64"),
        "period": pr["period"].astype("int64"),
        "team_id": pr["team_id"].astype("int64"),
        "t": pr["period_seconds"].astype("float64"),
        "d": pr["duration_s"].astype("float64"),
        "px": pr["x"].astype("float64"),
        "py": pr["y"].astype("float64"),
    })
    pr["t_end"] = pr["t"] + pr["d"]
    print(f"  pressure events {n_raw:,} raw, {len(pr):,} with duration and location")
    print(f"  duration p50 {pr.d.median():.3f}s  p90 {pr.d.quantile(.9):.3f}s  "
          f"p99 {pr.d.quantile(.99):.3f}s  max {pr.d.max():.3f}s")

    pr = pr.sort_values(["match_id", "period", "team_id", "t"], kind="stable")
    # running max of window end: "has ANY earlier pressure not yet expired"
    pr["cummax_end"] = pr.groupby(["match_id", "period", "team_id"])["t_end"].cummax()

    left = pd.DataFrame({
        "_i": np.arange(len(v)),
        "match_id": v["match_id"].astype("int64").to_numpy(),
        "period": v["period"].astype("int64").to_numpy(),
        "team_id": v["opponent_team_id"].astype("int64").to_numpy(),
        "t": v["period_seconds"].astype("float64").to_numpy(),
        "x": v["x"].astype("float64").to_numpy(),
        "y": v["y"].astype("float64").to_numpy(),
    }).sort_values("t", kind="stable")

    m = pd.merge_asof(
        left, pr.sort_values("t", kind="stable")[
            ["match_id", "period", "team_id", "t", "cummax_end"]],
        on="t", by=["match_id", "period", "team_id"],
        direction="backward", allow_exact_matches=True, suffixes=("", "_pr"),
    ).sort_values("_i", kind="stable")

    gap = m["t"].to_numpy() - m["cummax_end"].to_numpy()   # NaN if no prior press
    live = np.isfinite(gap) & (gap <= 0.0)

    # exact covering presser (and its distance) only where a window is open
    cover_dist = np.full(len(v), np.nan)
    groups = {k: g for k, g in pr.groupby(["match_id", "period", "team_id"], sort=False)}
    L = left.sort_values("_i", kind="stable")
    mid = L["match_id"].to_numpy(); per = L["period"].to_numpy()
    tid = L["team_id"].to_numpy(); tt = L["t"].to_numpy()
    xx = L["x"].to_numpy(); yy = L["y"].to_numpy()
    for i in np.flatnonzero(live):
        g = groups.get((mid[i], per[i], tid[i]))
        if g is None:
            continue
        gt = g["t"].to_numpy(); ge = g["t_end"].to_numpy()
        sel = (gt <= tt[i]) & (ge >= tt[i])
        if not sel.any():
            continue
        mx, my = mirror_xy(g["px"].to_numpy()[sel], g["py"].to_numpy()[sel])
        cover_dist[i] = float(np.min(np.hypot(mx - xx[i], my - yy[i])))

    out = v.copy()
    out["gap_to_window_end"] = gap
    out["live_press"] = live
    out["cover_dist"] = cover_dist
    return out


# --------------------------------------------------------------------------- #
# threat B: freeze-frame geometry at t
# --------------------------------------------------------------------------- #
def _count_dummies(s: pd.Series, top: int, prefix: str) -> pd.DataFrame:
    """Small-integer counts as dummies, top-coded. Linear-in-count is a functional
    form assumption a referee will not grant when the whole claim rests on the
    control having had a fair chance to absorb the effect."""
    c = s.astype("Float64").astype(float)
    c = np.where(np.isnan(c), -1.0, np.minimum(c, top))
    return pd.get_dummies(pd.Series(c, index=s.index).astype(int),
                          prefix=prefix, drop_first=True)


DIST_EDGES = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, np.inf]

# Which freeze-frame columns are safe to condition on.
#
# `build._add_360` computes some features around `end_location`. For a FAILED
# pass StatsBomb sets end_location to the INTERCEPTION POINT, so lane occupancy
# and receiver congestion are measured around wherever the ball was actually
# stopped. Those columns are post-treatment in exactly the way pass_length is
# (README caveat 4), and conditioning on them would let the "control" absorb the
# outcome rather than the defensive state. Adding them would make the history
# effect shrink for a reason that has nothing to do with contemporaneous shape.
#
#   ORIGIN-ONLY (pre-treatment)  nearest_opp_dist, opp_within_3, opp_within_5,
#                                n_opp_visible, n_team_visible, visible_r5
#   TARGET-DERIVED (post)        lane_opp, recv_opp_within_5, recv_visible_r5,
#                                lane_visible
#
# The defensible arm is origin-only. The target-derived arm is reported only to
# bound how much of the apparent absorption is leakage.
FF_POST_COLS = ("ff_lane_opp", "ff_recv_opp_within_5", "ff_recv_visible_r5",
                "ff_lane_visible")


def add_ff_block(X: pd.DataFrame, d: pd.DataFrame,
                 include_post: bool = False) -> pd.DataFrame:
    """Append observed defensive geometry at the moment of the pass.

    include_post=False keeps only quantities measured around the BALL ORIGIN,
    which is fixed before the outcome is realised.
    """
    reads = ["ff_nearest_opp_dist", "ff_opp_within_3", "ff_opp_within_5",
             "ff_n_opp_visible", "ff_n_team_visible"]
    if include_post:
        reads += list(FF_POST_COLS)
    else:
        # fires if anyone adds a target-derived column to the clean arm, which is
        # precisely the mistake this block was rebuilt to fix
        assert_pre_treatment(reads)

    nod = d["ff_nearest_opp_dist"].astype("Float64").astype(float)
    blocks = [
        X,
        # same encoding as inv_presser_dist: 0 means "no opponent, arbitrarily far"
        pd.DataFrame({"ff_inv_nearest": np.where(np.isnan(nod), 0.0,
                                                 1.0 / (1.0 + nod))}, index=X.index),
        pd.get_dummies(pd.cut(nod, DIST_EDGES), prefix="fd", drop_first=True),
        _count_dummies(d["ff_opp_within_3"], 3, "f3"),
        _count_dummies(d["ff_opp_within_5"], 6, "f5"),
        pd.DataFrame({
            "ff_n_opp": (d["ff_n_opp_visible"].astype("Float64").astype(float)
                         .fillna(0.0) - 8.0) / 4.0,
            "ff_n_team": (d["ff_n_team_visible"].astype("Float64").astype(float)
                          .fillna(0.0) - 8.0) / 4.0,
        }, index=X.index),
    ]
    if include_post:
        blocks += [
            _count_dummies(d["ff_lane_opp"], 3, "fl"),
            _count_dummies(d["ff_recv_opp_within_5"], 4, "fr"),
            pd.DataFrame({
                "ff_recv_vis": d["ff_recv_visible_r5"].fillna(False).astype(float),
                "ff_lane_vis": d["ff_lane_visible"].fillna(False).astype(float),
            }, index=X.index),
        ]
    return pd.concat(blocks, axis=1).astype(np.float32)


def fit_and_score(train: pd.DataFrame, part: pd.DataFrame, spec: str,
                  with_ff: bool, include_post: bool = False, min_nz: int = 30):
    """Fit on `train`, return out-of-sample predictions for `part`.

    Columns with almost no support in train are dropped; the Tier 2 subsample is
    ~10% of matches and carries competition dummies that are otherwise near-empty.
    Both arms are pruned identically so the comparison stays like-for-like.
    """
    Xtr, ytr = build_design(train, spec)
    if with_ff:
        Xtr = add_ff_block(Xtr, train, include_post)
    keep = [c for c in Xtr.columns
            if c == "const" or (Xtr[c].to_numpy() != 0).sum() >= min_nz]
    Xtr = Xtr[keep]
    beta, _ = fit_logit(Xtr.to_numpy(), ytr)

    Xp, yp = build_design(part, spec)
    if with_ff:
        Xp = add_ff_block(Xp, part, include_post)
    Xp = Xp.reindex(columns=Xtr.columns, fill_value=0.0)
    return predict_logit(Xp.to_numpy(), beta), yp, Xtr.shape[1]


# --------------------------------------------------------------------------- #
# the estimand, reusable
# --------------------------------------------------------------------------- #
def estimand(prim: pd.DataFrame, bench: pd.DataFrame, clock_col: str,
             label: str, edges: np.ndarray | None = None,
             show_table: bool = True) -> dict:
    """Adjusted and unadjusted profile of residual against elapsed time."""
    if edges is None:
        pooled_p = np.concatenate([prim["p"].to_numpy(), bench["p"].to_numpy()])
        edges = np.quantile(pooled_p, np.linspace(0, 1, 6))
        edges[0], edges[-1] = -np.inf, np.inf
    prim = prim.assign(q=np.digitize(prim["p"].to_numpy(), edges[1:-1]))
    bench = bench.assign(q=np.digitize(bench["p"].to_numpy(), edges[1:-1]))

    bm = bench["resid"].to_numpy()
    bm_mean = bm.mean()
    bench_q = {q: g["resid"].mean() for q, g in bench.groupby("q")}

    t = prim[clock_col].astype("Float64").astype(float).to_numpy()
    prim = prim.assign(bin=pd.cut(t, BIN_EDGES, right=False))

    # ---- adjusted: elapsed-bin + ordinal + quintile dummies, benchmark omitted
    pool = pd.concat([prim.assign(_b=prim["bin"].astype(str)),
                      bench.assign(_b="BENCHMARK")], ignore_index=True)
    ordc = pool["pass_ord_in_poss"].astype("Float64").astype(float).clip(0, 13)
    D_bin = pd.get_dummies(pool["_b"])
    bin_names = [c for c in D_bin.columns if c != "BENCHMARK"]
    Xa = pd.concat([
        pd.Series(1.0, index=pool.index, name="const"),
        D_bin[bin_names],
        pd.get_dummies(ordc.round().astype(int), prefix="o", drop_first=True),
        pd.get_dummies(pool["q"], prefix="q", drop_first=True),
    ], axis=1).astype(float)
    ba, sa = cluster_ols(pool["resid"].to_numpy(), Xa.to_numpy(),
                         pool["match_id"].to_numpy())
    names = list(Xa.columns)
    order = sorted(bin_names, key=lambda s: float(s.split(",")[0].strip("[ ")))
    adj = {nm: (100 * ba[names.index(nm)], 100 * sa[names.index(nm)]) for nm in order}

    if show_table:
        print(f"\n  {label}")
        print(f"  primary {len(prim):,}   benchmark {len(bench):,}   "
              f"benchmark mean residual {100*bm_mean:+.3f} pp")
        print(f"  {'elapsed (s)':<14} {'n':>9} {'raw':>9} {'SE':>7} "
              f"{'ADJUSTED':>10} {'SE':>7} {'t':>7}")
        for nm in order:
            g = prim[prim["bin"].astype(str) == nm]
            raw = 100 * (g["resid"].mean() - bm_mean)
            se = 100 * clustered_mean_se(g["resid"].to_numpy(),
                                         g["match_id"].to_numpy())
            a, s = adj[nm]
            print(f"  {nm:<14} {len(g):>9,} {raw:>+9.3f} {se:>7.3f} "
                  f"{a:>+10.3f} {s:>7.3f} {a/s:>+7.2f}")

    # ---- pooled two shortest bins, raw and adjusted ----------------------- #
    short = prim[t < 2.0]
    pooled = 100 * (short["resid"].mean() - bm_mean)
    pooled_se = 100 * clustered_mean_se(short["resid"].to_numpy(),
                                        short["match_id"].to_numpy())
    pooled_adj = float(np.mean([adj[nm][0] for nm in order[:2]]))

    # ---- 2-8 s band: beyond any plausible pressure window ------------------ #
    band = prim[(t >= 2.0) & (t < 8.0)]
    band_raw = 100 * (band["resid"].mean() - bm_mean)
    band_se = 100 * clustered_mean_se(band["resid"].to_numpy(),
                                      band["match_id"].to_numpy())
    band_adj = float(np.mean([adj[nm][0] for nm in order[2:5]]))

    # ---- attenuation trend ------------------------------------------------- #
    lt = np.log1p(t)
    b2, s2 = cluster_ols(prim["resid"].to_numpy() - bm_mean,
                         np.column_stack([np.ones(len(lt)), lt]),
                         prim["match_id"].to_numpy())

    return {"adj": adj, "order": order, "pooled": pooled, "pooled_se": pooled_se,
            "pooled_adj": pooled_adj, "band_raw": band_raw, "band_se": band_se,
            "band_adj": band_adj, "slope": 100 * b2[1], "slope_se": 100 * s2[1],
            "slope_t": b2[1] / s2[1], "n_prim": len(prim), "n_bench": len(bench),
            "bench_mean": 100 * bm_mean, "edges": edges}


def split_samples(v: pd.DataFrame, clock_col: str):
    off = v[~v["under_pressure"].fillna(False)]
    prim = off[off["events_since_last_press"].notna() & off[clock_col].notna()]
    bench = off[off["events_since_last_press"].isna()]
    return prim, bench


def brier(y, p):
    b = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y.mean() - y) ** 2))
    return b, 1 - b / ref


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clock", default="spine", choices=["spine", "pass"])
    args = ap.parse_args()
    clock_col = ("time_since_last_press_spine" if args.clock == "spine"
                 else "time_since_last_press_s")

    print("=" * 78)
    print("STAGE 2 MECHANISM TESTS   (validation only; test sealed)")
    print(f"clock: {clock_col}")
    print("=" * 78)

    d = analysis_sample(load_passes(columns=MECH_COLS))
    d = d[d["pass_success"].notna()]
    b = d["match_id"].map(match_bucket)
    train = d[b < TRAIN_HI]
    valid = d[(b >= TRAIN_HI) & (b < VALID_HI)]

    p, yv, ncol = fit_and_score(train, valid, "M0x", with_ff=False)
    v = valid.assign(p=p, resid=yv - p)
    print(f"M0x fitted on train ({len(train):,} rows, {ncol} cols), applied to "
          f"validation ({len(v):,} passes / {v.match_id.nunique():,} matches)")

    # ====================================================================== #
    print("\n" + "=" * 78)
    print("THREAT A -- WAS THE PRESS ACTUALLY OVER?")
    print("  Pressure windows rebuilt from raw Pressure events, independent of")
    print("  related_events. A pass is 'live' if ANY preceding opponent pressure")
    print("  window is still open at the moment it is played.")
    print("=" * 78)

    v = attach_pressure_windows(v)

    # ---- instrument validation -------------------------------------------- #
    # A near-zero live share is only evidence about threat A if the join actually
    # fires. Passes the annotator flagged `under_pressure` must come back mostly
    # live; if they do not, the window measurement is broken and a null here means
    # nothing. This is the same lesson as the frame-consistency check: an
    # instrument that returns "no problem" has to be shown capable of returning
    # "problem" first.
    up = v["under_pressure"].fillna(False).to_numpy()
    lv = v["live_press"].to_numpy()
    gp = v["gap_to_window_end"].to_numpy()
    print(f"\n  instrument validation -- live-window rate by annotator flag")
    print(f"    under_pressure = True   n={int(up.sum()):>8,}   live "
          f"{100*lv[up].mean():>6.2f}%   median gap {np.nanmedian(gp[up]):+.3f}s")
    print(f"    under_pressure = False  n={int((~up).sum()):>8,}   live "
          f"{100*lv[~up].mean():>6.2f}%   median gap {np.nanmedian(gp[~up]):+.3f}s")
    if lv[up].mean() < 0.30:
        print("    [WARN] pressed passes are not mostly inside a live window; the")
        print("           window join is suspect and threat A is NOT cleared.")

    prim, bench = split_samples(v, clock_col)
    t = prim[clock_col].astype("Float64").astype(float).to_numpy()
    prim = prim.assign(bin=pd.cut(t, BIN_EDGES, right=False))

    print(f"\n  live-window contamination by elapsed bin (primary sample, "
          f"n={len(prim):,})")
    print(f"  'gap' is seconds from the latest pressure-window close to the pass;")
    print(f"  negative means still inside a window.")
    print(f"  {'elapsed (s)':<14} {'n':>9} {'live %':>8} {'<0.5s %':>9} "
          f"{'<1s %':>8} {'med gap':>9} {'cover dist':>11}")
    for nm, g in prim.groupby("bin", observed=True):
        lvb = g["live_press"].to_numpy()
        cd = g["cover_dist"].to_numpy()
        gpb = g["gap_to_window_end"].to_numpy()
        fin = np.isfinite(gpb)
        print(f"  {str(nm):<14} {len(g):>9,} {100*lvb.mean():>8.2f} "
              f"{100*np.mean(fin & (gpb <= 0.5)):>9.2f} "
              f"{100*np.mean(fin & (gpb <= 1.0)):>8.2f} "
              f"{np.nanmedian(gpb[fin]):>+9.2f} "
              f"{(np.nanmedian(cd[lvb]) if lvb.any() else float('nan')):>11.2f}")

    lv_all = prim["live_press"].to_numpy()
    print(f"\n  overall live share of the primary sample {100*lv_all.mean():.2f}% "
          f"({int(lv_all.sum()):,} passes)")
    bl = bench["live_press"].to_numpy()
    print(f"  benchmark live share {100*bl.mean():.2f}%  -- the benchmark is not "
          f"press-free either, so this is a contrast of degree")

    print("\n  " + "-" * 74)
    print("  ESTIMAND WITH LIVE-WINDOW PASSES REMOVED (both samples)")
    print("  " + "-" * 74)
    base = estimand(prim, bench, clock_col, "A0. all passes (reproduces stage2.py)")
    clean = estimand(prim[~prim["live_press"]], bench[~bench["live_press"]],
                     clock_col, "A1. live-window passes removed, no radius "
                                "(strictest)")

    ladder = {}
    for r in NEAR_RADII:
        pm = prim["live_press"] & (prim["cover_dist"] <= r)
        bmk = bench["live_press"] & (bench["cover_dist"] <= r)
        ladder[r] = estimand(prim[~pm.fillna(False)], bench[~bmk.fillna(False)],
                             clock_col, f"A2. removed only if covering presser "
                                        f"within {r:.0f} m", show_table=False)

    print(f"\n  {'exclusion':<38} {'n prim':>9} {'<2s adj':>9} {'2-8s adj':>10} "
          f"{'slope t':>9}")
    for nm, res in [("none", base), ("live window, any distance", clean)] + \
                   [(f"live window, presser < {r:.0f} m", ladder[r]) for r in NEAR_RADII]:
        print(f"  {nm:<38} {res['n_prim']:>9,} {res['pooled_adj']:>+9.3f} "
              f"{res['band_adj']:>+10.3f} {res['slope_t']:>+9.2f}")

    # ====================================================================== #
    print("\n" + "=" * 78)
    print("THREAT B -- UNMEASURED CONTEMPORANEOUS DEFENSIVE STATE")
    print("  360 freeze-frame geometry at time t, added as controls.")
    print("=" * 78)

    ff_tr = train[train["ff_available"].fillna(False)]
    ff_va = valid[valid["ff_available"].fillna(False)]
    print(f"\n  frame available: train {len(ff_tr):,} / {len(train):,} "
          f"({100*len(ff_tr)/len(train):.1f}%), "
          f"valid {len(ff_va):,} / {len(valid):,} ({100*len(ff_va)/len(valid):.1f}%)")
    print(f"  matches with frames: train {ff_tr.match_id.nunique():,}, "
          f"valid {ff_va.match_id.nunique():,}")

    # visibility gate: geometry is only interpretable where the ball origin is
    # inside the visible polygon with margin. Otherwise "nearest opponent 30 m"
    # may just mean the camera did not see them.
    gate_tr = ff_tr[ff_tr["ff_visible_r5"].fillna(False)]
    gate_va = ff_va[ff_va["ff_visible_r5"].fillna(False)]
    print(f"  origin visible with 5 m margin: train {len(gate_tr):,} "
          f"({100*len(gate_tr)/max(len(ff_tr),1):.3f} of framed), "
          f"valid {len(gate_va):,} ({100*len(gate_va)/max(len(ff_va),1):.3f})")

    if len(gate_tr) < 20_000 or len(gate_va) < 5_000:
        print("\n  [mech] Tier 2 sample too small to fit; stopping threat B here.")
        return

    p_no, y_no, k_no = fit_and_score(gate_tr, gate_va, "M0x", with_ff=False)
    p_ff, y_ff, k_ff = fit_and_score(gate_tr, gate_va, "M0x", with_ff=True)
    p_pt, y_pt, k_pt = fit_and_score(gate_tr, gate_va, "M0x", with_ff=True,
                                     include_post=True)
    b_no, s_no = brier(y_no, p_no)
    b_ff, s_ff = brier(y_ff, p_ff)
    b_pt, s_pt = brier(y_pt, p_pt)
    print(f"\n  refit on the Tier 2 train subsample, evaluated out of sample:")
    print(f"    M0x                    {k_no:>4} cols   Brier {b_no:.5f}   "
          f"skill {s_no:.4f}")
    print(f"    M0x + FF origin-only   {k_ff:>4} cols   Brier {b_ff:.5f}   "
          f"skill {s_ff:.4f}")
    print(f"    M0x + FF incl. target  {k_pt:>4} cols   Brier {b_pt:.5f}   "
          f"skill {s_pt:.4f}   <- POST-TREATMENT, bound only")
    print(f"\n    origin-only geometry adds {100*(s_ff - s_no):+.3f} skill points.")
    print(f"    If that were ~0 the control had no power and 'the effect survives'")
    print(f"    would mean nothing. Target-derived features add a further")
    print(f"    {100*(s_pt - s_ff):+.3f}, but they are measured around the")
    print(f"    interception point on failed passes, so that gain is partly the")
    print(f"    outcome predicting itself (same defect as pass_length).")

    v_no = gate_va.assign(p=p_no, resid=y_no - p_no)
    v_ff = gate_va.assign(p=p_ff, resid=y_ff - p_ff)
    v_pt = gate_va.assign(p=p_pt, resid=y_pt - p_pt)

    # ---- where does the geometry actually bite? ---------------------------- #
    # An aggregate skill gain near zero alongside a large change in the estimand
    # is only coherent if the control moves predictions on a SUBSET. Brier skill
    # averages over 51k passes; the primary sample is 24k of them and the recently
    # pressed part is 4k. Report the restricted fit so the reader can see which it
    # is instead of taking the aggregate as proof of a toothless control.
    # must match split_samples exactly, including the pressure-OFF filter --
    # without it the shortest bin fills with passes that are pressed AT t, and the
    # composition table then "discovers" that pressed passes have a defender close
    # by, which is a tautology rather than a finding.
    m_off = ~gate_va["under_pressure"].fillna(False).to_numpy()
    m_prim = (m_off & gate_va["events_since_last_press"].notna().to_numpy()
              & gate_va[clock_col].notna().to_numpy())
    m_bench = m_off & gate_va["events_since_last_press"].isna().to_numpy()
    tt_all = gate_va[clock_col].astype("Float64").astype(float).to_numpy()
    m_short = m_prim & (tt_all < 2.0)
    print(f"\n  Brier restricted -- does the geometry bite where the estimand lives?")
    print(f"    {'subset':<28} {'n':>8} {'M0x':>10} {'+FF':>10} {'delta':>9}")
    for nm, m in (("all gated passes", np.ones(len(gate_va), bool)),
                  ("benchmark (no prior press)", m_bench),
                  ("primary (prior press)", m_prim),
                  ("primary, < 2 s since press", m_short)):
        if m.sum() < 200:
            continue
        bn = float(np.mean((p_no[m] - y_no[m]) ** 2))
        bf = float(np.mean((p_ff[m] - y_ff[m]) ** 2))
        print(f"    {nm:<28} {int(m.sum()):>8,} {bn:>10.5f} {bf:>10.5f} "
              f"{bf - bn:>+9.5f}")
    print(f"    mean |change in predicted p|: benchmark "
          f"{np.abs(p_ff - p_no)[m_bench].mean():.5f}, "
          f"primary <2 s {np.abs(p_ff - p_no)[m_short].mean():.5f}")

    # ---- is the defence in fact still compact after the press ends? --------- #
    print(f"\n  OBSERVED DEFENSIVE GEOMETRY AT t, by time since the press ended")
    print(f"  (freeze frames; this is the mechanism, measured rather than assumed)")
    print(f"  {'elapsed (s)':<14} {'n':>7} {'nearest opp':>12} {'opp<3m':>8} "
          f"{'opp<5m':>8} {'opp seen':>9}")
    gb = gate_va[m_bench]
    rows = [("BENCHMARK", gb)] + [
        (str(nm), g) for nm, g in
        gate_va[m_prim].assign(bin=pd.cut(tt_all[m_prim], BIN_EDGES, right=False))
        .groupby("bin", observed=True)]
    for nm, g in rows:
        f = lambda c: g[c].astype("Float64").astype(float).mean()   # noqa: E731
        print(f"  {nm:<14} {len(g):>7,} {f('ff_nearest_opp_dist'):>12.2f} "
              f"{f('ff_opp_within_3'):>8.3f} {f('ff_opp_within_5'):>8.3f} "
              f"{f('ff_n_opp_visible'):>9.2f}")

    print("\n  " + "-" * 74)
    print("  SAME SAMPLE, THREE BASELINES")
    print("  " + "-" * 74)
    pr_no, bn_no = split_samples(v_no, clock_col)
    pr_ff, bn_ff = split_samples(v_ff, clock_col)
    pr_pt, bn_pt = split_samples(v_pt, clock_col)
    r_no = estimand(pr_no, bn_no, clock_col, "B0. Tier 2 sample, M0x baseline "
                                             "(no geometry)")
    r_ff = estimand(pr_ff, bn_ff, clock_col, "B1. Tier 2 sample, M0x + ORIGIN-ONLY "
                                             "freeze-frame geometry  [PRIMARY]")
    r_pt = estimand(pr_pt, bn_pt, clock_col, "B1b. same + target-derived geometry "
                                             "(post-treatment; bound only)")

    # composed: threat A exclusion AND threat B control
    v_ff2 = attach_pressure_windows(v_ff)
    pr_c, bn_c = split_samples(v_ff2, clock_col)
    r_c = estimand(pr_c[~pr_c["live_press"]], bn_c[~bn_c["live_press"]], clock_col,
                   "B2. origin-only geometry AND live-window passes removed")

    # ====================================================================== #
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print(f"  {'specification':<46} {'<2s adj':>9} {'2-8s adj':>10} {'slope t':>9}")
    for nm, res in [
        ("full validation, M0x", base),
        ("  + live-window passes removed", clean),
        ("Tier 2 subsample, M0x", r_no),
        ("  + origin-only frame geometry at t  [PRIMARY]", r_ff),
        ("  + geometry AND live windows removed", r_c),
        ("  + target-derived geometry (post-treatment)", r_pt),
    ]:
        print(f"  {nm:<46} {res['pooled_adj']:>+9.3f} {res['band_adj']:>+10.3f} "
              f"{res['slope_t']:>+9.2f}")

    ret_short = (100 * r_ff["pooled_adj"] / r_no["pooled_adj"]
                 if r_no["pooled_adj"] else float("nan"))
    ret_band = (100 * r_ff["band_adj"] / r_no["band_adj"]
                if r_no["band_adj"] else float("nan"))
    print(f"\n  geometry controls retain {ret_short:.1f}% of the <2 s effect and "
          f"{ret_band:.1f}% of the 2-8 s effect")
    print(f"  relevance floor {FLOOR_PP} pp:  2-8 s band after both controls "
          f"{r_c['band_adj']:+.3f} pp -> "
          f"{'ABOVE floor' if abs(r_c['band_adj']) >= FLOOR_PP else 'BELOW floor'}")


if __name__ == "__main__":
    main()
