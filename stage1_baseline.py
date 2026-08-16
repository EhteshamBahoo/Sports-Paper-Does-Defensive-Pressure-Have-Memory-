#!/usr/bin/env python3
"""
stage1_baseline.py -- Stage 1: pass success as a function of LOCAL pressure only.

This is the baseline whose residual Stage 2 interrogates. It must be honest before
anything is built on it, so this script reports calibration and nothing else.
No residual analysis here.

Split discipline
----------------
Three-way split by MATCH, not by pass: passes within a match share teams,
conditions and annotator, so a pass-level split leaks. Deterministic SHA-256 hash
of match_id, so it is identical on every run and for every referee.

    train 50%   fit
    valid 20%   all model development and specification choices
    test  30%   touched once, for the numbers reported at the bottom

Disclosure: an earlier smoke run of two specifications was inspected on a 300k
random subsample before the validation split existed. The redesign below was
driven by TRAIN-ONLY diagnostics (indicator collinearity, the length functional
form, and the truncation finding). Test numbers are reported once.

pass_length and pass_angle are POST-TREATMENT
---------------------------------------------
StatsBomb sets `end_location` to where the ball actually ended. For an intercepted
pass that is the interception point, so failure mechanically shortens the recorded
length. Measured on train: median length 16.40 m for complete against 19.31 m for
incomplete, but p10 8.07 m against 3.61 m -- a heavy short tail that exists only
among failures. 61% of passes recorded at <= 5 m are interceptions, and they
complete at 0.387 against 0.825 overall.

So a baseline containing pass_length is partly predicting the outcome from the
outcome. It will look better calibrated and score a better Brier than it honestly
deserves, and its residual -- the object Stage 2 studies -- is contaminated.

Three specifications are therefore fitted:

    M0  pre-treatment only. Nothing derived from end_location. This is the
        defensible residual base for Stage 2.
    M1  spec-exact, as originally specified: pressure + zone + length/angle.
    M2  M0 plus length/angle. The conventional pass-completion model.

The M1/M2 minus M0 gap is an upper bound on how much apparent accuracy comes from
outcome leakage, not from football.

    python stage1_baseline.py
    python stage1_baseline.py --sample 400000     # fast smoke run
"""

from __future__ import annotations

import argparse
import hashlib

import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.load import load_passes, analysis_sample

HASH_BUCKETS = 1000
TRAIN_HI, VALID_HI = 500, 700          # <500 train, 500-699 valid, >=700 test

FEATURE_COLS = [
    "match_id", "comp_season_uid", "pass_success", "under_pressure", "counterpress",
    "presser_dist", "pressure_lead_s", "n_pressure_linked", "zone", "dist_to_goal",
    "x", "y", "pass_length", "pass_angle", "pass_height", "pass_body_part",
    "play_pattern", "score_diff", "is_possession_team", "is_set_piece_restart",
]

LEN_EDGES = [0, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40, 50, 60, 200]


CHUNK = 200_000


def fit_logit(X: np.ndarray, y: np.ndarray, max_iter: int = 40,
              tol: float = 1e-9) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by IRLS, accumulating X'WX in row chunks.

    statsmodels' GLM materialises weighted copies of the full design on every
    iteration, which OOMs at 186 columns x 1.6M rows on a 26 GB machine (observed:
    SIGKILL). X'WX is only k x k, so it can be accumulated over chunks and the
    design kept in float32. Verified against statsmodels: see --verify-fitter.
    """
    n, k = X.shape
    beta = np.zeros(k)
    XtWX = np.eye(k)
    for _ in range(max_iter):
        XtWX = np.zeros((k, k))
        XtWz = np.zeros(k)
        for s in range(0, n, CHUNK):
            Xc = np.asarray(X[s:s + CHUNK], dtype=np.float64)
            eta = Xc @ beta
            mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
            w = np.clip(mu * (1.0 - mu), 1e-9, None)
            z = eta + (y[s:s + CHUNK] - mu) / w
            XtWX += Xc.T @ (Xc * w[:, None])
            XtWz += Xc.T @ (w * z)
        ridge = 1e-8 * np.eye(k)
        new = np.linalg.solve(XtWX + ridge, XtWz)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    cov = np.linalg.inv(XtWX + 1e-8 * np.eye(k))
    return beta, np.sqrt(np.diag(cov))


def predict_logit(X: np.ndarray, beta: np.ndarray) -> np.ndarray:
    out = np.empty(len(X))
    for s in range(0, len(X), CHUNK):
        Xc = np.asarray(X[s:s + CHUNK], dtype=np.float64)
        out[s:s + CHUNK] = 1.0 / (1.0 + np.exp(-np.clip(Xc @ beta, -30, 30)))
    return out


def match_bucket(match_id: int) -> int:
    return int(hashlib.sha256(str(int(match_id)).encode()).hexdigest(), 16) % HASH_BUCKETS


# Which post-treatment inputs each specification is ALLOWED to consume. M1/M2/M3
# include pass_length and pass_angle deliberately -- they exist to quantify how
# much conventional model skill is outcome leakage, so their gap to M0 is the
# estimate. The clean specs must consume none.
#
# This is a declaration, and declarations rot. validate.py enforces it against
# build_design's source: it finds every d["col"] access under each spec branch
# and fails if one touches a registered post-treatment column not allowed here.
# A runtime assert cannot do this job -- build_design renames pass_length into
# L_(0,4] bins and pass_angle into ang_sin/ang_cos, so checking X.columns would
# pass vacuously on exactly the specifications that leak.
SPEC_POST_TREATMENT = {
    "M0": (), "M0i": (), "M0x": (),
    "M1": ("pass_length", "pass_angle"),
    "M2": ("pass_length", "pass_angle"),
    "M3": ("pass_length", "pass_angle"),
}
CLEAN_SPECS = tuple(s for s, v in SPEC_POST_TREATMENT.items() if not v)


def build_design(d: pd.DataFrame, spec: str) -> tuple[pd.DataFrame, np.ndarray]:
    """spec in {M0, M0i, M0x, M1, M2, M3}."""
    X = pd.DataFrame(index=d.index)

    # ---- carrier pressure -------------------------------------------------
    # under_pressure and "a presser was linked" agree on 98.1% of passes, and
    # n_pressure_linked correlates 0.92 with the flag. Using all three splits one
    # effect across three collinear columns and flips their signs. Collapsed to a
    # single indicator plus a continuous intensity and a multi-presser flag.
    X["under_pressure"] = d["under_pressure"].fillna(False).astype(float)
    dist = d["presser_dist"].astype("Float64").astype(float)
    # 1/(1+d) -> 0 as the presser recedes, so "no presser linked" maps to 0 on the
    # same scale as "presser infinitely far". No NULL is replaced by a fake value.
    X["inv_presser_dist"] = np.where(np.isnan(dist), 0.0, 1.0 / (1.0 + dist))
    lead = d["pressure_lead_s"].astype("Float64").astype(float)
    X["pressure_lead_s"] = np.where(np.isnan(lead), 0.0, lead)
    X["multi_presser"] = (d["n_pressure_linked"].astype("Float64").fillna(0) >= 2).astype(float)
    X["counterpress"] = d["counterpress"].fillna(False).astype(float)

    # ---- origin geometry (pre-treatment) ----------------------------------
    X["dist_to_goal"] = (d["dist_to_goal"].astype("Float64").astype(float) - 60.0) / 30.0
    X["origin_x"] = (d["x"].astype("Float64").astype(float) - 60.0) / 30.0
    X["origin_y"] = (d["y"].astype("Float64").astype(float) - 40.0) / 20.0
    X["score_diff"] = d["score_diff"].astype("Float64").astype(float).clip(-3, 3)

    cats = [pd.get_dummies(d["zone"].astype(str), prefix="z", drop_first=True),
            pd.get_dummies(d["comp_season_uid"].astype(str), prefix="cs", drop_first=True)]

    if spec in ("M0", "M0i", "M0x", "M2", "M3"):
        cats.append(pd.get_dummies(d["pass_height"].astype(str), prefix="h", drop_first=True))
        cats.append(pd.get_dummies(d["pass_body_part"].astype(str), prefix="b", drop_first=True))
        cats.append(pd.get_dummies(d["play_pattern"].astype(str), prefix="pp", drop_first=True))

    if spec in ("M0i", "M0x", "M3"):
        # Additive height + length underpredicts the hard tail: a long HIGH pass is
        # not as hopeless as the sum of its parts implies. Interacting them lets the
        # tail flatten instead of compounding.
        h = d["pass_height"].astype(str)
        cats.append(pd.get_dummies(h.str.cat(d["zone"].astype(str), sep="|"),
                                   prefix="hZ", drop_first=True))
        cats.append(pd.get_dummies(h.str.cat(d["play_pattern"].astype(str), sep="|"),
                                   prefix="hP", drop_first=True))
        if spec == "M3":
            Lb = pd.cut(d["pass_length"].astype("Float64").astype(float), LEN_EDGES)
            cats.append(pd.get_dummies(h.str.cat(Lb.astype(str), sep="|"),
                                       prefix="hL", drop_first=True))

    if spec in ("M1", "M2", "M3"):
        # Flexible bins: observed success is sharply non-monotone in length
        # (0.386 at <=5 m, 0.910 at 10-15 m, 0.360 above 60 m). A cubic cannot
        # track that, and the misfit lands in the low-probability tail.
        L = d["pass_length"].astype("Float64").astype(float)
        cats.append(pd.get_dummies(pd.cut(L, LEN_EDGES), prefix="L", drop_first=True))
        ang = d["pass_angle"].astype("Float64").astype(float)
        X["ang_sin"], X["ang_cos"] = np.sin(ang), np.cos(ang)

    X = pd.concat([X] + cats, axis=1).astype(np.float32)

    if spec == "M0x":
        # M0i with pressure interacted with the geometry block.
        #
        # M0i is miscalibrated WITHIN the pressed subsample: by quintile of
        # predicted probability the error runs +0.0609 at the bottom to -0.0309
        # at the top. That is a slope error, not a level error -- pressed
        # predictions are over-dispersed. A pooled additive logit forces the same
        # geometry coefficients on pressed and unpressed passes, so if geometry
        # matters less once a defender is on you, the pooled model over-applies it
        # to pressed passes and spreads their predictions too wide.
        #
        # Interacting pressure with the geometry block lets the pressed subsample
        # carry its own slope. Interactions only -- no new features.
        up = X["under_pressure"].to_numpy()
        ipd = X["inv_presser_dist"].to_numpy()
        geom_prefixes = ("z_", "h_", "pp_", "b_")
        geom = [c for c in X.columns if c.startswith(geom_prefixes)]
        geom += ["dist_to_goal", "origin_x", "origin_y"]
        blocks = [X]
        blocks.append(pd.DataFrame(
            {f"up_x_{c}": X[c].to_numpy() * up for c in geom},
            index=X.index, dtype=np.float32))
        # pressure INTENSITY also gets a slope on the strongest geometry terms
        intens = [c for c in X.columns if c.startswith("h_")] + ["dist_to_goal"]
        blocks.append(pd.DataFrame(
            {f"ipd_x_{c}": X[c].to_numpy() * ipd for c in intens},
            index=X.index, dtype=np.float32))
        X = pd.concat(blocks, axis=1)

    X.insert(0, "const", 1.0)
    y = d["pass_success"].astype("boolean").astype(float).to_numpy()
    return X, y


def strata_quintiles(y: np.ndarray, p: np.ndarray, pressed: np.ndarray,
                     label: str) -> dict:
    """Calibration by quintile of predicted probability, split by pressure state.

    Aggregate calibration on a stratum can look perfect while the errors merely
    cancel across the predicted-probability range. This is the table that shows it.
    """
    out = {}
    print(f"\n  {label} -- calibration by quintile of predicted probability")
    for name, m in (("PRESSED", pressed), ("UNPRESSED", ~pressed)):
        ys, ps = y[m], p[m]
        edges = np.quantile(ps, np.linspace(0, 1, 6))
        edges[0], edges[-1] = -np.inf, np.inf
        q = np.digitize(ps, edges[1:-1])
        print(f"\n    {name}  n={len(ys):,}   aggregate: pred {ps.mean():.4f}"
              f"  obs {ys.mean():.4f}  diff {ys.mean()-ps.mean():+.4f}")
        print(f"      {'quintile':<10} {'n':>9} {'predicted':>11} {'observed':>10}"
              f" {'diff':>9}")
        diffs = []
        for k in range(5):
            s = q == k
            if s.sum() < 50:
                continue
            dv = ys[s].mean() - ps[s].mean()
            diffs.append(dv)
            print(f"      {k+1:<10} {int(s.sum()):>9,} {ps[s].mean():>11.4f}"
                  f" {ys[s].mean():>10.4f} {dv:>+9.4f}")
        signs = [1 if v > 0 else -1 for v in diffs]
        monotone = all(a >= b for a, b in zip(diffs, diffs[1:]))
        flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
        print(f"      max |diff| {max(abs(v) for v in diffs):.4f}   "
              f"sign changes {flips}   monotone decreasing: {monotone}")
        out[name] = {"max_abs": max(abs(v) for v in diffs), "monotone": monotone,
                     "flips": flips, "diffs": diffs}
    return out


def calibration(y: np.ndarray, p: np.ndarray, pressed: np.ndarray, label: str) -> dict:
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y.mean() - y) ** 2))
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=int)
    ranks[order] = np.arange(len(p))
    dec = (ranks * 10 // len(p)).clip(0, 9)

    print(f"\n  {label}:  n={len(y):,}   base rate {y.mean():.4f}   "
          f"mean pred {p.mean():.4f}")
    print(f"  Brier {brier:.5f}   (base-rate-only reference {ref:.5f}, "
          f"skill {1 - brier/ref:.3f})")
    print(f"\n  {'decile':>7} {'n':>9} {'mean pred':>11} {'observed':>10} {'diff':>9}"
          f" {'pred range':>18}")
    worst = 0.0
    for k in range(10):
        m = dec == k
        mp, mo = p[m].mean(), y[m].mean()
        worst = max(worst, abs(mp - mo))
        print(f"  {k+1:>7} {int(m.sum()):>9,} {mp:>11.4f} {mo:>10.4f} {mp-mo:>+9.4f}"
              f"   [{p[m].min():.3f}, {p[m].max():.3f}]")
    print(f"  max |decile miscalibration| {worst:.4f}")

    print("\n  low-probability tail (fixed-width bins):")
    for lo, hi in [(0.0, .2), (.2, .4), (.4, .6), (.6, .8)]:
        m = (p >= lo) & (p < hi)
        if m.sum() < 200:
            continue
        print(f"    [{lo:.1f},{hi:.1f})  n={int(m.sum()):>8,}"
              f"   pred {p[m].mean():.4f}   obs {y[m].mean():.4f}"
              f"   diff {p[m].mean()-y[m].mean():+.4f}")
    for name, m in (("pressed", pressed), ("unpressed", ~pressed)):
        print(f"    {name:<10} n={int(m.sum()):>8,}   pred {p[m].mean():.4f}"
              f"   obs {y[m].mean():.4f}   diff {p[m].mean()-y[m].mean():+.4f}")
    return {"brier": brier, "skill": 1 - brier / ref, "worst_decile": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--specs", default=None,
                    help="comma-separated subset, e.g. M0i,M0x")
    ap.add_argument("--verify-fitter", action="store_true",
                    help="check the chunked IRLS against statsmodels")
    ap.add_argument("--final", action="store_true",
                    help="also evaluate on the held-out TEST set")
    args = ap.parse_args()

    print("=" * 78)
    print("STAGE 1 BASELINE -- pass success on local pressure only")
    print("=" * 78)

    d = analysis_sample(load_passes(columns=FEATURE_COLS))
    d = d[d["pass_success"].notna()]
    if args.sample:
        d = d.sample(n=min(args.sample, len(d)), random_state=0)

    b = d["match_id"].map(match_bucket)
    train, valid, test = d[b < TRAIN_HI], d[(b >= TRAIN_HI) & (b < VALID_HI)], d[b >= VALID_HI]
    print(f"analysis sample {len(d):,} passes / {d.match_id.nunique():,} matches")
    print(f"  train {len(train):>10,} passes  {train.match_id.nunique():>5,} matches")
    print(f"  valid {len(valid):>10,} passes  {valid.match_id.nunique():>5,} matches")
    print(f"  test  {len(test):>10,} passes  {test.match_id.nunique():>5,} matches")
    assert not (set(train.match_id) & set(test.match_id))
    assert not (set(valid.match_id) & set(test.match_id))
    print("  no match appears in more than one split")

    summary = {}
    ALL_SPECS = (("M0", "pre-treatment only (no end_location features)"),
                       ("M0i", "M0 + height x zone and height x play-pattern"),
                       ("M1", "spec-exact: pressure + zone + length/angle"),
                       ("M2", "M0 + length/angle (conventional model)"),
                       ("M3", "M2 + height x length and height x zone interactions"),
                  ("M0x", "M0i + pressure x geometry interactions"))
    wanted = args.specs.split(",") if args.specs else [k for k, _ in ALL_SPECS]
    for spec, desc in [t for t in ALL_SPECS if t[0] in wanted]:
        Xtr, ytr = build_design(train, spec)
        print(f"\n{'=' * 78}\n{spec}  {desc}\n  design {Xtr.shape[1]} columns, "
              f"{len(Xtr):,} rows")
        beta, bse = fit_logit(Xtr.to_numpy(), ytr)

        splits = [("VALIDATION", valid)]
        if args.final:
            splits.append(("TEST", test))
        for name, part in splits:
            Xp, yp = build_design(part, spec)
            Xp = Xp.reindex(columns=Xtr.columns, fill_value=0.0)
            p = predict_logit(Xp.to_numpy(), beta)
            pressed = part["under_pressure"].fillna(False).to_numpy()
            m = calibration(yp, p, pressed, f"{spec} on {name}")
            summary[(spec, name)] = m
            if name == "TEST":
                m["strata"] = strata_quintiles(yp, p, pressed, f"{spec} on TEST")

        print("\n  pressure coefficients (log-odds; iid SEs, too small "
              "because passes cluster in matches):")
        for nm in ["under_pressure", "inv_presser_dist", "pressure_lead_s",
                   "multi_presser", "counterpress"]:
            i = list(Xtr.columns).index(nm)
            print(f"    {nm:<20} {beta[i]:>+9.4f}  se {bse[i]:.4f}")

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"  {'model':<8} {'split':<12} {'Brier':>9} {'skill':>8} {'worst decile':>14}")
    for (spec, split), m in summary.items():
        print(f"  {spec:<8} {split:<12} {m['brier']:>9.5f} {m['skill']:>8.3f}"
              f" {m['worst_decile']:>14.4f}")


if __name__ == "__main__":
    main()
