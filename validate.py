#!/usr/bin/env python3
"""
validate.py -- physical-plausibility checks on the built tables.

Policy
------
Every derived geometric quantity is checked against a KNOWN PHYSICAL SCALE before
it is used. Not against itself, not against a schema, and not against a link-
integrity statistic.

This policy exists because of a specific near-miss. `presser_dist` -- the primary
Stage 1 pressure control -- was silently wrong for the entire corpus, because
StatsBomb logs each event in the acting team's own attacking frame and a Pressure
event is performed by the defending team. Carrier-to-presser distance had a median
of 68.2 m. No schema check, no null check, and no model diagnostic would have
caught it. What caught it was asking "how far apart are a presser and the player
being pressed, in metres, and is that a number football permits?"

An earlier check that the Pass<->Pressure `related_events` link is symmetric on
95% of pressed passes gave false comfort: link integrity says nothing about frame
consistency. Two things being correctly *associated* does not make their
coordinates *comparable*.

    python validate.py            # all checks
    python validate.py --strict   # warnings also fail the run

Exit status is non-zero if any check FAILs.
"""

from __future__ import annotations

import argparse
import math
import sys

import numpy as np
import pandas as pd

from src.load import load_passes, load_spine, assert_null_policy

PITCH_X, PITCH_Y = 120.0, 80.0
OFF_PITCH_TOL = 1.5          # StatsBomb allows locations slightly off the field

results: list[tuple[str, str, str]] = []


def record(status: str, name: str, detail: str) -> None:
    results.append((status, name, detail))
    mark = {"PASS": "  ok  ", "FAIL": " FAIL ", "WARN": " warn "}[status]
    print(f"[{mark}] {name}\n           {detail}")


def check(name: str, ok: bool, detail: str, warn_only: bool = False) -> None:
    record("PASS" if ok else ("WARN" if warn_only else "FAIL"), name, detail)


def f(df: pd.DataFrame, c: str) -> np.ndarray:
    return df[c].astype("Float64").astype(float).to_numpy()


# --------------------------------------------------------------------------- #
def check_bounds(df: pd.DataFrame) -> None:
    for c, lim in (("x", PITCH_X), ("y", PITCH_Y), ("end_x", PITCH_X),
                   ("end_y", PITCH_Y), ("presser_x", PITCH_X), ("presser_y", PITCH_Y)):
        v = f(df, c)
        v = v[~np.isnan(v)]
        lo, hi = v.min(), v.max()
        ok = lo >= -OFF_PITCH_TOL and hi <= lim + OFF_PITCH_TOL
        check(f"{c} within pitch bounds",
              ok, f"range [{lo:.2f}, {hi:.2f}] against 0-{lim:.0f} "
                  f"(tolerance {OFF_PITCH_TOL} m off-pitch)")


def check_pass_geometry(df: pd.DataFrame) -> None:
    comp = np.hypot(f(df, "end_x") - f(df, "x"), f(df, "end_y") - f(df, "y"))
    err = np.abs(f(df, "pass_length") - comp)
    m = np.nanmax(err)
    check("pass_length equals Euclidean(start, end)", m < 0.05,
          f"max |error| {m:.5f} m over {np.sum(~np.isnan(err)):,} passes "
          f"-- start and end are in one frame and one unit")

    ang = np.arctan2(f(df, "end_y") - f(df, "y"), f(df, "end_x") - f(df, "x"))
    da = np.abs(((f(df, "pass_angle") - ang + math.pi) % (2 * math.pi)) - math.pi)
    m = np.nanmax(da)
    check("pass_angle equals atan2(dy, dx)", m < 0.01,
          f"max |error| {m:.6f} rad")


def check_goal_anchors(df: pd.DataFrame) -> None:
    """Restart types occur at known points, so dist_to_goal has known values."""
    anchors = {
        "Goal Kick": (114.0, "own 6-yard line, 114 m from the attacking goal"),
        "Corner": (40.0, "corner flag, 40 m from the goal centre"),
        "Kick Off": (60.0, "centre spot, 60 m from the attacking goal"),
    }
    for ptype, (expected, why) in anchors.items():
        s = df[df["pass_type"] == ptype]
        if not len(s):
            continue
        med = float(np.nanmedian(f(s, "dist_to_goal")))
        check(f"dist_to_goal anchor: {ptype}", abs(med - expected) <= 1.5,
              f"median {med:.2f} m, expected ~{expected:.0f} ({why}); n={len(s):,}")


def check_presser_frame(df: pd.DataFrame) -> None:
    """The core check. A presser is a human being standing near the carrier."""
    d = f(df, "presser_dist")
    d = d[~np.isnan(d)]
    med = float(np.median(d))
    check("carrier-to-presser distance is physically possible",
          2.0 <= med <= 6.0,
          f"median {med:.2f} m over {len(d):,} pressed events. A press is a "
          f"defender within a few metres; an unmirrored frame gives ~68 m.")

    far = float(np.mean(d > 40))
    check("no unmirrored mass in carrier-to-presser distance", far < 0.01,
          f"{100*far:.3f}% exceed 40 m (a frame error puts ~50% near 68 m)")

    # period invariance: if the halftime end switch were NOT normalised upstream,
    # one period would be mirrored correctly and the other would not.
    med_by_p = {}
    for p in sorted(df["period"].dropna().unique()):
        v = f(df[df["period"] == p], "presser_dist")
        v = v[~np.isnan(v)]
        if len(v) > 1000:
            med_by_p[int(p)] = float(np.median(v))
    spread = max(med_by_p.values()) - min(med_by_p.values())
    check("presser distance is period-invariant", spread < 0.5,
          f"medians by period {med_by_p}, spread {spread:.3f} m. A per-period "
          f"frame would split these into a mirrored and an unmirrored mode.")


def check_lag_frame(df: pd.DataFrame) -> None:
    v = f(df, "lag1_presser_dist_to_t")
    v = v[~np.isnan(v)]
    hump = float(np.mean((v >= 60) & (v <= 80)))
    check("lag-1 presser distance shows no unmirrored hump", hump < 0.05,
          f"median {np.median(v):.2f} m; {100*hump:.2f}% in the 60-80 m band "
          f"where a frame error would pile up; n={len(v):,}")


def check_temporal(df: pd.DataFrame) -> None:
    for c, cap in (("pressure_lead_s", 60.0), ("time_since_receipt_s", 600.0),
                   ("time_since_last_press_s", 600.0)):
        v = f(df, c)
        v = v[~np.isnan(v)]
        check(f"{c} is non-negative", float(v.min()) >= 0.0,
              f"min {v.min():.3f} s, median {np.median(v):.2f}, "
              f"p99 {np.percentile(v, 99):.2f}, max {v.max():.2f}")


def check_360(df: pd.DataFrame) -> None:
    ff = df[df["ff_available"].fillna(False)]
    if not len(ff):
        return
    bad = int((f(ff, "ff_opp_within_3") > f(ff, "ff_opp_within_5")).sum())
    check("360 nested radii are consistent", bad == 0,
          f"{bad} rows with opp_within_3 > opp_within_5")

    n_over = int((f(ff, "ff_n_opp_visible") > 11).sum())
    check("360 never shows more than 11 opponents", n_over == 0,
          f"{n_over} frames of {len(ff):,} report >11 opponents "
          f"(upstream glitch; physically impossible)", warn_only=True)

    # cross-source: the nearest visible opponent should be no farther than the
    # annotated presser, since the presser is one of the opponents on the pitch.
    both = ff[ff["presser_dist"].notna() & ff["ff_nearest_opp_dist"].notna()]
    if len(both):
        frac = float(np.mean(f(both, "ff_nearest_opp_dist")
                             <= f(both, "presser_dist") + 1.0))
        check("360 nearest opponent is no farther than the annotated presser",
              frac > 0.90,
              f"holds for {100*frac:.1f}% of {len(both):,} rows with both "
              f"measures -- two independent sources agree on the local geometry")


def check_spine(spine: pd.DataFrame) -> None:
    defn = spine[spine["press_definitional"].fillna(False)]
    if len(defn):
        rate = float(defn["under_pressure"].mean())
        check("definitional-pressure types are indeed always flagged", rate > 0.999,
              f"under_pressure rate {rate:.4f} over {len(defn):,} rows "
              f"(Dribble/Duel/Dispossessed/Clearance) -- confirms these carry no "
              f"information and must stay off the exposure clock")

    clk = spine[spine["spine_role"] == "clock"]
    rate = float(clk["under_pressure"].mean())
    check("clock-role types carry genuine variation", 0.05 < rate < 0.60,
          f"under_pressure rate {rate:.4f} over {len(clk):,} Pass/Carry rows")

    pressed = clk[clk["under_pressure"].fillna(False)]
    z = pressed["events_since_last_press"].fillna(-1)
    check("events_since_last_press is 0 exactly when pressed",
          bool((z == 0).all()),
          f"{int((z != 0).sum())} pressed clock rows with a non-zero clock")


def check_nulls(df: pd.DataFrame) -> None:
    try:
        assert_null_policy(df)
        check("nullable dtypes survived the read", True,
              "null-critical columns kept extension dtypes, so NULL cannot be "
              "silently read as 0")
    except TypeError as exc:
        check("nullable dtypes survived the read", False, str(exc))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true", help="warnings fail the run")
    args = ap.parse_args()

    print("=" * 78)
    print("PHYSICAL-PLAUSIBILITY VALIDATION")
    print("=" * 78)
    df = load_passes()
    spine = load_spine(columns=["spine_role", "press_definitional", "under_pressure",
                                "events_since_last_press", "event_type"])
    print(f"passes {len(df):,}   spine {len(spine):,}\n")

    check_nulls(df)
    check_bounds(df)
    check_pass_geometry(df)
    check_goal_anchors(df)
    check_presser_frame(df)
    check_lag_frame(df)
    check_temporal(df)
    check_360(df)
    check_spine(spine)

    n_fail = sum(1 for s, _, _ in results if s == "FAIL")
    n_warn = sum(1 for s, _, _ in results if s == "WARN")
    print("\n" + "=" * 78)
    print(f"{len(results)} checks: {len(results) - n_fail - n_warn} passed, "
          f"{n_warn} warnings, {n_fail} failures")
    print("=" * 78)
    if n_fail or (args.strict and n_warn):
        sys.exit(1)


if __name__ == "__main__":
    main()
