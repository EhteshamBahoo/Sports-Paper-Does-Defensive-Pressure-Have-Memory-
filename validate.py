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
import ast
import math
import sys

import numpy as np
import pandas as pd

from src.load import (load_passes, load_spine, assert_null_policy,
                      POST_TREATMENT, TAINT_EXEMPT)

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


def check_post_treatment() -> None:
    """Re-derive, from build.py's source, which columns depend on end_location.

    A hand-maintained list of outcome-derived features rots the moment someone
    adds a column. This walks build.py's AST instead: seed the taint at
    `end_location`, `end_x`, `end_y` and the `target` vector built from them, let
    it propagate through local assignments, and report every output column whose
    value depends on a tainted name. The register in src/load.py must match.

    The check exists because post-treatment contamination has entered this project
    twice -- pass_length in Stage 1, and four ff_* columns in the first draft of
    the Stage 2 geometry control -- and it is genuinely hard to see by reading.
    """
    src = (__file__.rsplit("/", 1)[0] + "/build.py")
    tree = ast.parse(open(src).read())

    # Seeded with the raw endpoint names AND every already-registered column.
    # The second part looks circular but is not: taint only ever propagates
    # outward, so seeding with known-contaminated columns can reveal additional
    # unregistered ones and can never hide one. It is what catches derivation
    # ACROSS functions -- escape_sep_gain is built in _classify_escape from
    # presser_dist_to_end, which was contaminated back in parse_match, and a
    # strictly per-function analysis reports it clean.
    SEEDS = ({"end_location", "end_x", "end_y", "target"} | set(POST_TREATMENT))

    def names_in(node) -> set[str]:
        out = set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                out.add(n.id)
            elif isinstance(n, ast.Constant) and isinstance(n.value, str):
                out.add(n.value)
            elif isinstance(n, ast.Attribute):
                out.add(n.attr)
        return out

    derived: dict[str, str] = {}
    for fn in [n for n in ast.walk(tree)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        tainted = set(SEEDS)
        # two passes so an assignment can be tainted by a definition seen later
        for _ in range(2):
            for node in ast.walk(fn):
                # output columns are written two ways in build.py: r["col"] = expr,
                # and as keys of a dict literal. Missing the second form is how an
                # early version of this check reported prog_dist as clean.
                if isinstance(node, ast.Dict):
                    for k, val in zip(node.keys, node.values):
                        if (isinstance(k, ast.Constant)
                                and isinstance(k.value, str)
                                and (names_in(val) & tainted)):
                            derived.setdefault(
                                k.value, sorted(names_in(val) & tainted)[0])
                if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                    continue
                rhs = node.value
                if rhs is None:
                    continue
                hit = names_in(rhs) & tainted
                if not hit:
                    continue
                targets = ([node.target] if isinstance(node, ast.AnnAssign)
                           else node.targets)
                for tgt in targets:
                    if (isinstance(tgt, ast.Subscript)
                            and isinstance(tgt.slice, ast.Constant)
                            and isinstance(tgt.slice.value, str)):
                        derived.setdefault(tgt.slice.value, sorted(hit)[0])
                    else:
                        tainted |= names_in(tgt)

    known = set(POST_TREATMENT)
    found = set(derived)
    unregistered = sorted(found - known - set(TAINT_EXEMPT))
    check("post-treatment register covers every end_location-derived column",
          not unregistered,
          f"{len(found)} columns carry end_location through their construction in "
          f"build.py; "
          + (f"UNREGISTERED: {unregistered} -- add to src.load.POST_TREATMENT "
             f"with a reason, or to TAINT_EXEMPT with an argument for why the "
             f"value does not actually depend on the endpoint"
             if unregistered
             else f"{len(known)} registered, {len(TAINT_EXEMPT)} exempt with "
                  f"stated reasons"))

    # Only build-provenance entries should appear in the scan. pass_length and
    # pass_angle arrive already derived from the provider, so no reading of
    # build.py can reveal them -- the register is the only record.
    local = {c for c, (prov, _) in POST_TREATMENT.items() if prov == "build"}
    stale = sorted(local - found)
    check("post-treatment register has no stale entries", not stale,
          f"registered as build-derived but not found in build.py: {stale}"
          if stale else
          f"all {len(local)} build-derived entries confirmed in source; "
          f"{len(known) - len(local)} arrive post-treatment from StatsBomb "
          f"({sorted(known - local)}) and cannot be detected by reading build.py",
          warn_only=True)


def check_spec_purity() -> None:
    """Enforce stage1_baseline.SPEC_POST_TREATMENT against build_design's source.

    Walks the `if spec in (...)` branches inside build_design and collects the
    d["col"] accesses guarded by each. A registered post-treatment column reached
    under a spec that does not declare it is a leak.
    """
    from stage1_baseline import SPEC_POST_TREATMENT, CLEAN_SPECS

    src = (__file__.rsplit("/", 1)[0] + "/stage1_baseline.py")
    tree = ast.parse(open(src).read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "build_design")

    def cols_read(node) -> set[str]:
        out = set()
        for n in ast.walk(node):
            if (isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
                    and n.value.id == "d" and isinstance(n.slice, ast.Constant)
                    and isinstance(n.slice.value, str)):
                out.add(n.slice.value)
        return out

    all_specs = set(SPEC_POST_TREATMENT)
    reach: dict[str, set[str]] = {s: set() for s in all_specs}

    def visit(body, specs: set[str]) -> None:
        """Attribute each d["col"] read to the specs that can actually reach it.

        Must recurse rather than flat-walk: `if spec == "M3"` is nested inside
        `if spec in ("M0i", "M0x", "M3")`, so a flat walk credits pass_length to
        M0i and M0x and reports a leak in the two clean specifications the whole
        Stage 2 residual is built on.
        """
        for st in body:
            if isinstance(st, ast.If):
                named = {c.value for c in ast.walk(st.test)
                         if isinstance(c, ast.Constant)
                         and isinstance(c.value, str)} & all_specs
                visit(st.body, (specs & named) if named else specs)
                visit(st.orelse, specs)
            else:
                cols = cols_read(st)
                if cols:
                    for s in specs:
                        reach[s] |= cols

    visit(fn.body, all_specs)

    violations = {
        s: sorted((reach[s] & set(POST_TREATMENT)) - set(SPEC_POST_TREATMENT[s]))
        for s in SPEC_POST_TREATMENT
    }
    bad = {s: v for s, v in violations.items() if v}
    check("clean specifications read no post-treatment column", not bad,
          f"specs {list(CLEAN_SPECS)} declare no post-treatment inputs; "
          + (f"VIOLATIONS {bad}" if bad else
             "source confirms it. Leakage specs declare "
             f"{sorted(set(SPEC_POST_TREATMENT['M2']))} and are the leakage "
             f"bound, not a result."))


def check_post_treatment_signature(df: pd.DataFrame) -> None:
    """Show the contamination rather than asserting it.

    The register says these columns are outcome-derived by construction. This
    prints what that costs, so the number is in the validation output and not only
    in the README: a pre-treatment column may legitimately differ between complete
    and incomplete passes, but end_location-derived columns carry a short tail
    that exists only among failures, because interception truncates the ball path.
    """
    ok = df["pass_success"].notna()
    comp = df[ok & df["pass_success"].fillna(False)]
    fail = df[ok & ~df["pass_success"].fillna(True)]
    L_c, L_f = f(comp, "pass_length"), f(fail, "pass_length")
    p10_c, p10_f = np.nanpercentile(L_c, 10), np.nanpercentile(L_f, 10)
    short = df[ok & (f(df, "pass_length") <= 5.0)]
    rate = float(short["pass_success"].mean())
    base = float(df.loc[ok, "pass_success"].mean())
    check("post-treatment signature is present and quantified", True,
          f"pass_length p10 {p10_c:.2f} m complete against {p10_f:.2f} m "
          f"incomplete (median {np.nanmedian(L_c):.2f} against "
          f"{np.nanmedian(L_f):.2f}); passes recorded <=5 m complete at "
          f"{rate:.3f} against {base:.3f} overall. Interception truncates the "
          f"recorded path, so length is partly the outcome.")


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
    check_post_treatment()
    check_spec_purity()
    check_post_treatment_signature(df)
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
