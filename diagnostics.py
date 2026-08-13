#!/usr/bin/env python3
"""
diagnostics.py -- study-design diagnostics. Reads the built tables only.

Four questions, all answerable before any model is fitted:

  1. CHAIN COVERAGE. Stage 2 needs freeze frames at t, t-1 and sometimes t-2,
     each passing the visible-area gate. Per-event coverage is not the relevant
     number. Also reports whether censoring is related to the treatment.

  2. EXPOSURE CLOCK. How much pressure exposure a pass-level clock misses relative
     to the ball-event spine. This is measurement error in the treatment variable,
     not a gap in the hazard model.

  3. PRESS-RUN EXITS over the spine. Turnover, escape and stoppage select in
     opposite directions, so the mix determines the competing-risks specification.

  4. ESCAPE, POSITIVELY DEFINED. "Next event lacks the under_pressure flag" is an
     absence of annotation, not a football event. This reports how much of the
     structural escape category survives definitions with a positive signature.

  5. PRESSER STALENESS. The spatial-specificity measure for falsification test 3b
     uses a defender position recorded seconds earlier. This quantifies the
     resulting error against 360 ground truth and tests whether capping helps.

    python diagnostics.py
    python diagnostics.py --only exits
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.load import load_passes, load_spine, analysis_sample

TOURNAMENT_KEYS = ("World Cup", "UEFA Euro", "Women's Euro", "African Cup",
                   "Copa America")

EXIT_GROUPS = {
    "turnover": ["turnover"],
    "escape": ["escape", "shot"],
    "stoppage": ["stoppage_foul", "stoppage_out"],
    "other": ["clearance", "period_end", "other", "segment_break"],
}


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- #
def chain_coverage() -> None:
    rule("1. TIER 2 CHAIN COVERAGE  (frames needed at t, t-1, t-2)")
    df = load_passes(columns=[
        "match_id", "segment_uid", "event_index", "ff_available", "ff_visible_r3",
        "ff_visible_r5", "is_possession_team", "is_set_piece_restart",
        "pass_ord_in_seg", "under_pressure", "seg_press_count_spine", "zone",
        "competition_name", "season_name", "ff_opp_within_5",
    ])
    t2 = df.loc[df["ff_available"].fillna(False), "match_id"].unique()
    d = analysis_sample(df[df["match_id"].isin(t2)]).sort_values(
        ["segment_uid", "event_index"])
    print(f"Tier 2 matches with >=1 usable freeze frame: {len(t2)}")
    print(f"analysis-sample passes in those matches: {len(d):,}")

    seg = d["segment_uid"]
    have = d["ff_available"].fillna(False).to_numpy()
    gate3 = have & d["ff_visible_r3"].fillna(False).to_numpy()
    gate5 = have & d["ff_visible_r5"].fillna(False).to_numpy()
    ordv = d["pass_ord_in_seg"].astype("Float64").to_numpy()

    def lagged(col, k):
        return (pd.Series(col, index=d.index).groupby(seg, sort=False).shift(k)
                .fillna(False).to_numpy().astype(bool))

    print(f"\n{'gate':<20} {'P(t)':>8} {'P(t,t-1)':>10} {'P(t,t-1,t-2)':>14}"
          f" {'P(t)^2':>8} {'P(t)^3':>8}")
    for label, base in (("frame present", have), ("+ 3 m gate", gate3),
                        ("+ 5 m gate", gate5)):
        l1, l2 = lagged(base, 1), lagged(base, 2)
        h1, h2 = ordv >= 1, ordv >= 2
        p = base.mean()
        j1 = (base & l1)[h1].mean()
        j2 = (base & l1 & l2)[h2].mean()
        print(f"{label:<20} {p:>8.3f} {j1:>10.3f} {j2:>14.3f} {p**2:>8.3f} {p**3:>8.3f}")

    l1, l2 = lagged(gate5, 1), lagged(gate5, 2)
    print("\nusable passes under the 5 m gate:")
    print(f"   t only          {int(gate5.sum()):>9,}")
    print(f"   t and t-1       {int((gate5 & l1 & (ordv >= 1)).sum()):>9,}")
    print(f"   t, t-1 and t-2  {int((gate5 & l1 & l2 & (ordv >= 2)).sum()):>9,}")

    rule("1b. IS THE CENSORING RELATED TO THE TREATMENT?")
    d = d.assign(gate=gate5, have=have)
    print(f"{'stratum':<36} {'n':>10} {'P(frame)':>9} {'P(+5m gate)':>12}")
    for lab, s in (("all analysis-sample passes", d),
                   ("under_pressure = True", d[d.under_pressure.fillna(False)]),
                   ("under_pressure = False", d[~d.under_pressure.fillna(False)])):
        print(f"{lab:<36} {len(s):>10,} {s.have.mean():>9.3f} {s.gate.mean():>12.3f}")
    print(f"\n{'accumulated spine pressure':<36} {'n':>10} {'P(frame)':>9} {'P(+5m gate)':>12}")
    for k in [0, 1, 2, 3]:
        s = d[d.seg_press_count_spine == k]
        if len(s):
            print(f"{('= ' + str(k)):<36} {len(s):>10,} {s.have.mean():>9.3f} {s.gate.mean():>12.3f}")
    s = d[d.seg_press_count_spine >= 4]
    print(f"{'>= 4':<36} {len(s):>10,} {s.have.mean():>9.3f} {s.gate.mean():>12.3f}")

    print(f"\n{'pitch zone':<36} {'n':>10} {'P(frame)':>9} {'P(+5m gate)':>12}")
    for z in sorted(d.zone.dropna().unique()):
        s = d[d.zone == z]
        print(f"{z:<36} {len(s):>10,} {s.have.mean():>9.3f} {s.gate.mean():>12.3f}")
    print("\n  Wide channels (y0, y2) are the least covered. Press-to-touchline,")
    print("  which uses the sideline as an extra defender, is one of the most")
    print("  canonical pressing patterns in the game. Tier 2 systematically")
    print("  under-observes exactly that. Zone-as-control fixes the estimation;")
    print("  it does not recover the missing observations.")

    rule("1c. TIER 2 COMPOSITION")
    comp = (df[df.match_id.isin(t2)]
            .groupby(["competition_name", "season_name"], observed=True)["match_id"]
            .nunique().sort_values(ascending=False))
    tourn = sum(n for (c, _), n in comp.items()
                if any(k in str(c) for k in TOURNAMENT_KEYS))
    for (c, s), n in comp.items():
        mark = "T" if any(k in str(c) for k in TOURNAMENT_KEYS) else " "
        print(f" {mark} {str(c):<34} {str(s):<12} {n:>5}")
    tot = comp.sum()
    print(f"\ninternational tournament matches: {tourn}/{tot} = {100*tourn/tot:.1f}%")


# --------------------------------------------------------------------------- #
def exposure_clock() -> None:
    rule("2. EXPOSURE CLOCK: pass-level vs ball-event spine")
    d = analysis_sample(load_passes(columns=[
        "is_possession_team", "is_set_piece_restart", "passes_since_last_press",
        "events_since_last_press", "up_lag1", "up_lag1_spine", "poss_press_count",
        "seg_press_count_spine", "lag1_spine_type",
    ]))
    print(f"analysis-sample passes: {len(d):,}")
    pc, sc = d.passes_since_last_press, d.events_since_last_press

    blind = (pc.isna() & sc.notna())
    print(f"\n  pass clock reports NO press in the segment   {pc.isna().sum():>10,}"
          f"  {100*pc.isna().mean():5.1f}%")
    print(f"  spine clock reports no press                 {sc.isna().sum():>10,}"
          f"  {100*sc.isna().mean():5.1f}%")
    print(f"\n  >>> pass clock blind to a real press:        {blind.sum():>10,}"
          f"  {100*blind.mean():5.1f}% of the sample")
    print(f"      as a share of pass-clock nulls: {100*blind.sum()/max(pc.isna().sum(),1):.1f}%")

    both = pc.notna() & sc.notna()
    print(f"\n  both clocks defined: {both.sum():,}"
          f"   identical {100*(pc[both] == sc[both]).mean():.1f}%"
          f"   spine more recent {100*(sc[both] < pc[both]).mean():.1f}%")

    l = d.up_lag1.notna() & d.up_lag1_spine.notna()
    dis = d.up_lag1[l] != d.up_lag1_spine[l]
    print(f"\n  lag-1 pressure state differs on {100*dis.mean():.1f}% of passes"
          f"  ({int(dis.sum()):,} of {int(l.sum()):,})")
    print(f"      spine says pressed where the pass clock says not: "
          f"{int(((d.up_lag1_spine[l]) & (~d.up_lag1[l])).sum()):,}")
    print(f"\n  mean accumulated presses: pass {d.poss_press_count.mean():.3f}"
          f"   spine {d.seg_press_count_spine.mean():.3f}"
          f"   ratio {d.seg_press_count_spine.mean()/d.poss_press_count.mean():.2f}x")
    print("\n  ball event immediately preceding a pass:")
    for k, v in d.lag1_spine_type.value_counts(dropna=False).head(6).items():
        print(f"      {str(k):<10} {v:>10,}  {100*v/len(d):5.1f}%")


# --------------------------------------------------------------------------- #
def press_run_exits() -> None:
    rule("3. PRESS-RUN EXIT DECOMPOSITION  (over the spine)")
    s = load_spine(columns=[
        "event_type", "spine_role", "press_definitional", "under_pressure",
        "press_run_is_last_spine", "press_run_exit_spine", "press_run_len_spine",
        "presser_dist", "presser_dist_to_end", "prog_dist", "end_x",
    ])
    print("spine composition (under_pressure rate decides the role):")
    comp = s.groupby("event_type", observed=True).agg(
        n=("under_pressure", "size"), up_rate=("under_pressure", "mean"))
    role = s.groupby("event_type", observed=True)["spine_role"].first()
    for t in comp.sort_values("n", ascending=False).index:
        print(f"   {t:<14} {comp.loc[t,'n']:>9,}  up={comp.loc[t,'up_rate']:.4f}"
              f"   role={role[t]}")

    last = s[s.press_run_is_last_spine.fillna(False)]
    vc = last.press_run_exit_spine.value_counts(dropna=False)
    print(f"\npress runs over Pass+Carry: {len(last):,}")
    print(f"\n{'exit route':<18} {'runs':>10} {'share':>8}")
    for k in vc.index:
        print(f"{str(k):<18} {vc[k]:>10,} {100*vc[k]/vc.sum():>7.1f}%")
    print(f"\n{'competing risk':<18} {'runs':>10} {'share':>8}")
    for name, keys in EXIT_GROUPS.items():
        n = int(vc.reindex(keys).fillna(0).sum())
        print(f"{name:<18} {n:>10,} {100*n/vc.sum():>7.1f}%")

    print("\nexit mix by run length:")
    lr = last.assign(run_len=last.press_run_len_spine.clip(upper=6))
    tab = pd.crosstab(lr.run_len, lr.press_run_exit_spine, normalize="index") * 100
    cnt = lr.run_len.value_counts().sort_index()
    order = [c for c in ["turnover", "escape", "shot", "stoppage_foul",
                         "stoppage_out", "clearance", "other"] if c in tab.columns]
    print(f"{'len':>5} {'n':>9}  " + "".join(f"{c:>14}" for c in order))
    for i in tab.index:
        print(f"{int(i):>5} {cnt[i]:>9,}  "
              + "".join(f"{tab.loc[i, c]:>13.1f}%" for c in order))
    print("  len 6 = '6 or more'.")


# --------------------------------------------------------------------------- #
def escape_definition() -> None:
    rule("4. ESCAPE, POSITIVELY DEFINED")
    s = load_spine(columns=[
        "press_run_is_last_spine", "press_run_exit_spine", "presser_dist",
        "presser_dist_to_end", "prog_dist", "end_x", "event_type",
    ])
    last = s[s.press_run_is_last_spine.fillna(False)]
    n_runs = len(last)
    esc = last[last.press_run_exit_spine == "escape"]
    print(f"structural escapes: {len(esc):,} of {n_runs:,} runs "
          f"({100*len(esc)/n_runs:.1f}%)")
    print("  structural = possession retained and the next Pass/Carry is not")
    print("  flagged under_pressure. That is an ABSENCE OF ANNOTATION, so it needs")
    print("  a positive signature before it can carry a competing-risks category.")

    gain = (esc.presser_dist_to_end.astype("Float64")
            - esc.presser_dist.astype("Float64"))
    prog = esc.prog_dist.astype("Float64")
    g = gain.dropna().astype(float)
    p = prog.dropna().astype(float)
    print(f"\n  separation gained from the presser over the terminating event:")
    print(f"     n={len(g):,} ({100*len(g)/len(esc):.1f}% coverage)"
          f"  median {np.median(g):+.2f} m  frac>0 {100*np.mean(g>0):.1f}%")
    print(f"  progression toward goal:")
    print(f"     n={len(p):,} ({100*len(p)/len(esc):.1f}% coverage)"
          f"  median {np.median(p):+.2f} m  frac>0 {100*np.mean(p>0):.1f}%")

    cands = {
        "E0 structural (current)": pd.Series(True, index=esc.index),
        "E1 ball ends >=10 m from the presser": esc.presser_dist_to_end.astype("Float64") >= 10,
        "E2 gained >=5 m of separation": gain >= 5,
        "E3 progressed >=5 m toward goal": prog >= 5,
        "E4 gained >=5 m AND progressed >=5 m": (gain >= 5) & (prog >= 5),
        "E5 gained separation AND progressed (any)": (gain > 0) & (prog > 0),
    }
    print(f"\n  {'definition':<44} {'runs':>9} {'of escapes':>11} {'of all runs':>12}")
    for name, m in cands.items():
        n = int(m.fillna(False).sum())
        print(f"  {name:<44} {n:>9,} {100*n/len(esc):>10.1f}% {100*n/n_runs:>11.1f}%")
    print("\n  Separation has a clear positive signature; progression does not.")
    print("  The modal escape keeps the ball and gets away from the presser without")
    print("  going forward, so 'escape' should be split into relief and progressive")
    print("  escape rather than treated as one risk.")


# Presser drift, measured against 360 ground truth: the Pressure event records the
# presser's position at t-lead, the freeze frame records every opponent at the
# moment of the pass, and the gap between them as a function of lead is the drift.
# Fitted over 14,532 pressed passes with a freeze frame across 120 matches:
#     median gap = 2.46 m + 1.02 m/s * lead
# The 2.46 m intercept is the irreducible floor (the nearest visible opponent is
# not necessarily the annotated presser, plus annotation precision).
DRIFT_M_PER_S = 1.02
DRIFT_FLOOR_M = 2.46


def presser_staleness() -> None:
    rule("5. PRESSER STALENESS AND FALSIFICATION TEST 3b")
    d = analysis_sample(load_passes(columns=[
        "segment_uid", "event_index", "match_seconds", "is_possession_team",
        "is_set_piece_restart", "lag1_presser_dist_to_t", "lag1_pressure_lead_s",
        "presser_involved_at_t",
    ])).sort_values(["segment_uid", "event_index"])

    dt = (d.match_seconds.astype("Float64")
          - d.groupby("segment_uid", sort=False)["match_seconds"]
             .shift(1).astype("Float64"))
    tot = d.lag1_pressure_lead_s.astype("Float64") + dt
    err = DRIFT_FLOOR_M + DRIFT_M_PER_S * tot
    m = d.lag1_presser_dist_to_t.notna() & tot.notna()

    t = tot[m].astype(float)
    e = err[m].astype(float)
    v = d.lag1_presser_dist_to_t[m].astype(float)
    print(f"lag1_presser_dist_to_t is available on {int(m.sum()):,} passes.")
    print("Its staleness compounds: the t-1 presser position is recorded at")
    print("(t-1 minus lead), while the carrier position is at t.")
    print(f"\n  total elapsed time  median {np.median(t):.2f} s"
          f"   p90 {np.percentile(t, 90):.2f} s")
    print(f"  implied position error  median {np.median(e):.2f} m"
          f"   p90 {np.percentile(e, 90):.2f} m")
    print(f"  the measure itself      median {np.median(v):.2f} m"
          f"   -> error/signal {np.median(e)/np.median(v):.2f}")

    print(f"\n  capping on staleness (the proposed sensitivity check):")
    print(f"  {'cap':<10} {'n':>11} {'% kept':>8} {'med err':>10} {'med measure':>13}"
          f" {'err/signal':>11}")
    for cap in [1.0, 2.0, 3.0, 5.0, float("inf")]:
        k = m & (tot <= cap)
        if int(k.sum()) < 100:
            continue
        ee = err[k].astype(float)
        vv = d.lag1_presser_dist_to_t[k].astype(float)
        lab = "none" if cap == float("inf") else f"<= {cap:.0f} s"
        print(f"  {lab:<10} {int(k.sum()):>11,} {100*k.sum()/m.sum():>7.1f}%"
              f" {np.median(ee):>9.2f} m {np.median(vv):>12.2f} m"
              f" {np.median(ee)/np.median(vv):>11.2f}")
    print("\n  Capping does NOT improve signal-to-noise. A short elapsed time also")
    print("  means the ball has not moved far, so the measure shrinks faster than")
    print("  the error does. Carry staleness as a control; do not cap.")

    idm = d.presser_involved_at_t.notna()
    print(f"\n  The discrete form of 3b carries no staleness at all:")
    print(f"     comparable presser identity at t-1 and t: {int(idm.sum()):,} passes")
    print(f"     same defender pressing at both:           "
          f"{100*d.presser_involved_at_t[idm].mean():.1f}%")
    print("  Run 3b on presser IDENTITY as the primary contrast and distance as a")
    print("  continuous secondary. Identity is immune to positional drift.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["coverage", "clock", "exits", "escape",
                                       "staleness"])
    args = ap.parse_args()
    if args.only in (None, "coverage"):
        chain_coverage()
    if args.only in (None, "clock"):
        exposure_clock()
    if args.only in (None, "exits"):
        press_run_exits()
    if args.only in (None, "escape"):
        escape_definition()
    if args.only in (None, "staleness"):
        presser_staleness()


if __name__ == "__main__":
    main()
