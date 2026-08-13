#!/usr/bin/env python3
"""
diagnostics.py -- study-design diagnostics computed from data/processed/passes.parquet.

Two questions, both answerable before any model is fitted:

  1. CHAIN COVERAGE. Stage 2 needs freeze frames at t, t-1 and sometimes t-2, each
     passing the visible-area gate. Per-event coverage is not the relevant number.
     This reports the actual joint coverage over the lag chain and contrasts it
     with the naive independence product, which shows whether frame availability
     clusters (it is not missing at random).

  2. PRESS-RUN EXIT DECOMPOSITION. A press run can end by turnover, by escape, or
     by stoppage. These select in opposite directions, so the mix determines
     whether a competing-risks model is needed and how it must be specified.

Reads the Parquet only. Computes nothing that is not in the table.

    python diagnostics.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PARQUET = ROOT / "data" / "processed" / "passes.parquet"

# International tournaments, for the Tier 2 composition question.
TOURNAMENT_KEYS = (
    "World Cup", "UEFA Euro", "Women's Euro", "African Cup", "Copa America",
)


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load() -> pd.DataFrame:
    if not PARQUET.exists():
        raise SystemExit(f"[diag] {PARQUET} not found. Run `python build.py` first.")
    # numpy_nullable keeps Int16/boolean rather than degrading to float64/object,
    # so "no history here" stays NA through every downstream operation.
    return pd.read_parquet(PARQUET, dtype_backend="numpy_nullable")


# --------------------------------------------------------------------------- #
# 1. chain coverage
# --------------------------------------------------------------------------- #
def chain_coverage(df: pd.DataFrame) -> None:
    rule("1. TIER 2 CHAIN COVERAGE  (frames needed at t, t-1, t-2)")

    tier2_matches = df.loc[df["ff_available"].fillna(False), "match_id"].unique()
    d = df[df["match_id"].isin(tier2_matches)].copy()
    print(f"Tier 2 matches (>=1 freeze frame): {len(tier2_matches)}")

    # The analysis sample: possession-team passes, excluding set-piece restarts.
    d = d[d["is_possession_team"].fillna(False) & ~d["is_set_piece_restart"].fillna(False)]
    d = d.sort_values(["segment_uid", "event_index"])
    print(f"analysis-sample passes in those matches: {len(d):,}")

    have = d["ff_available"].fillna(False).to_numpy()
    gate = have & d["ff_visible_r5"].fillna(False).to_numpy()
    gate3 = have & d["ff_visible_r3"].fillna(False).to_numpy()

    seg = d["segment_uid"]

    def lagged(col: np.ndarray, k: int) -> np.ndarray:
        """Value of `col` k passes earlier in the same segment; False if absent."""
        s = pd.Series(col, index=d.index)
        return (
            s.groupby(seg, sort=False)
            .shift(k)
            .fillna(False)
            .to_numpy()
            .astype(bool)
        )

    rows = []
    for label, base in (("frame present", have),
                        ("frame + r3 gate", gate3),
                        ("frame + r5 gate", gate)):
        l1 = lagged(base, 1)
        l2 = lagged(base, 2)
        # a lag only exists inside the same segment; require it to exist at all
        ord_seg = d["pass_ord_in_seg"].to_numpy()
        has1 = ord_seg >= 1
        has2 = ord_seg >= 2

        p_t = base.mean()
        # joint coverage among passes that actually have the required history
        j1 = (base & l1)[has1].mean() if has1.any() else np.nan
        j2 = (base & l1 & l2)[has2].mean() if has2.any() else np.nan
        rows.append((label, p_t, j1, j2, p_t**2, p_t**3))

    print(f"\n{'gate':<18} {'P(t)':>8} {'P(t,t-1)':>10} {'P(t,t-1,t-2)':>14}"
          f" {'naive^2':>9} {'naive^3':>9}")
    for label, p, j1, j2, n2, n3 in rows:
        print(f"{label:<18} {p:>8.3f} {j1:>10.3f} {j2:>14.3f} {n2:>9.3f} {n3:>9.3f}")

    print("\n  'naive^k' = P(t)^k, i.e. what joint coverage would be if frame")
    print("  availability were independent across consecutive passes. Observed")
    print("  above naive => availability clusters within segments.")

    # absolute usable N, the number that actually decides the design
    ord_seg = d["pass_ord_in_seg"].to_numpy()
    l1 = lagged(gate, 1)
    l2 = lagged(gate, 2)
    n_t = int(gate.sum())
    n_t1 = int((gate & l1 & (ord_seg >= 1)).sum())
    n_t2 = int((gate & l1 & l2 & (ord_seg >= 2)).sum())
    print(f"\nusable passes under the r5 gate:")
    print(f"   t only          {n_t:>9,}")
    print(f"   t and t-1       {n_t1:>9,}")
    print(f"   t, t-1 and t-2  {n_t2:>9,}")

    # composition
    rule("1b. TIER 2 COMPOSITION  (generalizability cost)")
    comp = (
        df[df["match_id"].isin(tier2_matches)]
        .groupby(["competition_name", "season_name"], observed=True)["match_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    total = comp.sum()
    tourn = sum(
        n for (c, s), n in comp.items() if any(k in str(c) for k in TOURNAMENT_KEYS)
    )
    print(f"{'competition':<38} {'season':<12} {'matches':>8}")
    for (c, s), n in comp.items():
        mark = "T" if any(k in str(c) for k in TOURNAMENT_KEYS) else " "
        print(f"{mark} {str(c):<36} {str(s):<12} {n:>8}")
    print(f"\ninternational tournament matches: {tourn}/{total} = {100*tourn/total:.1f}%")
    print("league matches:                  "
          f"{total - tourn}/{total} = {100*(total-tourn)/total:.1f}%")


# --------------------------------------------------------------------------- #
# 2. press-run exits
# --------------------------------------------------------------------------- #
def press_run_exits(df: pd.DataFrame) -> None:
    rule("2. PRESS-RUN EXIT DECOMPOSITION")

    runs = df[df["press_run_is_last"].fillna(False)].copy()
    print(f"press runs identified: {len(runs):,}")
    print(f"  (a run = maximal consecutive under_pressure passes by the")
    print(f"   possession team within one segment_uid)")

    vc = runs["press_run_exit"].value_counts(dropna=False)
    pct = 100 * vc / vc.sum()
    print(f"\n{'exit route':<18} {'runs':>9} {'share':>8}")
    for k in vc.index:
        print(f"{str(k):<18} {vc[k]:>9,} {pct[k]:>7.1f}%")

    grouped = {
        "turnover": ["turnover"],
        "escape": ["escape", "shot"],
        "stoppage": ["stoppage_foul", "stoppage_out"],
        "other": ["period_end", "other"],
    }
    print(f"\ncollapsed to the three competing risks:")
    print(f"{'risk':<18} {'runs':>9} {'share':>8}")
    for name, keys in grouped.items():
        n = int(vc.reindex(keys).fillna(0).sum())
        print(f"{name:<18} {n:>9,} {100*n/vc.sum():>7.1f}%")
    print("\n  'escape' folds in runs that reached a shot; 'stoppage' splits into")
    print("  foul/injury vs ball-out-of-play above, since only the former is the")
    print("  whistle mechanism and the latter also transfers possession.")

    # exit mix by run length -- the shape that matters for competing risks
    rule("2b. EXIT MIX BY PRESS-RUN LENGTH")
    runs["run_len"] = runs["press_run_len"].clip(upper=6)
    tab = pd.crosstab(runs["run_len"], runs["press_run_exit"], normalize="index") * 100
    counts = runs["run_len"].value_counts().sort_index()
    order = [c for c in ["turnover", "escape", "shot", "stoppage_foul",
                         "stoppage_out", "period_end", "other"] if c in tab.columns]
    tab = tab[order]
    print(f"{'run len':>8} {'n':>8}  " + "".join(f"{c:>15}" for c in order))
    for i in tab.index:
        cells = "".join(f"{tab.loc[i, c]:>14.1f}%" for c in order)
        print(f"{int(i):>8} {counts[i]:>8,}  {cells}")
    print("\n  run len 6 = '6 or more'.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", choices=["coverage", "exits"], default=None)
    args = ap.parse_args()

    df = load()
    print(f"[diag] loaded {len(df):,} passes from {PARQUET.name}")
    if args.only in (None, "coverage"):
        chain_coverage(df)
    if args.only in (None, "exits"):
        press_run_exits(df)


if __name__ == "__main__":
    main()
