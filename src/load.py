#!/usr/bin/env python3
"""
src/load.py -- the only sanctioned way to read the built tables.

Why this module exists
----------------------
`build.py` writes every "not applicable" value as NULL: no preceding press, no
t-1 within the segment, no freeze frame. Read the Parquet with pandas' default
backend and those columns come back as float64/object, at which point a stray
`.fillna(0)` or `.astype(int)` silently turns "we never observed a press here"
into "there were zero passes since the last press" -- a real number, in the
middle of the exposure clock the whole study depends on.

A README warning is not a guard. Import this instead:

    from src.load import load_passes, load_spine
    df = load_passes()

Both loaders force dtype_backend="numpy_nullable", so Int16/boolean survive and
missingness propagates through arithmetic instead of becoming a number.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
PASSES_PATH = PROCESSED / "passes.parquet"
SPINE_PATH = PROCESSED / "spine.parquet"

# Columns where a 0 is a legitimate value AND a NULL means "not applicable".
# Conflating them is the specific failure this module prevents.
NULL_CRITICAL = (
    "passes_since_last_press",
    "events_since_last_press",
    "passes_since_press_onset",
    "events_since_press_onset",
    "poss_press_frac",
    "seg_press_frac_spine",
    "presser_dist",
    "lag1_presser_dist_to_t",
    "time_since_receipt_s",
    "time_since_last_press_s",
)


# --------------------------------------------------------------------------- #
# post-treatment register
# --------------------------------------------------------------------------- #
# StatsBomb sets `end_location` to where the ball ACTUALLY ended. On an
# intercepted pass that is the interception point, not the intended target. Any
# feature derived from it is therefore partly a function of the outcome, and a
# model containing one predicts the outcome from the outcome.
#
# This has now happened twice in this project: pass_length/pass_angle in Stage 1
# (worth about a third of conventional model skill), and four ff_* columns in the
# first draft of the Stage 2 threat-B control (worth +7.0 Brier skill points
# against +0.07 for the clean block). The second instance came one stage after the
# first was documented, in unfamiliar columns. A prose warning did not transfer;
# a register plus a guard might.
#
# Membership is verified against build.py's source by validate.py, so this list
# cannot quietly fall out of date as columns are added.
# name -> (provenance, reason). Provenance "build" means build.py computes it
# from end_x/end_y, so validate.py's taint scan must find it; "statsbomb" means
# the provider already derived it from end_location before we saw it, so no
# amount of reading build.py reveals the dependency -- which is exactly why
# pass_length was believed innocent for as long as it was.
POST_TREATMENT = {
    "end_x": ("build", "the ball's actual endpoint; interception point on a failure"),
    "end_y": ("build", "the ball's actual endpoint; interception point on a failure"),
    "pass_length": ("statsbomb", "provider's Euclidean(start, end_location)"),
    "pass_angle": ("statsbomb", "provider's atan2 over end_location"),
    "prog_dist": ("build", "goal-ward progress measured to end_location"),
    "presser_dist_to_end": ("build", "presser distance measured to end_location"),
    "escape_sep_gain": ("build", "difference of two distances, one to end_location"),
    "ff_lane_opp": ("build", "opponents in the corridor from origin to end_location"),
    "ff_lane_visible": ("build", "visibility of that same corridor"),
    "ff_recv_opp_within_5": ("build", "opponents within 5 m of end_location"),
    "ff_recv_visible_r5": ("build", "visibility around end_location"),
}

# Columns the taint scan flags because end_location flows through the same
# expression, but whose VALUE provably does not depend on it. Each needs a reason
# that can be checked by reading, not a promise.
#
# This distinction is load-bearing, not bookkeeping: ff_visible_r5 is the SAMPLE
# GATE for the entire Tier 2 arm of the Stage 2 mechanism test. Were it
# target-derived, the gate would select on the outcome -- a worse defect than the
# control-block leak it was introduced to avoid.
TAINT_EXEMPT = {
    "ff_visible_r3":
        "reads probe[0] only, which is the ORIGIN; the target enters probe[1] "
        "and the lane samples. points_inside and points_edge_dist are strictly "
        "rowwise -- output element i depends only on input row i, no cross-row "
        "normalisation -- so index 0 cannot carry the target.",
    "ff_visible_r5":
        "same construction as ff_visible_r3, at the 5 m margin. Verified rowwise "
        "by reading both helpers in build.py.",
}


def assert_pre_treatment(columns, allow=()) -> None:
    """Refuse to build a design matrix out of outcome-derived features.

    Call this from any modelling script before fitting. `allow` is the escape
    hatch, and using it is a claim that you have a stated reason -- e.g. Stage 1's
    M1/M2/M3, which include pass_length deliberately in order to *quantify* the
    leakage against M0.

        assert_pre_treatment(X.columns)                       # clean specs
        assert_pre_treatment(X.columns, allow=("pass_length",))  # leakage bound
    """
    bad = sorted(set(columns) & set(POST_TREATMENT) - set(allow))
    if bad:
        reasons = "\n".join(f"    {c}: {POST_TREATMENT[c][1]}" for c in bad)
        raise ValueError(
            "post-treatment features in a design matrix -- each is partly a "
            f"function of the outcome:\n{reasons}\n"
            "  Pass allow=(...) if the specification includes them on purpose, "
            "and say so in the README."
        )


def _read(path: Path, columns: list[str] | None) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(
            f"[load] {path} not found. Run `python fetch.py` then `python build.py`."
        )
    return pd.read_parquet(path, columns=columns, dtype_backend="numpy_nullable")


def load_passes(columns: list[str] | None = None) -> pd.DataFrame:
    """One row per pass. The estimation table."""
    return _read(PASSES_PATH, columns)


def load_spine(columns: list[str] | None = None) -> pd.DataFrame:
    """One row per pressed-eligible ball event. The exposure clock lives here."""
    return _read(SPINE_PATH, columns)


def analysis_sample(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Passes eligible for Stage 1 baseline fitting.

    Excludes non-possession-team passes (5.9%; a possession can contain passes
    from both sides) and set-piece restarts (no press is possible on a dead ball
    and completion rates differ wildly, so they are a different data-generating
    process). Rows remain in the table; this is only the estimation filter.
    """
    if df is None:
        df = load_passes()
    return df[
        df["is_possession_team"].fillna(False)
        & ~df["is_set_piece_restart"].fillna(False)
    ]


def assert_null_policy(df: pd.DataFrame) -> None:
    """Fail loudly if a null-critical column arrived as a non-nullable dtype."""
    bad = [
        c for c in NULL_CRITICAL
        if c in df.columns and not isinstance(df[c].dtype, pd.api.extensions.ExtensionDtype)
    ]
    if bad:
        raise TypeError(
            "these columns lost their nullable dtype, so NULL may have become a "
            f"number: {bad}. Load via src.load, not bare pd.read_parquet."
        )
