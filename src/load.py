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
