#!/usr/bin/env python3
"""
pressures.py -- extract every `Pressure` event with its duration.

Why this table exists
---------------------
`build.py` attaches a presser to a ball event through `related_events` and keeps
the presser's identity, location and *lead* (how long before the ball event the
pressure was logged). It does not keep the Pressure event's own `duration`.

That omission matters for Stage 2. The primary estimand contrasts unpressed
passes that follow a press against unpressed passes that do not, binned by
seconds elapsed since the press ended. But "the press ended" is defined by the
last *ball event* carrying `under_pressure`, not by the pressure itself ending.
A Pressure event is a window, not an instant:

    duration quantiles over the full corpus, seconds
    p10 0.309   p25 0.471   p50 0.729   p75 1.138   p90 1.764   max 9.195

So a pass logged 0.4 s after a pressed carry, and flagged unpressed, may still
lie inside a live pressure window. Counting it as memory would be counting
contemporaneous pressure that the event-level flag failed to carry forward.

This module writes one row per Pressure event so that question can be answered
directly, without going through `related_events` at all -- the linkage and the
window are independent measurements, and Stage 2 should not depend on the same
chain twice.

Coordinates are stored RAW, in the pressing (defending) team's attacking frame,
exactly as StatsBomb logs them. Consumers mirror at join time via
`build.mirror`, because only the consumer knows which team is acting. Storing a
pre-mirrored coordinate here would bake in an assumption this table cannot
check -- and the frame bug that cost this project a day was exactly that class
of error.

Usage
-----
    python pressures.py                 # all matches -> data/processed/pressures.parquet
    python pressures.py --limit 50      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "open-data" / "data"
OUT_DIR = ROOT / "data" / "processed"
OUT_PRESSURES = OUT_DIR / "pressures.parquet"

SCHEMA = pa.schema([
    ("event_id", pa.string()),
    ("match_id", pa.int32()),
    ("period", pa.int8()),
    ("period_seconds", pa.float32()),   # period-relative, same clock as passes
    ("duration_s", pa.float32()),
    ("team_id", pa.int32()),            # the PRESSING team
    ("player_id", pa.int32()),
    ("x", pa.float32()),                # raw, pressing team's attacking frame
    ("y", pa.float32()),
    ("counterpress", pa.bool_()),
])
COLUMNS = [f.name for f in SCHEMA]


def ts_seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_match(path: Path) -> tuple[list[dict], str | None]:
    try:
        with open(path, "rb") as fh:
            events = json.load(fh)
    except Exception as exc:                                # noqa: BLE001
        return [], f"{path.stem}: {exc}"

    mid = int(path.stem)
    out: list[dict] = []
    for e in events:
        if e["type"]["name"] != "Pressure":
            continue
        if e["period"] == 5:                                # shootout
            continue
        loc = e.get("location")
        dur = e.get("duration")
        out.append({
            "event_id": e["id"],
            "match_id": mid,
            "period": e["period"],
            "period_seconds": ts_seconds(e["timestamp"]),
            # NULL, never 0.0 -- a missing duration is not a zero-length press.
            "duration_s": float(dur) if dur is not None else None,
            "team_id": e["team"]["id"],
            "player_id": e["player"]["id"] if e.get("player") else None,
            "x": float(loc[0]) if loc else None,
            "y": float(loc[1]) if loc else None,
            "counterpress": bool(e.get("counterpress", False)),
        })
    return out, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract Pressure events with durations")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    args = ap.parse_args()

    files = sorted((RAW / "events").glob("*.json"), key=lambda p: int(p.stem))
    if not files:
        sys.exit("[pressures] no raw events found. Run `python fetch.py` first.")
    if args.limit:
        files = files[: args.limit]
    print(f"[pressures] {len(files)} matches, {args.workers} workers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(OUT_PRESSURES, SCHEMA, compression="zstd")
    buf: list[dict] = []
    n = done = 0
    errors: list[str] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rows, err in ex.map(parse_match, files, chunksize=8):
            done += 1
            if err:
                errors.append(err)
                continue
            buf.extend(rows)
            if len(buf) >= 200_000:
                df = pd.DataFrame(buf, columns=COLUMNS)
                writer.write_table(
                    pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                n += len(df)
                buf = []
            if done % 500 == 0:
                print(f"[pressures] {done}/{len(files)} matches, {n + len(buf):,} rows")

    if buf:
        df = pd.DataFrame(buf, columns=COLUMNS)
        writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
        n += len(df)
    writer.close()

    print(f"[pressures] wrote {n:,} rows -> {OUT_PRESSURES}")
    if errors:
        print(f"[pressures] {len(errors)} match(es) failed:")
        for e in errors[:10]:
            print("   ", e)


if __name__ == "__main__":
    main()
