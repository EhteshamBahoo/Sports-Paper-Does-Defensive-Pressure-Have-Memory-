#!/usr/bin/env python3
"""
build.py -- parse StatsBomb raw JSON into ONE pass-level table.

    data/raw/open-data/data/{events,matches,three-sixty}/*.json
        -> data/processed/passes.parquet          (one row per pass)

Every downstream stage reads that Parquet file. No analysis step re-opens the
raw JSON. Run `python fetch.py` first.

Dtype policy
------------
Anything that can be "not applicable" uses a nullable Arrow type and is written
as NULL. Absent history is never 0 and is never forward-filled. Read the table
back with `pd.read_parquet(...)` and the nullability survives.

Reset boundaries
----------------
possession_uid  = match_id : period : possession
    StatsBomb `possession` ids can span a half boundary (~1 per match), so the
    period is part of the key.

segment_uid     = possession_uid # k
    k increments at every set-piece restart pass. All pressure-history counters
    reset here: a stoppage dissolves the press, so history must not carry across
    it. Set-piece restart passes are retained as rows and flagged
    `is_set_piece_restart` so the choice is testable rather than baked in.

Usage
-----
    python build.py                  # full corpus
    python build.py --limit 200      # first 200 matches, for a smoke test
    python build.py --tier2-only     # only matches with a freeze-frame file
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "open-data" / "data"
OUT_DIR = ROOT / "data" / "processed"
OUT_PATH = OUT_DIR / "passes.parquet"

GOAL = np.array([120.0, 40.0])       # StatsBomb pitch is 120 x 80
LANE_HALF_WIDTH = 3.0                # metres either side of the passing lane
VIS_MARGIN = (3.0, 5.0)              # radii checked against visible_area

SET_PIECE_TYPES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}
FAIL_OUTCOMES = {"Incomplete", "Pass Offside"}
NULL_OUTCOMES = {"Unknown", "Injury Clearance"}

# Events after which a player no longer holds the ball. Seeing one invalidates
# their cached Ball Receipt*, so the decision window is only ever measured over
# an unbroken hold. Without this, a player who received, lost the ball, then
# recovered it later in the same possession gets a spuriously long window.
BALL_RELEASE_TYPES = {
    "Shot", "Clearance", "Miscontrol", "Dispossessed", "Foul Committed", "Error",
}


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def ts_seconds(stamp: str) -> float:
    """'00:12:34.567' -> seconds since the start of the period."""
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mirror(x: float, y: float) -> tuple[float, float]:
    """Rotate a location 180 degrees into the opposing team's attacking frame.

    StatsBomb logs every event in the frame of the team performing it: the acting
    team always attacks toward x=120. A `Pressure` event is performed by the
    DEFENDING team, so its coordinates are rotated relative to the pass it acts
    on, and the two cannot be compared until one is mirrored. Verified on 6,795
    linked pass/pressure pairs: raw separation has median 65.7 m, mirrored 3.9 m.
    """
    return 120.0 - x, 80.0 - y


def zone_of(x: float, y: float) -> str:
    third = "def" if x < 40 else ("mid" if x < 80 else "att")
    band = "y0" if y < 80 / 3 else ("y1" if y < 160 / 3 else "y2")
    return f"{third}_{band}"


def dist_to_goal(x: float, y: float) -> float:
    return float(math.hypot(GOAL[0] - x, GOAL[1] - y))


def poly_from_visible_area(va) -> np.ndarray:
    pts = np.asarray(va, dtype=float).reshape(-1, 2)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def points_inside(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Crossing-number point-in-polygon, vectorised over P."""
    x, y = P[:, 0], P[:, 1]
    inside = np.zeros(len(P), dtype=bool)
    n = len(V)
    for i in range(n):
        x1, y1 = V[i]
        x2, y2 = V[(i + 1) % n]
        straddles = (y1 > y) != (y2 > y)
        dy = y2 - y1
        with np.errstate(divide="ignore", invalid="ignore"):
            xint = np.where(dy == 0, np.inf, (x2 - x1) * (y - y1) / dy + x1)
        inside ^= straddles & (x < xint)
    return inside


def points_edge_dist(P: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Minimum distance from each point to the polygon boundary."""
    best = np.full(len(P), np.inf)
    n = len(V)
    for i in range(n):
        a = V[i]
        b = V[(i + 1) % n]
        ab = b - a
        L = float(ab @ ab)
        if L == 0.0:
            d = np.linalg.norm(P - a, axis=1)
        else:
            t = np.clip(((P - a) @ ab) / L, 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


# --------------------------------------------------------------------------- #
# per-match parse
# --------------------------------------------------------------------------- #
def parse_match(task: tuple) -> tuple:
    """Return (rows, score_check, error) for one match."""
    match_meta, ev_path, ff_path = task
    try:
        with open(ev_path, "rb") as fh:
            events = json.load(fh)
    except Exception as exc:                       # noqa: BLE001
        return [], None, f"{match_meta['match_id']}: {exc}"

    events.sort(key=lambda e: e["index"])
    by_id = {e["id"]: e for e in events}

    frames = {}
    if ff_path is not None:
        try:
            with open(ff_path, "rb") as fh:
                frames = {f["event_uuid"]: f for f in json.load(fh)}
        except Exception:                          # noqa: BLE001
            frames = {}

    home_id = match_meta["home_team_id"]
    away_id = match_meta["away_team_id"]

    # ---- running score, own goals included -------------------------------- #
    # StatsBomb logs a own goal twice: "Own Goal Against" for the team that put
    # it in, "Own Goal For" for the beneficiary. Counting only "Own Goal For"
    # avoids double counting. Shootouts (period 5) never count.
    goals = {home_id: 0, away_id: 0}
    score_before: dict[str, tuple[int, int]] = {}
    for e in events:
        score_before[e["id"]] = (goals[home_id], goals[away_id])
        if e["period"] == 5:
            continue
        etype = e["type"]["name"]
        tid = e["team"]["id"]
        if etype == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            if tid in goals:
                goals[tid] += 1
        elif etype == "Own Goal For":
            if tid in goals:
                goals[tid] += 1

    score_check = {
        "match_id": match_meta["match_id"],
        "recon_home": goals[home_id],
        "recon_away": goals[away_id],
        "index_home": match_meta["home_score"],
        "index_away": match_meta["away_score"],
    }
    score_check["ok"] = (
        score_check["recon_home"] == score_check["index_home"]
        and score_check["recon_away"] == score_check["index_away"]
    )

    # ---- pass 1: flat per-pass records ------------------------------------ #
    rows = []

    # Who currently holds the ball, as (possession_uid, player_id, receipt_time).
    # Only one player holds the ball at a time, so any new receipt invalidates the
    # previous hold. The decision window is therefore only ever measured across an
    # unbroken hold: received it, still has it, releases it.
    hold: tuple[str, int, float] | None = None

    seg_counter: dict[str, int] = {}
    row_of_event: dict[str, int] = {}

    for e in events:
        period = e["period"]
        if period == 5:                       # penalty shootout
            continue
        etype = e["type"]["name"]
        puid = f"{match_meta['match_id']}:{period}:{e['possession']}"
        t_ev = ts_seconds(e["timestamp"])

        if etype == "Ball Receipt*" and e.get("player"):
            hold = (puid, e["player"]["id"], t_ev)
        elif etype in BALL_RELEASE_TYPES:
            hold = None

        if etype != "Pass":
            continue

        pa_ = e["pass"]
        loc = e.get("location")
        end = pa_.get("end_location")
        if not loc or not end:
            continue

        team_id = e["team"]["id"]
        poss_team_id = e["possession_team"]["id"]
        opp_id = away_id if team_id == home_id else home_id
        ptype = pa_.get("type", {}).get("name", "OPEN_PLAY")
        is_restart = ptype in SET_PIECE_TYPES

        # segment id: bumped at every set-piece restart
        if is_restart:
            seg_counter[puid] = seg_counter.get(puid, 0) + 1
        k = seg_counter.get(puid, 0)
        seg_uid = f"{puid}#{k}"

        outcome = pa_.get("outcome", {}).get("name", "COMPLETE")
        if outcome == "COMPLETE":
            success = True
        elif outcome in FAIL_OUTCOMES or outcome == "Out":
            success = False
        else:
            success = None                      # Unknown / Injury Clearance

        x, y = float(loc[0]), float(loc[1])
        ex, ey = float(end[0]), float(end[1])

        # ---- linked Pressure event(s) ------------------------------------- #
        rel = [by_id[r] for r in e.get("related_events", []) if r in by_id]
        pressures = [r for r in rel if r["type"]["name"] == "Pressure"]
        presser_id = presser_x = presser_y = presser_dist = lead = None
        presser_cp = None
        if pressures:
            # Pressure locations are in the DEFENDING team's frame; mirror them
            # into this passer's frame before any distance is taken.
            best = None
            for pr in pressures:
                pl = pr.get("location")
                if pl:
                    px, py = pl
                    if pr["team"]["id"] != team_id:
                        px, py = mirror(float(px), float(py))
                    d = math.hypot(px - x, py - y)
                else:
                    px = py = None
                    d = math.inf
                if best is None or d < best[0]:
                    best = (d, pr, px, py)
            d, pr, px, py = best
            if pr.get("player"):
                presser_id = pr["player"]["id"]
            if px is not None:
                presser_x, presser_y = float(px), float(py)
                presser_dist = float(d)
            lead = t_ev - ts_seconds(pr["timestamp"])
            presser_cp = bool(pr.get("counterpress", False))

        # ---- linked Ball Receipt* (the receiver's, i.e. forward in time) --- #
        receipts = [r for r in rel if r["type"]["name"] == "Ball Receipt*"]
        receipt_up = bool(receipts[0].get("under_pressure", False)) if receipts else None

        # ---- decision window: this passer's own most recent receipt ------- #
        # player_pressed_earlier_in_poss needs strictly-before semantics and is
        # filled in pass 2, where the event stream is replayed in order.
        pid = e["player"]["id"] if e.get("player") else None

        time_since_receipt = None
        if hold is not None and pid is not None and hold[0] == puid and hold[1] == pid:
            gap = t_ev - hold[2]
            time_since_receipt = gap if gap >= 0 else None
        hold = None                      # the ball has now left the passer

        row = {
            "event_id": e["id"],
            "match_id": match_meta["match_id"],
            "event_index": e["index"],
            "period": period,
            "possession": e["possession"],
            "possession_uid": puid,
            "segment_uid": seg_uid,
            "competition_id": match_meta["competition_id"],
            "season_id": match_meta["season_id"],
            "comp_season_uid": match_meta["comp_season_uid"],
            "competition_name": match_meta["competition_name"],
            "season_name": match_meta["season_name"],
            "match_date": match_meta["match_date"],
            "team_id": team_id,
            "possession_team_id": poss_team_id,
            "is_possession_team": team_id == poss_team_id,
            "opponent_team_id": opp_id,
            "player_id": pid,
            "is_home": team_id == home_id,
            "period_seconds": t_ev,
            "match_seconds": None,             # filled in pass 2
            "score_diff": None,                # filled in pass 2
            "x": x, "y": y, "end_x": ex, "end_y": ey,
            "pass_length": float(pa_.get("length")) if pa_.get("length") is not None else None,
            "pass_angle": float(pa_.get("angle")) if pa_.get("angle") is not None else None,
            "pass_height": pa_.get("height", {}).get("name"),
            "pass_body_part": pa_.get("body_part", {}).get("name"),
            "pass_type": ptype,
            "play_pattern": e["play_pattern"]["name"],
            "pass_outcome_raw": outcome,
            "pass_success": success,
            "recipient_id": pa_.get("recipient", {}).get("id"),
            "is_set_piece_restart": is_restart,
            "zone": zone_of(x, y),
            "dist_to_goal": dist_to_goal(x, y),
            "prog_dist": dist_to_goal(x, y) - dist_to_goal(ex, ey),
            "under_pressure": bool(e.get("under_pressure", False)),
            "counterpress": bool(e.get("counterpress", False)),
            "n_pressure_linked": len(pressures),
            "presser_id": presser_id,
            "presser_x": presser_x,
            "presser_y": presser_y,
            "presser_dist": presser_dist,
            "pressure_lead_s": lead,
            "presser_counterpress": presser_cp,
            "receipt_under_pressure": receipt_up,
            "time_since_receipt_s": time_since_receipt,
            "_score_before": score_before[e["id"]],
            "_home_id": home_id,
        }
        row_of_event[e["id"]] = len(rows)
        rows.append(row)

    if not rows:
        return [], score_check, None

    # ---- pass 2: match-level derived fields ------------------------------- #
    period_offsets = {}
    acc = 0.0
    for p in sorted({e["period"] for e in events if e["period"] != 5}):
        period_offsets[p] = acc
        last = max(
            (ts_seconds(e["timestamp"]) for e in events if e["period"] == p),
            default=0.0,
        )
        acc += last

    # strictly-before per-player pressure, recomputed exactly
    pressed_seen: dict[tuple[str, int], bool] = {}
    for e in events:
        if e["period"] == 5:
            continue
        puid = f"{match_meta['match_id']}:{e['period']}:{e['possession']}"
        if e["type"]["name"] == "Pass" and e["id"] in row_of_event:
            r = rows[row_of_event[e["id"]]]
            key = (puid, r["player_id"])
            r["player_pressed_earlier_in_poss"] = bool(pressed_seen.get(key, False))
        if e.get("under_pressure") and e.get("player"):
            pressed_seen[(puid, e["player"]["id"])] = True

    for r in rows:
        r["match_seconds"] = period_offsets.get(r["period"], 0.0) + r["period_seconds"]
        h, a = r.pop("_score_before")
        hid = r.pop("_home_id")
        r["score_diff"] = (h - a) if r["team_id"] == hid else (a - h)

    # ---- pass 3: pressure history within segment -------------------------- #
    _add_history(rows)

    # ---- pass 4: press-run exits (needs the raw event stream) ------------- #
    _add_press_runs(rows, events, row_of_event)

    # ---- pass 5: 360 geometry --------------------------------------------- #
    _add_360(rows, frames)

    return rows, score_check, None


def _add_history(rows: list[dict]) -> None:
    """Pressure-history counters, computed within segment_uid on team passes."""
    for r in rows:
        for c in (
            "pass_ord_in_poss", "pass_ord_in_seg", "up_lag1", "up_lag2",
            "presser_dist_lag1", "press_run_len", "passes_since_press_onset",
            "passes_since_last_press", "time_since_last_press_s",
            "poss_press_count", "poss_press_frac", "up_lead1", "poss_ends_at_t",
            "presser_id_lag1", "presser_involved_at_t", "lag1_presser_dist_to_t",
            "zone_lag1",
        ):
            r.setdefault(c, None)

    by_poss: dict[str, list[dict]] = {}
    by_seg: dict[str, list[dict]] = {}
    for r in rows:
        if not r["is_possession_team"]:
            continue
        by_poss.setdefault(r["possession_uid"], []).append(r)
        by_seg.setdefault(r["segment_uid"], []).append(r)

    for seq in by_poss.values():
        seq.sort(key=lambda r: r["event_index"])
        for i, r in enumerate(seq):
            r["pass_ord_in_poss"] = i
            r["poss_ends_at_t"] = i == len(seq) - 1

    for seq in by_seg.values():
        seq.sort(key=lambda r: r["event_index"])
        run_len = 0
        press_count = 0
        last_press_i = None
        last_press_t = None
        for i, r in enumerate(seq):
            prev = seq[i - 1] if i >= 1 else None
            prev2 = seq[i - 2] if i >= 2 else None
            nxt = seq[i + 1] if i + 1 < len(seq) else None

            r["pass_ord_in_seg"] = i
            r["up_lag1"] = prev["under_pressure"] if prev else None
            r["up_lag2"] = prev2["under_pressure"] if prev2 else None
            r["presser_dist_lag1"] = prev["presser_dist"] if prev else None
            r["up_lead1"] = nxt["under_pressure"] if nxt else None

            if prev:
                r["presser_id_lag1"] = prev["presser_id"]
                r["zone_lag1"] = prev["zone"]
                if prev["presser_id"] is not None:
                    r["presser_involved_at_t"] = prev["presser_id"] == r["presser_id"]
                if prev["presser_x"] is not None:
                    r["lag1_presser_dist_to_t"] = math.hypot(
                        prev["presser_x"] - r["x"], prev["presser_y"] - r["y"]
                    )

            # counters describing history strictly before t
            r["poss_press_count"] = press_count
            r["poss_press_frac"] = (press_count / i) if i > 0 else None
            if last_press_i is None:
                r["passes_since_last_press"] = None
                r["time_since_last_press_s"] = None
            else:
                r["passes_since_last_press"] = i - last_press_i
                r["time_since_last_press_s"] = r["match_seconds"] - last_press_t

            if r["under_pressure"]:
                run_len += 1
                r["press_run_len"] = run_len
                r["passes_since_press_onset"] = run_len - 1
                r["passes_since_last_press"] = 0
                r["time_since_last_press_s"] = 0.0
                press_count += 1
                last_press_i = i
                last_press_t = r["match_seconds"]
            else:
                run_len = 0
                r["press_run_len"] = 0
                r["passes_since_press_onset"] = None


def _add_press_runs(rows: list[dict], events: list[dict], row_of_event: dict) -> None:
    """Label maximal under-pressure runs and classify how each one ended.

    Exit routes, in priority order, evaluated from the last pressed pass of a run:
      turnover       possession lost (failed pass, or possession ends with no
                     stoppage/shot/retained pass)
      escape         pass completed, possession retained, next team pass unpressed
      stoppage_out   the pass itself went out of play
      stoppage_foul  a foul/injury stoppage follows before the next team pass
      shot           the possession reaches a shot before the next team pass
      period_end     the half ended
    """
    for r in rows:
        r.setdefault("press_run_id", None)
        r.setdefault("press_run_is_last", None)
        r.setdefault("press_run_exit", None)

    by_seg: dict[str, list[dict]] = {}
    for r in rows:
        if r["is_possession_team"]:
            by_seg.setdefault(r["segment_uid"], []).append(r)

    idx_of = {e["index"]: i for i, e in enumerate(events)}

    for seg, seq in by_seg.items():
        seq.sort(key=lambda r: r["event_index"])
        run_no = 0
        i = 0
        while i < len(seq):
            if not seq[i]["under_pressure"]:
                i += 1
                continue
            j = i
            while j + 1 < len(seq) and seq[j + 1]["under_pressure"]:
                j += 1
            run_no += 1
            rid = f"{seg}#r{run_no}"
            for k in range(i, j + 1):
                seq[k]["press_run_id"] = rid
                seq[k]["press_run_is_last"] = k == j
            seq[j]["press_run_exit"] = _classify_exit(seq[j], events, idx_of)
            i = j + 1


def _classify_exit(r: dict, events: list[dict], idx_of: dict) -> str:
    outcome = r["pass_outcome_raw"]
    if outcome == "Out":
        return "stoppage_out"
    if outcome in FAIL_OUTCOMES:
        return "turnover"
    if outcome in NULL_OUTCOMES:
        return "other"

    start = idx_of.get(r["event_index"])
    if start is None:
        return "other"
    team = r["team_id"]
    poss = r["possession"]
    period = r["period"]

    for e in events[start + 1:]:
        if e["period"] != period or e["possession"] != poss:
            break                                   # possession ended
        etype = e["type"]["name"]
        tid = e["team"]["id"]
        if etype == "Foul Won" and tid == team:
            return "stoppage_foul"
        if etype == "Foul Committed":
            return "stoppage_foul" if tid != team else "turnover"
        if etype == "Injury Stoppage":
            return "stoppage_foul"
        if etype == "Half End":
            return "period_end"
        if etype == "Shot" and tid == team:
            return "shot"
        if etype == "Pass" and tid == team and e["possession_team"]["id"] == team:
            return "escape"
    return "turnover"


def _add_360(rows: list[dict], frames: dict) -> None:
    cols = (
        "ff_available", "ff_n_opp_visible", "ff_n_team_visible",
        "ff_visible_r3", "ff_visible_r5", "ff_recv_visible_r5",
        "ff_nearest_opp_dist", "ff_opp_within_3", "ff_opp_within_5",
        "ff_lane_opp", "ff_lane_visible", "ff_recv_opp_within_5",
    )
    for r in rows:
        for c in cols:
            r[c] = None
        r["ff_available"] = False

    if not frames:
        return

    for r in rows:
        fr = frames.get(r["event_id"])
        if fr is None:
            continue
        ff = fr.get("freeze_frame") or []
        if not ff:
            continue
        r["ff_available"] = True

        locs = np.array([p["location"] for p in ff], dtype=float)
        is_mate = np.array([bool(p.get("teammate")) for p in ff])
        opp = locs[~is_mate]
        r["ff_n_opp_visible"] = int((~is_mate).sum())
        r["ff_n_team_visible"] = int(is_mate.sum())

        origin = np.array([r["x"], r["y"]])
        target = np.array([r["end_x"], r["end_y"]])

        if len(opp):
            d = np.linalg.norm(opp - origin, axis=1)
            r["ff_nearest_opp_dist"] = float(d.min())
            r["ff_opp_within_3"] = int((d <= 3.0).sum())
            r["ff_opp_within_5"] = int((d <= 5.0).sum())
            dr = np.linalg.norm(opp - target, axis=1)
            r["ff_recv_opp_within_5"] = int((dr <= 5.0).sum())

            seg = target - origin
            L = float(seg @ seg)
            if L > 0:
                t = np.clip(((opp - origin) @ seg) / L, 0.0, 1.0)
                perp = np.linalg.norm(opp - (origin + t[:, None] * seg), axis=1)
                strictly_between = (t > 0.0) & (t < 1.0)
                r["ff_lane_opp"] = int(
                    ((perp <= LANE_HALF_WIDTH) & strictly_between).sum()
                )
            else:
                r["ff_lane_opp"] = 0
        else:
            r["ff_nearest_opp_dist"] = None
            r["ff_opp_within_3"] = 0
            r["ff_opp_within_5"] = 0
            r["ff_recv_opp_within_5"] = 0
            r["ff_lane_opp"] = 0

        va = fr.get("visible_area")
        if not va or len(va) < 6:
            continue
        V = poly_from_visible_area(va)
        if len(V) < 3:
            continue

        lane_ts = np.linspace(0.0, 1.0, 9)
        probe = np.vstack([origin, target, origin + (target - origin) * lane_ts[:, None]])
        ins = points_inside(probe, V)
        edge = points_edge_dist(probe, V)

        r["ff_visible_r3"] = bool(ins[0] and edge[0] >= VIS_MARGIN[0])
        r["ff_visible_r5"] = bool(ins[0] and edge[0] >= VIS_MARGIN[1])
        r["ff_recv_visible_r5"] = bool(ins[1] and edge[1] >= VIS_MARGIN[1])
        lane_ok = ins[2:] & (edge[2:] >= LANE_HALF_WIDTH)
        r["ff_lane_visible"] = bool(lane_ok.all())


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
SCHEMA = pa.schema([
    ("event_id", pa.string()),
    ("match_id", pa.int32()),
    ("event_index", pa.int32()),
    ("period", pa.int8()),
    ("possession", pa.int16()),
    ("possession_uid", pa.string()),
    ("segment_uid", pa.string()),
    ("competition_id", pa.int16()),
    ("season_id", pa.int16()),
    ("comp_season_uid", pa.string()),
    ("competition_name", pa.string()),
    ("season_name", pa.string()),
    ("match_date", pa.string()),
    ("team_id", pa.int32()),
    ("possession_team_id", pa.int32()),
    ("is_possession_team", pa.bool_()),
    ("opponent_team_id", pa.int32()),
    ("player_id", pa.int32()),
    ("is_home", pa.bool_()),
    ("period_seconds", pa.float32()),
    ("match_seconds", pa.float32()),
    ("score_diff", pa.int8()),
    ("x", pa.float32()), ("y", pa.float32()),
    ("end_x", pa.float32()), ("end_y", pa.float32()),
    ("pass_length", pa.float32()),
    ("pass_angle", pa.float32()),
    ("pass_height", pa.string()),
    ("pass_body_part", pa.string()),
    ("pass_type", pa.string()),
    ("play_pattern", pa.string()),
    ("pass_outcome_raw", pa.string()),
    ("pass_success", pa.bool_()),
    ("recipient_id", pa.int32()),
    ("is_set_piece_restart", pa.bool_()),
    ("zone", pa.string()),
    ("dist_to_goal", pa.float32()),
    ("prog_dist", pa.float32()),
    ("under_pressure", pa.bool_()),
    ("counterpress", pa.bool_()),
    ("n_pressure_linked", pa.int8()),
    ("presser_id", pa.int32()),
    ("presser_x", pa.float32()), ("presser_y", pa.float32()),
    ("presser_dist", pa.float32()),
    ("pressure_lead_s", pa.float32()),
    ("presser_counterpress", pa.bool_()),
    ("receipt_under_pressure", pa.bool_()),
    ("time_since_receipt_s", pa.float32()),
    ("player_pressed_earlier_in_poss", pa.bool_()),
    # history
    ("pass_ord_in_poss", pa.int16()),
    ("pass_ord_in_seg", pa.int16()),
    ("up_lag1", pa.bool_()), ("up_lag2", pa.bool_()),
    ("presser_dist_lag1", pa.float32()),
    ("press_run_len", pa.int16()),
    ("passes_since_press_onset", pa.int16()),
    ("passes_since_last_press", pa.int16()),
    ("time_since_last_press_s", pa.float32()),
    ("poss_press_count", pa.int16()),
    ("poss_press_frac", pa.float32()),
    # falsification support
    ("up_lead1", pa.bool_()),
    ("poss_ends_at_t", pa.bool_()),
    ("presser_id_lag1", pa.int32()),
    ("presser_involved_at_t", pa.bool_()),
    ("lag1_presser_dist_to_t", pa.float32()),
    ("zone_lag1", pa.string()),
    # press runs
    ("press_run_id", pa.string()),
    ("press_run_is_last", pa.bool_()),
    ("press_run_exit", pa.string()),
    # tier 2
    ("ff_available", pa.bool_()),
    ("ff_n_opp_visible", pa.int8()),
    ("ff_n_team_visible", pa.int8()),
    ("ff_visible_r3", pa.bool_()),
    ("ff_visible_r5", pa.bool_()),
    ("ff_recv_visible_r5", pa.bool_()),
    ("ff_nearest_opp_dist", pa.float32()),
    ("ff_opp_within_3", pa.int8()),
    ("ff_opp_within_5", pa.int8()),
    ("ff_lane_opp", pa.int8()),
    ("ff_lane_visible", pa.bool_()),
    ("ff_recv_opp_within_5", pa.int8()),
])
COLUMNS = [f.name for f in SCHEMA]


def load_match_index(tier2_only: bool) -> list[tuple]:
    comps = {
        (c["competition_id"], c["season_id"]): c
        for c in json.loads((RAW / "competitions.json").read_text())
    }
    tasks = []
    for mf in sorted((RAW / "matches").rglob("*.json")):
        cid, sid = int(mf.parent.name), int(mf.stem)
        info = comps.get((cid, sid), {})
        for m in json.loads(mf.read_text()):
            mid = m["match_id"]
            ev = RAW / "events" / f"{mid}.json"
            if not ev.exists():
                continue
            ff = RAW / "three-sixty" / f"{mid}.json"
            ff = ff if ff.exists() else None
            if tier2_only and ff is None:
                continue
            tasks.append((
                {
                    "match_id": mid,
                    "competition_id": cid,
                    "season_id": sid,
                    "comp_season_uid": f"{cid}:{sid}",
                    "competition_name": info.get("competition_name"),
                    "season_name": info.get("season_name"),
                    "match_date": m.get("match_date"),
                    "home_team_id": m["home_team"]["home_team_id"],
                    "away_team_id": m["away_team"]["away_team_id"],
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                },
                ev,
                ff,
            ))
    return tasks


def flush(writer, buf: list[dict]) -> int:
    df = pd.DataFrame(buf, columns=COLUMNS)
    table = pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False)
    writer.write_table(table)
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build data/processed/passes.parquet")
    ap.add_argument("--limit", type=int, default=None, help="first N matches only")
    ap.add_argument("--tier2-only", action="store_true", help="only 360 matches")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--batch", type=int, default=150, help="matches per row group")
    args = ap.parse_args()

    if not (RAW / "competitions.json").exists():
        sys.exit("[build] no raw data found. Run `python fetch.py` first.")

    tasks = load_match_index(args.tier2_only)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[build] {len(tasks)} matches, {args.workers} workers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(OUT_PATH, SCHEMA, compression="zstd")

    buf: list[dict] = []
    n_rows = 0
    done = 0
    errors: list[str] = []
    score_bad: list[dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(parse_match, t): t[0]["match_id"] for t in tasks}
        for fut in as_completed(futs):
            rows, chk, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            if chk and not chk["ok"]:
                score_bad.append(chk)
            buf.extend(rows)
            if len(buf) >= args.batch * 900:
                n_rows += flush(writer, buf)
                buf = []
            if done % 250 == 0 or done == len(tasks):
                el = time.time() - t0
                print(
                    f"[build] {done}/{len(tasks)} matches  {n_rows + len(buf):,} passes"
                    f"  {el:6.1f}s"
                )

    if buf:
        n_rows += flush(writer, buf)
    writer.close()

    try:
        shown = OUT_PATH.relative_to(ROOT)
    except ValueError:
        shown = OUT_PATH
    print(f"\n[build] wrote {n_rows:,} passes -> {shown}")
    print(f"[build] file size {OUT_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"[build] elapsed {time.time() - t0:.1f}s")

    if errors:
        print(f"\n[build] {len(errors)} match(es) failed to parse:")
        for e in errors[:10]:
            print(f"          {e}")
    if score_bad:
        print(f"\n[build] score reconstruction mismatch in {len(score_bad)} match(es):")
        for s in score_bad[:10]:
            print(
                f"          match {s['match_id']}: "
                f"rebuilt {s['recon_home']}-{s['recon_away']} vs "
                f"index {s['index_home']}-{s['index_away']}"
            )
    else:
        print("[build] score reconstruction matches the match index in every match")


if __name__ == "__main__":
    main()
