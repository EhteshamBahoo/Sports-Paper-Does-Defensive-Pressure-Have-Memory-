#!/usr/bin/env python3
"""
build.py -- parse StatsBomb raw JSON into a spine and an estimation table.

    data/raw/open-data/data/{events,matches,three-sixty}/*.json
        -> data/processed/spine.parquet     one row per pressed-eligible ball event
        -> data/processed/passes.parquet    one row per pass (the estimation table)

Every downstream stage reads those two Parquet files via `src.load`. No analysis
step re-opens the raw event JSON. Run `python fetch.py` first.

Why a spine
-----------
Pressure exposure does not happen only on passes. In this corpus there are 265.9
under-pressure carries per match against 152.8 under-pressure passes. A clock
counted over passes alone therefore records "no press" for a carrier who was in
fact pressed between pass t-1 and pass t. That is non-classical measurement error
in the exposure variable itself, not merely a gap in the hazard model.

So the pressure clock, press-run construction and exit taxonomy all live on the
spine, over ball events. Passes remain the only outcome rows: the estimation
table is still one row per pass, joining its history from the spine.

Spine membership, and why
-------------------------
Measured under_pressure rate by event type, over 150 matches of raw events:

    Pass 16.0%   Carry 35.8%   Miscontrol 27.9%   Shot 25.9%
    Dribble 100.0%   Duel 100.0%   Dispossessed 100.0%   Clearance 100.0%

That split decides the roles. The four types at exactly 100.0% carry no
information: StatsBomb sets `under_pressure` on them by definition, not by
observation.

role "clock" -- advances the pressure clock, and the only role runs are built over:
    Pass      the outcome unit, and genuinely variable (16.0%)
    Carry     the dominant locus of exposure, genuinely variable (35.8%);
              1.74 pressed carries for every pressed pass

role "definitional" -- in the spine, never on the clock:
    Dribble, Duel, Dispossessed, Clearance
              all 100.0% under_pressure. Counting them would make the clock
              tautological: every dribble would reset "events since last press"
              whether or not any new pressure occurred. They are retained because
              they are real ball events that terminate runs and supply exits.
              Flagged `press_definitional`.

role "terminal" -- measured, but outcome- or possession-determined:
    Miscontrol  an outcome partly *caused by* the pressure being measured;
                counting it would condition the exposure clock on the outcome
    Shot        possession-ending

EXCLUDED entirely:
    Ball Receipt*  near-simultaneous with the following Carry, so it would double
                   count the same instant of pressure. Receipt pressure is already
                   carried on the pass as `receipt_under_pressure`.
    Pressure       a defending-team event, not an on-ball action by the carrier.

Reset boundaries
----------------
possession_uid = match_id : period : possession
    StatsBomb `possession` ids can span a half boundary (~1 per match), so the
    period is part of the key.

segment_uid = possession_uid # k
    k increments at every set-piece restart pass. All pressure history resets
    here: a stoppage dissolves the press, so history must not carry across it.

Coordinate frames
-----------------
StatsBomb logs every event in the ACTING team's attacking frame -- the acting
team always attacks toward x=120, in every period. A `Pressure` event is
performed by the defending team, so its coordinates are rotated 180 degrees
relative to the pass it acts on and must be mirrored before any distance is
taken. See `mirror()`. Run `python validate.py` to re-check this and the other
physical-plausibility invariants.

Usage
-----
    python build.py                  # full corpus
    python build.py --limit 200      # first 200 matches, smoke test
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
OUT_PASSES = OUT_DIR / "passes.parquet"
OUT_SPINE = OUT_DIR / "spine.parquet"

GOAL = (120.0, 40.0)                 # StatsBomb pitch is 120 x 80
LANE_HALF_WIDTH = 3.0
VIS_MARGIN = (3.0, 5.0)

SET_PIECE_TYPES = {"Throw-in", "Corner", "Free Kick", "Goal Kick", "Kick Off"}
FAIL_OUTCOMES = {"Incomplete", "Pass Offside"}
NULL_OUTCOMES = {"Unknown", "Injury Clearance"}

# Spine roles. Measured under_pressure rates, from 150 matches of raw events:
#   Pass 16.0%   Carry 35.8%   Miscontrol 27.9%   Shot 25.9%
#   Dribble 100.0%   Duel 100.0%   Dispossessed 100.0%   Clearance 100.0%
#
# The 100.0% types carry no information: StatsBomb sets under_pressure on them by
# definition, not by observation. Counting them on the exposure clock would make
# the clock partly tautological -- every dribble would reset "time since press"
# whether or not any new pressure occurred. They stay in the spine because they
# are real ball events that terminate runs, but they never advance the clock.
SPINE_CLOCK_TYPES = {"Pass", "Carry"}
SPINE_DEFINITIONAL_TYPES = {"Dribble", "Duel", "Dispossessed", "Clearance"}
SPINE_TERMINAL_TYPES = {"Miscontrol", "Shot"}
SPINE_TYPES = SPINE_CLOCK_TYPES | SPINE_DEFINITIONAL_TYPES | SPINE_TERMINAL_TYPES


def spine_role_of(etype: str) -> str:
    if etype in SPINE_CLOCK_TYPES:
        return "clock"
    if etype in SPINE_DEFINITIONAL_TYPES:
        return "definitional"
    return "terminal"

BALL_RELEASE_TYPES = {
    "Shot", "Clearance", "Miscontrol", "Dispossessed", "Foul Committed", "Error",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def ts_seconds(stamp: str) -> float:
    h, m, s = stamp.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def mirror(x: float, y: float) -> tuple[float, float]:
    """Rotate a location 180 degrees into the opposing team's attacking frame.

    StatsBomb logs every event in the frame of the team performing it. A
    `Pressure` event is performed by the DEFENDING team, so its coordinates are
    rotated relative to the pass it acts on. Verified three ways: mirrored
    carrier-to-presser separation has median 3.8 m against 68.2 m raw; the median
    is period-invariant (3.81/3.82/3.94/3.92 m across periods 1-4), so the
    halftime end switch is already normalised out upstream; and the mirrored
    presser lies a median 2.68 m from the nearest opponent in the independently
    sourced 360 freeze frame, against 41.39 m raw.
    """
    return 120.0 - x, 80.0 - y


def zone_of(x: float, y: float) -> str:
    third = "def" if x < 40 else ("mid" if x < 80 else "att")
    band = "y0" if y < 80 / 3 else ("y1" if y < 160 / 3 else "y2")
    return f"{third}_{band}"


def dist_to_goal(x: float, y: float) -> float:
    return math.hypot(GOAL[0] - x, GOAL[1] - y)


def poly_from_visible_area(va) -> np.ndarray:
    pts = np.asarray(va, dtype=float).reshape(-1, 2)
    if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    return pts


def points_inside(P: np.ndarray, V: np.ndarray) -> np.ndarray:
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
    best = np.full(len(P), np.inf)
    n = len(V)
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        ab = b - a
        L = float(ab @ ab)
        if L == 0.0:
            d = np.linalg.norm(P - a, axis=1)
        else:
            t = np.clip(((P - a) @ ab) / L, 0.0, 1.0)
            d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        best = np.minimum(best, d)
    return best


def link_pressure(e: dict, by_id: dict, team_id: int, x: float, y: float,
                  t_ev: float) -> dict:
    """Attach the pressing defender via related_events, in the ACTOR's frame."""
    out = {
        "n_pressure_linked": 0, "presser_id": None, "presser_x": None,
        "presser_y": None, "presser_dist": None, "pressure_lead_s": None,
        "presser_counterpress": None,
    }
    pressures = [
        by_id[r] for r in e.get("related_events", [])
        if r in by_id and by_id[r]["type"]["name"] == "Pressure"
    ]
    out["n_pressure_linked"] = len(pressures)
    if not pressures:
        return out

    best = None
    for pr in pressures:
        pl = pr.get("location")
        if pl:
            px, py = float(pl[0]), float(pl[1])
            if pr["team"]["id"] != team_id:
                px, py = mirror(px, py)
            d = math.hypot(px - x, py - y)
        else:
            px = py = None
            d = math.inf
        if best is None or d < best[0]:
            best = (d, pr, px, py)

    d, pr, px, py = best
    if pr.get("player"):
        out["presser_id"] = pr["player"]["id"]
    if px is not None:
        out["presser_x"], out["presser_y"] = px, py
        out["presser_dist"] = float(d)
    out["pressure_lead_s"] = t_ev - ts_seconds(pr["timestamp"])
    out["presser_counterpress"] = bool(pr.get("counterpress", False))
    return out


# --------------------------------------------------------------------------- #
# per-match parse
# --------------------------------------------------------------------------- #
def parse_match(task: tuple) -> tuple:
    match_meta, ev_path, ff_path = task
    try:
        with open(ev_path, "rb") as fh:
            events = json.load(fh)
    except Exception as exc:                                    # noqa: BLE001
        return [], [], None, f"{match_meta['match_id']}: {exc}"

    events.sort(key=lambda e: e["index"])
    by_id = {e["id"]: e for e in events}

    frames = {}
    if ff_path is not None:
        try:
            with open(ff_path, "rb") as fh:
                frames = {f["event_uuid"]: f for f in json.load(fh)}
        except Exception:                                       # noqa: BLE001
            frames = {}

    home_id = match_meta["home_team_id"]
    away_id = match_meta["away_team_id"]

    # ---- running score, own goals included -------------------------------- #
    # StatsBomb logs an own goal twice: "Own Goal Against" for the team that put
    # it in and "Own Goal For" for the beneficiary. Counting only "Own Goal For"
    # avoids double counting. Shootouts (period 5) never count.
    goals = {home_id: 0, away_id: 0}
    score_before: dict[str, tuple[int, int]] = {}
    for e in events:
        score_before[e["id"]] = (goals[home_id], goals[away_id])
        if e["period"] == 5:
            continue
        etype, tid = e["type"]["name"], e["team"]["id"]
        if etype == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            goals[tid] = goals.get(tid, 0) + 1
        elif etype == "Own Goal For":
            goals[tid] = goals.get(tid, 0) + 1

    score_check = {
        "match_id": match_meta["match_id"],
        "recon_home": goals[home_id], "recon_away": goals[away_id],
        "index_home": match_meta["home_score"], "index_away": match_meta["away_score"],
    }
    score_check["ok"] = (
        score_check["recon_home"] == score_check["index_home"]
        and score_check["recon_away"] == score_check["index_away"]
    )

    # ---- period offsets for a monotone match clock ------------------------ #
    period_offsets, acc = {}, 0.0
    for p in sorted({e["period"] for e in events if e["period"] != 5}):
        period_offsets[p] = acc
        acc += max((ts_seconds(e["timestamp"]) for e in events if e["period"] == p),
                   default=0.0)

    # ---- single streaming pass over the event list ------------------------ #
    passes: list[dict] = []
    spine: list[dict] = []
    hold: tuple[str, int, float] | None = None      # (puid, player, receipt time)
    seg_counter: dict[str, int] = {}

    for e in events:
        period = e["period"]
        if period == 5:
            continue
        etype = e["type"]["name"]
        puid = f"{match_meta['match_id']}:{period}:{e['possession']}"
        t_ev = ts_seconds(e["timestamp"])
        t_match = period_offsets.get(period, 0.0) + t_ev

        if etype == "Ball Receipt*" and e.get("player"):
            hold = (puid, e["player"]["id"], t_ev)
        elif etype in BALL_RELEASE_TYPES:
            hold = None

        if etype not in SPINE_TYPES:
            continue

        loc = e.get("location")
        if not loc:
            continue

        team_id = e["team"]["id"]
        poss_team_id = e["possession_team"]["id"]
        is_poss_team = team_id == poss_team_id
        pid = e["player"]["id"] if e.get("player") else None
        x, y = float(loc[0]), float(loc[1])

        # segment id: bumped at every set-piece restart pass
        is_restart = False
        if etype == "Pass":
            ptype = e["pass"].get("type", {}).get("name", "OPEN_PLAY")
            is_restart = ptype in SET_PIECE_TYPES
            if is_restart:
                seg_counter[puid] = seg_counter.get(puid, 0) + 1
        seg_uid = f"{puid}#{seg_counter.get(puid, 0)}"

        press = link_pressure(e, by_id, team_id, x, y, t_ev)

        # end location, where the event type has one
        end_x = end_y = None
        outcome_raw = None
        if etype == "Pass":
            el = e["pass"].get("end_location")
            if el:
                end_x, end_y = float(el[0]), float(el[1])
            outcome_raw = e["pass"].get("outcome", {}).get("name", "COMPLETE")
        elif etype == "Carry":
            el = e.get("carry", {}).get("end_location")
            if el:
                end_x, end_y = float(el[0]), float(el[1])
        elif etype == "Dribble":
            outcome_raw = e.get("dribble", {}).get("outcome", {}).get("name")
        elif etype == "Duel":
            outcome_raw = e.get("duel", {}).get("outcome", {}).get("name")

        presser_dist_to_end = None
        if press["presser_x"] is not None and end_x is not None:
            presser_dist_to_end = math.hypot(
                press["presser_x"] - end_x, press["presser_y"] - end_y
            )

        # ---- spine row (possession-team ball events only) ----------------- #
        if is_poss_team:
            spine.append({
                "event_id": e["id"],
                "match_id": match_meta["match_id"],
                "event_index": e["index"],
                "period": period,
                "possession": e["possession"],
                "possession_uid": puid,
                "segment_uid": seg_uid,
                "comp_season_uid": match_meta["comp_season_uid"],
                "team_id": team_id,
                "player_id": pid,
                "event_type": etype,
                "spine_role": spine_role_of(etype),
                "press_definitional": etype in SPINE_DEFINITIONAL_TYPES,
                "outcome_raw": outcome_raw,
                "is_set_piece_restart": is_restart,
                "under_pressure": bool(e.get("under_pressure", False)),
                "counterpress": bool(e.get("counterpress", False)),
                "x": x, "y": y, "end_x": end_x, "end_y": end_y,
                "match_seconds": t_match,
                "prog_dist": (dist_to_goal(x, y) - dist_to_goal(end_x, end_y)
                              if end_x is not None else None),
                "presser_id": press["presser_id"],
                "presser_x": press["presser_x"], "presser_y": press["presser_y"],
                "presser_dist": press["presser_dist"],
                "presser_dist_to_end": presser_dist_to_end,
                "pressure_lead_s": press["pressure_lead_s"],
            })

        # ---- pass row ------------------------------------------------------ #
        if etype != "Pass" or end_x is None:
            continue

        pa_ = e["pass"]
        if outcome_raw == "COMPLETE":
            success = True
        elif outcome_raw in FAIL_OUTCOMES or outcome_raw == "Out":
            success = False
        else:
            success = None

        rel = [by_id[r] for r in e.get("related_events", []) if r in by_id]
        receipts = [r for r in rel if r["type"]["name"] == "Ball Receipt*"]
        receipt_up = bool(receipts[0].get("under_pressure", False)) if receipts else None

        time_since_receipt = None
        if hold is not None and pid is not None and hold[0] == puid and hold[1] == pid:
            gap = t_ev - hold[2]
            time_since_receipt = gap if gap >= 0 else None
        hold = None                      # the ball has now left the passer

        h, a = score_before[e["id"]]
        passes.append({
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
            "is_possession_team": is_poss_team,
            "opponent_team_id": away_id if team_id == home_id else home_id,
            "player_id": pid,
            "is_home": team_id == home_id,
            "period_seconds": t_ev,
            "match_seconds": t_match,
            "score_diff": (h - a) if team_id == home_id else (a - h),
            "x": x, "y": y, "end_x": end_x, "end_y": end_y,
            "pass_length": float(pa_["length"]) if pa_.get("length") is not None else None,
            "pass_angle": float(pa_["angle"]) if pa_.get("angle") is not None else None,
            "pass_height": pa_.get("height", {}).get("name"),
            "pass_body_part": pa_.get("body_part", {}).get("name"),
            "pass_type": pa_.get("type", {}).get("name", "OPEN_PLAY"),
            "play_pattern": e["play_pattern"]["name"],
            "pass_outcome_raw": outcome_raw,
            "pass_success": success,
            "recipient_id": pa_.get("recipient", {}).get("id"),
            "is_set_piece_restart": is_restart,
            "zone": zone_of(x, y),
            "dist_to_goal": dist_to_goal(x, y),
            "prog_dist": dist_to_goal(x, y) - dist_to_goal(end_x, end_y),
            "under_pressure": bool(e.get("under_pressure", False)),
            "counterpress": bool(e.get("counterpress", False)),
            "n_pressure_linked": press["n_pressure_linked"],
            "presser_id": press["presser_id"],
            "presser_x": press["presser_x"], "presser_y": press["presser_y"],
            "presser_dist": press["presser_dist"],
            "presser_dist_to_end": presser_dist_to_end,
            "pressure_lead_s": press["pressure_lead_s"],
            "presser_counterpress": press["presser_counterpress"],
            "receipt_under_pressure": receipt_up,
            "time_since_receipt_s": time_since_receipt,
        })

    if not passes and not spine:
        return [], [], score_check, None

    _add_player_pressure_history(passes, events, match_meta["match_id"])
    _add_spine_history(spine)
    _add_spine_runs(spine, events)
    _add_pass_history(passes)
    _join_spine_to_passes(passes, spine)
    _add_360(passes, frames)
    return passes, spine, score_check, None


def _add_player_pressure_history(passes: list[dict], events: list[dict],
                                 match_id: int) -> None:
    """player_pressed_earlier_in_poss, with strictly-before semantics."""
    row_of = {r["event_id"]: r for r in passes}
    seen: dict[tuple[str, int], bool] = {}
    for e in events:
        if e["period"] == 5:
            continue
        puid = f"{match_id}:{e['period']}:{e['possession']}"
        r = row_of.get(e["id"])
        if r is not None:
            r["player_pressed_earlier_in_poss"] = bool(
                seen.get((puid, r["player_id"]), False)
            )
        if e.get("under_pressure") and e.get("player"):
            seen[(puid, e["player"]["id"])] = True
    for r in passes:
        r.setdefault("player_pressed_earlier_in_poss", None)


# --------------------------------------------------------------------------- #
# spine history: the exposure clock
# --------------------------------------------------------------------------- #
SPINE_HISTORY_COLS = (
    "spine_ord_in_seg", "exposure_ord_in_seg", "up_lag1_spine", "up_lag2_spine",
    "lag1_spine_type", "lag1_presser_dist_spine", "lag1_pressure_lead_s_spine",
    "press_run_len_spine", "events_since_press_onset", "events_since_last_press",
    "time_since_last_press_spine", "seg_press_count_spine", "seg_press_frac_spine",
    "press_run_id_spine", "press_run_is_last_spine", "press_run_exit_spine",
    "lag1_presser_id_spine", "lag1_presser_dist_to_t_spine",
)


def _add_spine_history(spine: list[dict]) -> None:
    for r in spine:
        for c in SPINE_HISTORY_COLS:
            r.setdefault(c, None)

    by_seg: dict[str, list[dict]] = {}
    for r in spine:
        by_seg.setdefault(r["segment_uid"], []).append(r)

    for seq in by_seg.values():
        seq.sort(key=lambda r: r["event_index"])
        for i, r in enumerate(seq):
            r["spine_ord_in_seg"] = i

        # the clock advances over role=="clock" events only (Pass, Carry):
        # the other roles have a definitional or outcome-determined pressure flag
        expo = [r for r in seq if r["spine_role"] == "clock"]
        run_len = press_count = 0
        last_press_i = last_press_t = None
        for i, r in enumerate(expo):
            prev = expo[i - 1] if i >= 1 else None
            prev2 = expo[i - 2] if i >= 2 else None

            r["exposure_ord_in_seg"] = i
            r["up_lag1_spine"] = prev["under_pressure"] if prev else None
            r["up_lag2_spine"] = prev2["under_pressure"] if prev2 else None
            if prev:
                r["lag1_spine_type"] = prev["event_type"]
                r["lag1_presser_dist_spine"] = prev["presser_dist"]
                r["lag1_pressure_lead_s_spine"] = prev["pressure_lead_s"]
                r["lag1_presser_id_spine"] = prev["presser_id"]
                if prev["presser_x"] is not None:
                    r["lag1_presser_dist_to_t_spine"] = math.hypot(
                        prev["presser_x"] - r["x"], prev["presser_y"] - r["y"]
                    )

            r["seg_press_count_spine"] = press_count
            r["seg_press_frac_spine"] = (press_count / i) if i > 0 else None
            if last_press_i is None:
                r["events_since_last_press"] = None
                r["time_since_last_press_spine"] = None
            else:
                r["events_since_last_press"] = i - last_press_i
                r["time_since_last_press_spine"] = r["match_seconds"] - last_press_t

            if r["under_pressure"]:
                run_len += 1
                r["press_run_len_spine"] = run_len
                r["events_since_press_onset"] = run_len - 1
                r["events_since_last_press"] = 0
                r["time_since_last_press_spine"] = 0.0
                press_count += 1
                last_press_i, last_press_t = i, r["match_seconds"]
            else:
                run_len = 0
                r["press_run_len_spine"] = 0
                r["events_since_press_onset"] = None


def _add_spine_runs(spine: list[dict], events: list[dict]) -> None:
    """Label press runs over exposure events and classify how each one ended."""
    idx_of = {e["index"]: i for i, e in enumerate(events)}
    by_seg: dict[str, list[dict]] = {}
    for r in spine:
        if r["spine_role"] == "clock":
            by_seg.setdefault(r["segment_uid"], []).append(r)

    for seg, seq in by_seg.items():
        seq.sort(key=lambda r: r["event_index"])
        run_no = i = 0
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
                seq[k]["press_run_id_spine"] = rid
                seq[k]["press_run_is_last_spine"] = k == j
            seq[j]["press_run_exit_spine"] = _classify_exit(seq[j], events, idx_of)
            i = j + 1


def _classify_exit(r: dict, events: list[dict], idx_of: dict) -> str:
    et, oc = r["event_type"], r["outcome_raw"]
    if et == "Pass":
        if oc == "Out":
            return "stoppage_out"
        if oc in FAIL_OUTCOMES:
            return "turnover"
        if oc in NULL_OUTCOMES:
            return "other"
    elif et == "Dribble" and oc and oc != "Complete":
        return "turnover"
    elif et == "Duel" and oc and "Lost" in oc:
        return "turnover"

    start = idx_of.get(r["event_index"])
    if start is None:
        return "other"
    team, poss, period = r["team_id"], r["possession"], r["period"]

    for e in events[start + 1:]:
        if e["period"] != period or e["possession"] != poss:
            break                                    # possession ended
        etype, tid = e["type"]["name"], e["team"]["id"]
        if etype == "Foul Won" and tid == team:
            return "stoppage_foul"
        if etype == "Foul Committed":
            return "stoppage_foul" if tid != team else "turnover"
        if etype == "Injury Stoppage":
            return "stoppage_foul"
        if etype == "Half End":
            return "period_end"
        if tid != team:
            continue
        if etype == "Shot":
            return "shot"
        if etype in ("Miscontrol", "Dispossessed"):
            return "turnover"
        if etype == "Clearance":
            return "clearance"
        if etype == "Dribble":
            oc2 = e.get("dribble", {}).get("outcome", {}).get("name")
            if oc2 and oc2 != "Complete":
                return "turnover"
            continue                     # definitional pressure: not an escape
        if etype == "Duel":
            oc2 = e.get("duel", {}).get("outcome", {}).get("name")
            if oc2 and "Lost" in oc2:
                return "turnover"
            continue
        if etype in SPINE_CLOCK_TYPES:
            # the run ended, so this next clock event is unpressed unless a
            # set-piece restart opened a new segment underneath us
            return "escape" if not e.get("under_pressure") else "segment_break"
    return "turnover"


def _join_spine_to_passes(passes: list[dict], spine: list[dict]) -> None:
    by_id = {r["event_id"]: r for r in spine}
    for p in passes:
        s = by_id.get(p["event_id"])
        for c in SPINE_HISTORY_COLS:
            p[c] = s.get(c) if s is not None else None


# --------------------------------------------------------------------------- #
# pass-level clock, retained so the two clocks can be compared directly
# --------------------------------------------------------------------------- #
PASS_HISTORY_COLS = (
    "pass_ord_in_poss", "pass_ord_in_seg", "up_lag1", "up_lag2",
    "presser_dist_lag1", "lag1_pressure_lead_s", "press_run_len",
    "passes_since_press_onset", "passes_since_last_press", "time_since_last_press_s",
    "poss_press_count", "poss_press_frac", "up_lead1", "poss_ends_at_t",
    "presser_id_lag1", "presser_involved_at_t", "lag1_presser_dist_to_t", "zone_lag1",
)


def _add_pass_history(passes: list[dict]) -> None:
    for r in passes:
        for c in PASS_HISTORY_COLS:
            r.setdefault(c, None)

    by_poss: dict[str, list[dict]] = {}
    by_seg: dict[str, list[dict]] = {}
    for r in passes:
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
        run_len = press_count = 0
        last_press_i = last_press_t = None
        for i, r in enumerate(seq):
            prev = seq[i - 1] if i >= 1 else None
            prev2 = seq[i - 2] if i >= 2 else None
            nxt = seq[i + 1] if i + 1 < len(seq) else None

            r["pass_ord_in_seg"] = i
            r["up_lag1"] = prev["under_pressure"] if prev else None
            r["up_lag2"] = prev2["under_pressure"] if prev2 else None
            r["presser_dist_lag1"] = prev["presser_dist"] if prev else None
            r["lag1_pressure_lead_s"] = prev["pressure_lead_s"] if prev else None
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
                last_press_i, last_press_t = i, r["match_seconds"]
            else:
                run_len = 0
                r["press_run_len"] = 0
                r["passes_since_press_onset"] = None


# --------------------------------------------------------------------------- #
# tier 2 geometry
# --------------------------------------------------------------------------- #
FF_COLS = (
    "ff_available", "ff_n_opp_visible", "ff_n_team_visible", "ff_visible_r3",
    "ff_visible_r5", "ff_recv_visible_r5", "ff_nearest_opp_dist",
    "ff_opp_within_3", "ff_opp_within_5", "ff_lane_opp", "ff_lane_visible",
    "ff_recv_opp_within_5",
)


def _add_360(rows: list[dict], frames: dict) -> None:
    for r in rows:
        for c in FF_COLS:
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
            r["ff_recv_opp_within_5"] = int(
                (np.linalg.norm(opp - target, axis=1) <= 5.0).sum()
            )
            seg = target - origin
            L = float(seg @ seg)
            if L > 0:
                t = np.clip(((opp - origin) @ seg) / L, 0.0, 1.0)
                perp = np.linalg.norm(opp - (origin + t[:, None] * seg), axis=1)
                r["ff_lane_opp"] = int(
                    ((perp <= LANE_HALF_WIDTH) & (t > 0.0) & (t < 1.0)).sum()
                )
            else:
                r["ff_lane_opp"] = 0
        else:
            r["ff_opp_within_3"] = r["ff_opp_within_5"] = 0
            r["ff_recv_opp_within_5"] = r["ff_lane_opp"] = 0

        va = fr.get("visible_area")
        if not va or len(va) < 6:
            continue
        V = poly_from_visible_area(va)
        if len(V) < 3:
            continue

        ts = np.linspace(0.0, 1.0, 9)
        probe = np.vstack([origin, target, origin + (target - origin) * ts[:, None]])
        ins = points_inside(probe, V)
        edge = points_edge_dist(probe, V)
        r["ff_visible_r3"] = bool(ins[0] and edge[0] >= VIS_MARGIN[0])
        r["ff_visible_r5"] = bool(ins[0] and edge[0] >= VIS_MARGIN[1])
        r["ff_recv_visible_r5"] = bool(ins[1] and edge[1] >= VIS_MARGIN[1])
        r["ff_lane_visible"] = bool((ins[2:] & (edge[2:] >= LANE_HALF_WIDTH)).all())


# --------------------------------------------------------------------------- #
# schemas
# --------------------------------------------------------------------------- #
SPINE_SCHEMA = pa.schema([
    ("event_id", pa.string()), ("match_id", pa.int32()),
    ("event_index", pa.int32()), ("period", pa.int8()),
    ("possession", pa.int16()), ("possession_uid", pa.string()),
    ("segment_uid", pa.string()), ("comp_season_uid", pa.string()),
    ("team_id", pa.int32()), ("player_id", pa.int32()),
    ("event_type", pa.string()), ("spine_role", pa.string()),
    ("press_definitional", pa.bool_()),
    ("outcome_raw", pa.string()), ("is_set_piece_restart", pa.bool_()),
    ("under_pressure", pa.bool_()), ("counterpress", pa.bool_()),
    ("x", pa.float32()), ("y", pa.float32()),
    ("end_x", pa.float32()), ("end_y", pa.float32()),
    ("match_seconds", pa.float32()), ("prog_dist", pa.float32()),
    ("presser_id", pa.int32()), ("presser_x", pa.float32()),
    ("presser_y", pa.float32()), ("presser_dist", pa.float32()),
    ("presser_dist_to_end", pa.float32()), ("pressure_lead_s", pa.float32()),
    ("spine_ord_in_seg", pa.int16()), ("exposure_ord_in_seg", pa.int16()),
    ("up_lag1_spine", pa.bool_()), ("up_lag2_spine", pa.bool_()),
    ("lag1_spine_type", pa.string()), ("lag1_presser_dist_spine", pa.float32()),
    ("lag1_pressure_lead_s_spine", pa.float32()),
    ("lag1_presser_id_spine", pa.int32()),
    ("lag1_presser_dist_to_t_spine", pa.float32()),
    ("press_run_len_spine", pa.int16()),
    ("events_since_press_onset", pa.int16()),
    ("events_since_last_press", pa.int16()),
    ("time_since_last_press_spine", pa.float32()),
    ("seg_press_count_spine", pa.int16()),
    ("seg_press_frac_spine", pa.float32()),
    ("press_run_id_spine", pa.string()),
    ("press_run_is_last_spine", pa.bool_()),
    ("press_run_exit_spine", pa.string()),
])

SCHEMA = pa.schema([
    ("event_id", pa.string()), ("match_id", pa.int32()),
    ("event_index", pa.int32()), ("period", pa.int8()),
    ("possession", pa.int16()), ("possession_uid", pa.string()),
    ("segment_uid", pa.string()), ("competition_id", pa.int16()),
    ("season_id", pa.int16()), ("comp_season_uid", pa.string()),
    ("competition_name", pa.string()), ("season_name", pa.string()),
    ("match_date", pa.string()), ("team_id", pa.int32()),
    ("possession_team_id", pa.int32()), ("is_possession_team", pa.bool_()),
    ("opponent_team_id", pa.int32()), ("player_id", pa.int32()),
    ("is_home", pa.bool_()), ("period_seconds", pa.float32()),
    ("match_seconds", pa.float32()), ("score_diff", pa.int8()),
    ("x", pa.float32()), ("y", pa.float32()),
    ("end_x", pa.float32()), ("end_y", pa.float32()),
    ("pass_length", pa.float32()), ("pass_angle", pa.float32()),
    ("pass_height", pa.string()), ("pass_body_part", pa.string()),
    ("pass_type", pa.string()), ("play_pattern", pa.string()),
    ("pass_outcome_raw", pa.string()), ("pass_success", pa.bool_()),
    ("recipient_id", pa.int32()), ("is_set_piece_restart", pa.bool_()),
    ("zone", pa.string()), ("dist_to_goal", pa.float32()),
    ("prog_dist", pa.float32()), ("under_pressure", pa.bool_()),
    ("counterpress", pa.bool_()), ("n_pressure_linked", pa.int8()),
    ("presser_id", pa.int32()), ("presser_x", pa.float32()),
    ("presser_y", pa.float32()), ("presser_dist", pa.float32()),
    ("presser_dist_to_end", pa.float32()), ("pressure_lead_s", pa.float32()),
    ("presser_counterpress", pa.bool_()), ("receipt_under_pressure", pa.bool_()),
    ("time_since_receipt_s", pa.float32()),
    ("player_pressed_earlier_in_poss", pa.bool_()),
    # pass-level clock
    ("pass_ord_in_poss", pa.int16()), ("pass_ord_in_seg", pa.int16()),
    ("up_lag1", pa.bool_()), ("up_lag2", pa.bool_()),
    ("presser_dist_lag1", pa.float32()), ("lag1_pressure_lead_s", pa.float32()),
    ("press_run_len", pa.int16()), ("passes_since_press_onset", pa.int16()),
    ("passes_since_last_press", pa.int16()),
    ("time_since_last_press_s", pa.float32()),
    ("poss_press_count", pa.int16()), ("poss_press_frac", pa.float32()),
    ("up_lead1", pa.bool_()), ("poss_ends_at_t", pa.bool_()),
    ("presser_id_lag1", pa.int32()), ("presser_involved_at_t", pa.bool_()),
    ("lag1_presser_dist_to_t", pa.float32()), ("zone_lag1", pa.string()),
    # spine clock
    ("spine_ord_in_seg", pa.int16()), ("exposure_ord_in_seg", pa.int16()),
    ("up_lag1_spine", pa.bool_()), ("up_lag2_spine", pa.bool_()),
    ("lag1_spine_type", pa.string()), ("lag1_presser_dist_spine", pa.float32()),
    ("lag1_pressure_lead_s_spine", pa.float32()),
    ("lag1_presser_id_spine", pa.int32()),
    ("lag1_presser_dist_to_t_spine", pa.float32()),
    ("press_run_len_spine", pa.int16()),
    ("events_since_press_onset", pa.int16()),
    ("events_since_last_press", pa.int16()),
    ("time_since_last_press_spine", pa.float32()),
    ("seg_press_count_spine", pa.int16()),
    ("seg_press_frac_spine", pa.float32()),
    ("press_run_id_spine", pa.string()),
    ("press_run_is_last_spine", pa.bool_()),
    ("press_run_exit_spine", pa.string()),
    # tier 2
    ("ff_available", pa.bool_()), ("ff_n_opp_visible", pa.int8()),
    ("ff_n_team_visible", pa.int8()), ("ff_visible_r3", pa.bool_()),
    ("ff_visible_r5", pa.bool_()), ("ff_recv_visible_r5", pa.bool_()),
    ("ff_nearest_opp_dist", pa.float32()), ("ff_opp_within_3", pa.int8()),
    ("ff_opp_within_5", pa.int8()), ("ff_lane_opp", pa.int8()),
    ("ff_lane_visible", pa.bool_()), ("ff_recv_opp_within_5", pa.int8()),
])

COLUMNS = [f.name for f in SCHEMA]
SPINE_COLUMNS = [f.name for f in SPINE_SCHEMA]


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
                    "match_id": mid, "competition_id": cid, "season_id": sid,
                    "comp_season_uid": f"{cid}:{sid}",
                    "competition_name": info.get("competition_name"),
                    "season_name": info.get("season_name"),
                    "match_date": m.get("match_date"),
                    "home_team_id": m["home_team"]["home_team_id"],
                    "away_team_id": m["away_team"]["away_team_id"],
                    "home_score": m.get("home_score"),
                    "away_score": m.get("away_score"),
                },
                ev, ff,
            ))
    return tasks


def flush(writer, buf: list[dict], columns: list[str], schema: pa.Schema) -> int:
    df = pd.DataFrame(buf, columns=columns)
    writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
    return len(df)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the spine and estimation tables")
    ap.add_argument("--limit", type=int, default=None, help="first N matches only")
    ap.add_argument("--tier2-only", action="store_true", help="only 360 matches")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--batch", type=int, default=150_000, help="rows per row group")
    args = ap.parse_args()

    if not (RAW / "competitions.json").exists():
        sys.exit("[build] no raw data found. Run `python fetch.py` first.")

    tasks = load_match_index(args.tier2_only)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"[build] {len(tasks)} matches, {args.workers} workers")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w_pass = pq.ParquetWriter(OUT_PASSES, SCHEMA, compression="zstd")
    w_spine = pq.ParquetWriter(OUT_SPINE, SPINE_SCHEMA, compression="zstd")

    pbuf: list[dict] = []
    sbuf: list[dict] = []
    n_pass = n_spine = done = 0
    errors: list[str] = []
    score_bad: list[dict] = []
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(parse_match, t) for t in tasks]
        for fut in as_completed(futs):
            prows, srows, chk, err = fut.result()
            done += 1
            if err:
                errors.append(err)
            if chk and not chk["ok"]:
                score_bad.append(chk)
            pbuf.extend(prows)
            sbuf.extend(srows)
            if len(pbuf) >= args.batch:
                n_pass += flush(w_pass, pbuf, COLUMNS, SCHEMA)
                pbuf = []
            if len(sbuf) >= args.batch:
                n_spine += flush(w_spine, sbuf, SPINE_COLUMNS, SPINE_SCHEMA)
                sbuf = []
            if done % 500 == 0 or done == len(tasks):
                print(f"[build] {done}/{len(tasks)} matches"
                      f"  {n_pass + len(pbuf):,} passes"
                      f"  {n_spine + len(sbuf):,} spine  {time.time() - t0:6.1f}s")

    if pbuf:
        n_pass += flush(w_pass, pbuf, COLUMNS, SCHEMA)
    if sbuf:
        n_spine += flush(w_spine, sbuf, SPINE_COLUMNS, SPINE_SCHEMA)
    w_pass.close()
    w_spine.close()

    for label, path, n in (("passes", OUT_PASSES, n_pass), ("spine", OUT_SPINE, n_spine)):
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"[build] {label:<7} {n:>10,} rows -> {shown}"
              f"  ({path.stat().st_size / 1e6:.1f} MB)")
    print(f"[build] elapsed {time.time() - t0:.1f}s")

    if errors:
        print(f"\n[build] {len(errors)} match(es) failed to parse:")
        for e in errors[:10]:
            print(f"          {e}")
    if score_bad:
        print(f"\n[build] score mismatch in {len(score_bad)} match(es):")
        for s in score_bad[:10]:
            print(f"          match {s['match_id']}: rebuilt "
                  f"{s['recon_home']}-{s['recon_away']} vs index "
                  f"{s['index_home']}-{s['index_away']}")
    else:
        print("[build] score reconstruction matches the match index in every match")


if __name__ == "__main__":
    main()
