# press-memory

**Does defensive pressure have memory?**

Soccer valuation models (EPV, OBSO, PAUSA, DEFCON) treat each action as a function of
the instantaneous state. This project tests whether that is enough: after *local*
pressure features are controlled for, does the **pressure history of a possession**
still predict degraded outcomes on subsequent passes?

Claim under test: *pressure is a possession-level state, not an event-level attribute.*

Data: StatsBomb open data only. No video, no tracking-provider feeds, no synthetic data.

---

## Pinned data version

All results in this paper are computed against a single pinned upstream commit:

```
repository  https://github.com/statsbomb/open-data
commit      b0bc9f22dd77c206ddedc1d742893b3bbe64baec
tree        bad02895ec3616202e076d9d96adcef48dc7c134
pinned on   2026-08-13
```

`fetch.py` downloads the archive **addressed by that commit SHA**, so a referee
running it in November gets exactly the match set used here, not whatever `master`
has drifted to. The pin is a single constant at the top of `fetch.py`; changing it
changes the dataset and invalidates every reported N.

It then **verifies the extracted tree against file counts recorded from the pinned
commit** (4,235 event files, 426 freeze-frame files, 80 match-index files) and
fails loudly on any mismatch, so a truncated or drifted download cannot silently
produce a different N.

> **Why an archive rather than `git clone`.** open-data is ~16 GB across ~9,000
> large JSON blobs. Both `git fetch --depth 1 <sha>` and a blob-filtered partial
> clone were measured stalling indefinitely against GitHub — server-side pack
> enumeration for this repository never begins streaming (84 KB moved in 75 s;
> 320 KB in 20 min). The codeload archive for the identical commit streams at
> ~3.5 MB/s. The pin is exactly as strong; only the transport differs.

At this commit the corpus is:

| tier | scope | matches |
|---|---|---|
| Tier 1 | all matches in the match index | **3,961** |
| Tier 2 | matches with a `three-sixty` freeze-frame file | **426** |

---

## Reproducing from scratch

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python fetch.py                 # ~16 GB on disk: events + three-sixty
python fetch.py --include core  # ~13 GB: Tier 1 only, skips freeze frames

python fetch.py --verify        # re-check an existing tree; downloads nothing
```

The download resumes across dropped connections, extracts only the requested
`data/` subtrees, and writes `data/raw/MANIFEST.json` recording the commit SHA,
the archive's sha256, and per-directory file counts and byte totals. `--verify`
re-checks a tree against both the pinned counts and that manifest.

Disk budget at this pin: `data/events` 12.8 GB, `data/three-sixty` 3.2 GB,
`data/lineups` 0.08 GB, `data/matches` 7 MB.

Then:

```bash
python build.py                 # -> data/processed/passes.parquet
python diagnostics.py           # study-design diagnostics, reads the Parquet only
```

`data/raw/` and `data/processed/` are gitignored. The repository holds code and
documentation only; the data layer is always rebuilt from the pin.

---

## Layout

```
fetch.py          pinned download of StatsBomb open-data + verification
build.py          raw JSON -> spine + estimation table
validate.py       physical-plausibility checks; run after every build
diagnostics.py    study-design diagnostics
src/load.py       the only sanctioned reader (enforces nullable dtypes)
requirements.txt  pandas, pyarrow, numpy, statsmodels, matplotlib
data/raw/         gitignored; written by fetch.py
data/processed/   gitignored; written by build.py
```

**JSON is parsed exactly once.** `build.py` writes two tables and every
downstream stage reads them. No analysis step re-opens the raw event JSON.

| table | grain | rows | purpose |
|---|---|---|---|
| `spine.parquet` | one pressed-eligible ball event | 6,969,870 | the exposure clock, press runs, exit taxonomy |
| `passes.parquet` | one pass | 3,836,550 | the estimation table; joins its history from the spine |

Passes remain the only outcome rows. The spine exists because pressure exposure
does not happen only on passes: for 77.9% of passes the immediately preceding
ball event is a **carry**, and a pass-only clock is blind to pressure applied
during it.

**Always read via `src.load`, never bare `pd.read_parquet`:**

```python
from src.load import load_passes, load_spine, analysis_sample
df = analysis_sample(load_passes())
```

Anything that can be "not applicable" is written as NULL, never 0 and never
forward-filled. `src.load` forces `dtype_backend="numpy_nullable"` so those
columns stay `Int16`/`boolean`; read them with the default backend and they
degrade to `float64`/`object`, where a stray `.fillna(0)` silently turns "no
press was ever observed here" into "zero events since the last press".

---

## Validation policy

**Every derived geometric quantity is checked against a known physical scale
before it is used.** Run `python validate.py` after every build; it exits
non-zero on failure.

This policy exists because of a specific near-miss, recorded in caveat 0 below.
An earlier check that the `Pass`&harr;`Pressure` link is symmetric on 95% of
pressed passes gave false comfort: **link integrity says nothing about frame
consistency.** Two things being correctly *associated* does not make their
coordinates *comparable*. What caught the error was asking how many metres apart
a presser and the player being pressed were, and whether football permits that
number.

Current status: 25 checks, 24 pass, 1 warning (3 upstream freeze frames of
373,640 report more than 11 opponents). Anchors include pass length against
Euclidean distance (max error 1e-5 m), goal kicks at 114.00 m from the attacking
goal, corners at 40.00 m, and carrier-to-presser distance at 3.81 m with
period-to-period spread of 0.010 m.

---


## Data provenance caveats

Established by direct inspection of the corpus at the pinned commit. These are
properties of the upstream data, and they constrain the design:

0. **Every event is logged in the acting team's own attacking frame.** The team
   performing an event always attacks toward `x = 120`. A `Pressure` event is
   performed by the *defending* team, so its coordinates are rotated 180° relative
   to the pass it acts on, and the two cannot be compared until one is mirrored to
   `(120 - x, 80 - y)`. Measured over 6,795 linked pass/pressure pairs, raw
   carrier-to-presser separation has median **65.7 m**; mirrored it is **3.9 m**,
   with 66% within 5 m. `build.py` mirrors pressure locations into the passer's
   frame. Freeze frames need no such correction: they are already in the actor's
   frame (actor location matches event location to within 0.7 m, median 0.0).

1. **`Pressure`-event density is not uniform across competition-seasons.** In a
   131-match sample it ranges from ~184 to ~468 pressure events per match, and the
   share of passes flagged `under_pressure` ranges from 6.2% to 25.4%. Adjacent
   seasons of the *same* competition differ by ~2.8x. Part of this is real football,
   part is annotation regime. Pooling competition-seasons without fixed effects
   would confound annotation intensity with pressure.

2. **`possession` ids can span a half boundary** (~1 per match). `possession` alone
   is not a safe grouping key; `(match_id, period, possession)` is.

3. **5.9% of passes are made by a team other than `possession_team`.** A StatsBomb
   possession can contain passes from both sides.

4. **`match_available_360` in `competitions.json` overstates coverage.** African Cup
   of Nations 2023 is flagged, but only 1 of its 52 matches has a freeze-frame file.
   360 availability must be determined from file presence, never from the flag.

5. **274 `data/events/*.json` files are not referenced by any match index entry.**
   Iterate matches from the index, not from the events directory.

6. **Freeze frames are camera-limited.** Only 8.5% of pass frames contain all 11
   opponents; the median is 10. Each frame carries a `visible_area` polygon, so
   local-pressure validity is checkable per event: a full 5 m disc around the carrier
   is inside the visible area for 86.3% of passes, but only 49.5% at 10 m. Tier 2
   geometric pressure is therefore defined at a small radius and gated on that flag.

---


7. **`under_pressure` is definitional, not measured, on four event types.** Over 150
   matches of raw events: Pass 16.0%, Carry 35.8%, Miscontrol 27.9%, Shot 25.9% —
   but Dribble, Duel, Dispossessed and Clearance are all **exactly 100.0%**.
   StatsBomb sets the flag on those by definition. They must stay off the exposure
   clock or it becomes tautological: every dribble would reset "events since last
   press" whether or not new pressure occurred. Hence the spine's `clock` role is
   Pass and Carry only.

---

## Design diagnostics (computed at the pin)

Run `python diagnostics.py`. These are properties of the data layer. **No model
has been fitted and no result below is a finding about football.**

### Exposure clock: the pass-level clock is blind to most pressure

For 77.9% of passes the immediately preceding ball event is a **carry**, not a
pass. Measured over 3,160,436 analysis-sample passes:

| | pass clock | spine clock |
|---|---|---|
| reports no press in the segment | 52.9% | 38.7% |
| mean accumulated presses | 0.625 | 1.787 (**2.86×**) |

- **448,224 passes (14.2%)** have a pass clock reporting *no press* where the
  spine shows one — 26.8% of all pass-clock nulls.
- The lag-1 pressure state differs on **25.2%** of passes; in 509,993 of those the
  spine says pressed where the pass clock says not.

This is non-classical measurement error in the treatment variable, not a gap in
the hazard model.

### Press-run exits (846,870 runs over Pass+Carry)

| competing risk | runs | share |
|---|---|---|
| escape (incl. reaching a shot) | 539,852 | 63.7% |
| turnover | 233,747 | 27.6% |
| stoppage (6.9% foul, 1.4% out) | 70,098 | 8.3% |
| other | 3,173 | 0.4% |

The whistle share rises from 2.4% under a pass-only definition to **6.9%** here,
because fouls terminate carries, not passes: the foul-exit rate is **11.91%** on
carry-terminated runs against **0.26%** on pass-terminated ones, a 46× gap.

⚠️ **Run length is confounded with terminating event type by parity.** Passes and
carries alternate, so runs of length 1/3/5 end on a carry (80.9%/65.5%/56.4%) and
runs of length 2/4/6 end on a pass. Since only carries can end in a foul, the
exit mix oscillates with parity. A competing-risks model using run length as a
duration **must condition on terminating event type**, or parity will masquerade
as a duration effect. The monotone turnover gradient reported earlier under the
pass-only clock does not survive this correction.

### Escape, positively defined

Structural escape (possession retained, next Pass/Carry unflagged) is 60.8% of
runs. It does have a positive signature — median **+6.29 m** of separation gained
from the presser, positive in **82.0%** of cases — but it is not progressive:
median progression **+0.00 m**, positive in only 49.2%.

| definition | share of escapes | of all runs |
|---|---|---|
| E0 structural (current) | 100.0% | 60.8% |
| E1 ball ends ≥10 m from the presser | 46.2% | 28.1% |
| E2 gained ≥5 m of separation | 50.1% | 30.5% |
| E3 progressed ≥5 m toward goal | 25.7% | 15.6% |
| E4 gained ≥5 m **and** progressed ≥5 m | 14.4% | 8.8% |
| E5 gained separation **and** progressed (any) | 31.4% | 19.1% |

The modal escape keeps the ball and gets away from the presser without going
forward. "Escape" should be split into **relief** and **progressive escape**
rather than treated as one risk.

### Presser staleness and falsification test 3b

The `Pressure` event records the presser's position at *t − lead*, while the
freeze frame records every opponent at the moment of the pass. The gap between
them, as a function of lead, is the drift. Fitted over 14,532 pressed passes with
a freeze frame across 120 matches:

```
median positional error  =  2.46 m  +  1.02 m/s x lead
```

The 2.46 m intercept is the irreducible floor. For `lag1_presser_dist_to_t` the
staleness **compounds**, because the presser position is recorded at (t−1 − lead)
while the carrier position is at t: median total elapsed time **3.30 s**, implying
a median positional error of **5.82 m** against a measure whose own median is
17.76 m — an error/signal ratio of 0.33.

| staleness cap | n retained | % kept | median error | median measure | error/signal |
|---|---|---|---|---|---|
| ≤ 1 s | 6,561 | 1.8% | 3.35 m | 8.16 m | 0.41 |
| ≤ 2 s | 68,877 | 19.1% | 4.03 m | 10.66 m | 0.38 |
| ≤ 3 s | 156,019 | 43.3% | 4.61 m | 12.90 m | 0.36 |
| ≤ 5 s | 277,258 | 76.9% | 5.32 m | 15.70 m | 0.34 |
| none | 360,489 | 100.0% | 5.82 m | 17.76 m | **0.33** |

⚠️ **Capping does not improve signal-to-noise — it makes it slightly worse while
discarding up to 98% of the data.** A short elapsed time also means the ball has
not travelled far, so the measure shrinks faster than the error does. Carry
staleness as a control; do not cap.

**Recommendation for 3b:** run it on **presser identity** as the primary contrast
(the same defender presses at both t−1 and t on only 5.1% of 360,647 comparable
passes), with distance as a continuous secondary. Identity is immune to positional
drift; distance is not.

### Tier 2 chain coverage

418 usable matches (not the 426 files on disk; eight contain no usable frames),
377,509 analysis-sample passes.

| gate | P(t) | P(t, t−1) | P(t, t−1, t−2) | P(t)³ |
|---|---|---|---|---|
| frame present | 0.887 | 0.735 | 0.670 | 0.697 |
| + 3 m gate | 0.846 | 0.675 | 0.594 | 0.605 |
| + 5 m gate | 0.757 | 0.548 | **0.439** | 0.433 |

Usable passes under the 5 m gate: 285,614 at t; 190,306 for (t, t−1); 127,935 for
(t, t−1, t−2). Joint coverage tracks the independence product, so availability is
**not** clustered along the lag chain.

**Censoring is not treatment-related.** P(frame + 5 m gate) is 0.762 on pressed
passes against 0.756 on unpressed, and moves only 0.752 → 0.764 as accumulated
spine pressure rises from 0 to ≥4.

⚠️ **But it is spatially patterned, and that is a coverage limitation, not a
nuisance to control away.** Coverage is 0.920 in the central attacking channel
against 0.692 and 0.689 in the wide attacking channels, because wide positions sit
near the edge of the visible-area polygon. Press-to-touchline — using the sideline
as an extra defender — is one of the most canonical pressing patterns in the game,
and Tier 2 systematically under-observes exactly it. Zone-as-control fixes the
estimation; it does not recover the missing observations.

**Composition.** 292 of 418 Tier 2 matches (**69.9%**) are international
tournaments, which press differently from league football and have less squad
continuity.

---

## Scope

This repository contains the pressure-memory analysis only. A separate computer
vision pipeline and match simulator exist and are **deliberately excluded**; no
result here depends on them.

Single author. Target venue: MIT Sloan Sports Analytics Conference (SSAC27).
