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
fetch.py          pinned download of StatsBomb open-data + manifest/verification
build.py          raw JSON -> one pass-level table
diagnostics.py    chain-coverage and press-run-exit diagnostics
requirements.txt  pandas, pyarrow, numpy, statsmodels, matplotlib
data/raw/         gitignored; written by fetch.py
data/processed/   gitignored; written by build.py
```

**JSON is parsed exactly once.** `build.py` writes `data/processed/passes.parquet`
(one row per pass) and every downstream stage reads that file. No analysis step
re-opens the raw event JSON.

**Always read it like this:**

```python
df = pd.read_parquet("data/processed/passes.parquet", dtype_backend="numpy_nullable")
```

Anything that can be "not applicable" is written as NULL, never 0 and never
forward-filled — no preceding press, no t−1 within the segment, no freeze frame.
Without `dtype_backend="numpy_nullable"` pandas degrades those columns to
`float64`/`object`; with it they stay `Int16`/`boolean` and missingness
propagates through arithmetic instead of silently becoming a number.

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

## Design diagnostics (computed at the pin)

Run `python diagnostics.py`. These are properties of the data layer, not results:
no model has been fitted.

**Built table:** 3,836,550 passes from 3,961 matches, 285 MB Parquet, 78 s.
Reconstructed scores match the match index in every one of the 3,961 matches.

**Chain coverage.** Tier 2 usable matches: **418**, not the 426 files on disk —
eight files contain no usable freeze frames. Analysis sample (possession-team,
non-restart) in those matches: 377,509 passes.

| gate | P(t) | P(t, t−1) | P(t, t−1, t−2) | P(t)³ |
|---|---|---|---|---|
| frame present | 0.887 | 0.735 | 0.670 | 0.697 |
| + 3 m visibility gate | 0.846 | 0.675 | 0.594 | 0.605 |
| + 5 m visibility gate | 0.757 | 0.548 | **0.439** | 0.433 |

Usable passes under the 5 m gate: 285,614 at t; 190,306 for (t, t−1); 127,935 for
(t, t−1, t−2). Joint coverage tracks the independence product almost exactly, so
frame availability is **not** strongly clustered along the lag chain.

**Censoring is not treatment-related.** P(frame + 5 m gate) is 0.762 on pressed
passes vs 0.756 on unpressed, and moves from 0.752 to 0.769 as accumulated
within-segment pressure rises from 0 to ≥4. Both gradients are negligible. The
censoring is *spatial*: 0.920 in the central attacking channel vs 0.692–0.689 in
the wide attacking channels, because wide positions sit near the edge of the
visible-area polygon. Zone is already a Stage 1 control.

**Composition.** 292 of 418 Tier 2 matches (**69.9%**) are international
tournaments.

**Press-run exits.** 442,170 runs (maximal consecutive under-pressure passes by
the possession team within a segment):

| risk | runs | share |
|---|---|---|
| escape (incl. reaching a shot) | 285,195 | 64.5% |
| turnover | 131,116 | 29.7% |
| stoppage (2.7% out of play, 2.4% foul) | 22,607 | 5.1% |
| other / period end | 3,252 | 0.7% |

Turnover share rises monotonically with run length (29.3% at length 1 to 35.3% at
length 4) while escape falls (62.6% to 55.7%).

⚠️ **The 5.1% stoppage share is an artifact of defining runs over passes.** There
are 25.2 under-pressure `Foul Won` events per match but only ~2.7 recorded
`stoppage_foul` run exits — the pass-level definition captures about **11%** of
press-terminating whistles. 59.5% of pressed fouls occur in possessions containing
no pressed pass at all, and pressed carries outnumber pressed passes 265.9 to
152.8 per match. A competing-risks model must therefore be specified over pressed
**ball-events (passes and carries)**, not passes alone, or the whistle risk is
structurally undercounted by roughly an order of magnitude.

---

## Scope

This repository contains the pressure-memory analysis only. A separate computer
vision pipeline and match simulator exist and are **deliberately excluded**; no
result here depends on them.

Single author. Target venue: MIT Sloan Sports Analytics Conference (SSAC27).
