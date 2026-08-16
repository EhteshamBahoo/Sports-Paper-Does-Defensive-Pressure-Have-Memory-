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
python build.py                 # -> data/processed/{passes,spine}.parquet   ~140 s
python pressures.py             # -> data/processed/pressures.parquet         ~17 s
python validate.py              # physical-plausibility policy, 25 checks
python diagnostics.py           # study-design diagnostics, reads the Parquet only

python stage1_baseline.py       # baseline calibration (add --final for the test set)
python stage2.py                # Stage 2 primary estimand, validation only
python stage2_mechanism.py      # Stage 2 threats A and B, validation only
```

`data/raw/` and `data/processed/` are gitignored. The repository holds code and
documentation only; the data layer is always rebuilt from the pin.

---

## Layout

```
fetch.py             pinned download of StatsBomb open-data + verification
build.py             raw JSON -> spine + estimation table
pressures.py         raw JSON -> Pressure events with their durations
validate.py          physical-plausibility checks; run after every build
diagnostics.py       study-design diagnostics
stage1_baseline.py   Stage 1: local pressure only; calibration, no residuals
stage2.py            Stage 2 primary estimand (validation only)
stage2_mechanism.py  Stage 2 threats A and B (validation only)
src/load.py          the only sanctioned reader (enforces nullable dtypes)
requirements.txt     pandas, pyarrow, numpy, statsmodels, matplotlib
data/raw/            gitignored; written by fetch.py
data/processed/      gitignored; written by build.py and pressures.py
```

**JSON is parsed once per table.** `build.py` writes the two analysis tables and
every downstream stage reads them; no analysis step re-opens the raw event JSON.
`pressures.py` is the one deliberate second pass, added on 2026-08-17 because the
Stage 2 mechanism test needs the `Pressure` event's own `duration`, which the
build does not retain.

| table | grain | rows | purpose |
|---|---|---|---|
| `spine.parquet` | one pressed-eligible ball event | 6,969,870 | the exposure clock, press runs, exit taxonomy |
| `passes.parquet` | one pass | 3,836,550 | the estimation table; joins its history from the spine |
| `pressures.parquet` | one `Pressure` event | 1,394,692 | pressure windows, for the Stage 2 threat-A test |

`pressures.parquet` stores **raw** coordinates in the pressing team's attacking
frame; consumers mirror at join time, because only the consumer knows which team
is acting. Storing a pre-mirrored coordinate would bake in an assumption the
table cannot check — which is the exact class of error in ledger entry 1.

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

Current status: 29 checks, 28 pass, 1 warning (3 upstream freeze frames of
373,640 report more than 11 opponents). Anchors include pass length against
Euclidean distance (max error 1e-5 m), goal kicks at 114.00 m from the attacking
goal, corners at 40.00 m, and carrier-to-presser distance at 3.81 m with
period-to-period spread of 0.010 m.

### Post-treatment features are guilty until justified

*Added 2026-08-17, after the second instance of the same error.*

**Any feature derived from `end_location` is post-treatment by construction and
must be declared before it can enter a model.** StatsBomb sets `end_location` to
where the ball actually ended, which on an intercepted pass is the interception
point — so such a feature is partly a function of the outcome.

This has now happened twice: `pass_length`/`pass_angle` in Stage 1, worth about a
third of conventional model skill; and four `ff_*` columns in the first draft of
the Stage 2 threat-B control, worth +7.0 Brier skill points against +0.07 for the
clean block. The second instance came one stage *after* the first was documented,
in unfamiliar columns. Prose did not transfer, so it is now mechanical:

1. **A register** — `src.load.POST_TREATMENT`, each entry with its provenance
   (`build` = computed here, `statsbomb` = arrived already contaminated) and a
   reason.
2. **A runtime guard** — `src.load.assert_pre_treatment(cols, allow=(...))`.
   Using `allow` is a claim that the inclusion is deliberate and stated.
3. **A source check** — `validate.py` re-derives the truth from `build.py`'s AST
   rather than trusting the register: taint seeded at `end_location`/`end_x`/
   `end_y`/`target`, propagated through local assignments and across functions,
   reported for every output column. An unregistered hit fails the run. It found
   11 columns; two (`ff_visible_r3`, `ff_visible_r5`) are exempt with a stated
   argument, because the helpers they use are strictly rowwise so the flagged
   dataflow carries no value.
4. **A specification check** — `validate.py` walks `build_design`'s branches and
   verifies that `M0`/`M0i`/`M0x` read no registered column, while `M1`/`M2`/`M3`
   read only what they declare. Those three include `pass_length` on purpose:
   their gap to `M0` *is* the leakage estimate.

The exemption in (3) is load-bearing rather than bookkeeping: `ff_visible_r5` is
the **sample gate** for the entire Tier 2 arm. Had it been target-derived, the
gate would have selected on the outcome — a worse defect than the control leak it
was introduced to avoid. Both guards were verified to fire against a deliberately
poisoned source before being relied on.

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

*(Superseded specification retained for comparison — see **Corrections ledger #3**.)*

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
pass-only clock does not survive this correction — see **Corrections ledger #2**.

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

## Fallback thesis (logged 2026-08-13, before Stage 2 is fitted)

Recorded now so it cannot be a post-hoc pivot if the residual signal is null.

Of 846,870 press runs, only **8.8%** end in a progressive escape. The full
decomposition:

| outcome of a press run | share |
|---|---|
| turnover | 27.6% |
| escape, no separation gained (`neither`) | 24.5% |
| escape with separation but no progression (`relief`) | 21.7% |
| **escape with both (`progressive`)** | **8.8%** |
| stoppage (foul 6.9%, out of play 1.4%) | 8.3% |
| escape, separation unmeasurable | 5.9% |
| shot | 2.9% |

**Claim: successful pressing does not primarily win the ball; it primarily denies
progression without winning it.** Pressure ends in a turnover 27.6% of the time,
but leaves the attacking team holding the ball and going nowhere 46.2% of the
time (`neither` + `relief`). The modal press outcome is not a takeaway.

This is measurable on Tier 1, needs no freeze frames, is coach-legible, and does
not depend on the memory hypothesis surviving Stage 2.

---

## Stage 1 baseline (fitted 2026-08-13)

`python stage1_baseline.py --specs M0i,M0x --final`. Three-way split **by match**
(50/20/30, deterministic SHA-256 of `match_id`); specification developed on
validation; test evaluated at the end.

### pass_length and pass_angle are post-treatment

StatsBomb sets `end_location` to where the ball actually ended, so an intercepted
pass has its length truncated at the interception point. On train: complete
passes have p10 length **8.07 m**, incomplete **3.61 m**, and **61%** of passes
recorded under 5 m are interceptions (completing at 0.387 against 0.825 overall).
A baseline containing `pass_length` is therefore partly predicting the outcome
from the outcome.

### Model comparison, held-out test

| model | features | Brier | skill | worst decile |
|---|---|---|---|---|
| M0 | pre-treatment only | 0.11060 | 0.232 | 0.0358 |
| M0i | M0 + height×zone, height×play-pattern | 0.10947 | 0.240 | 0.0193 |
| **M0x** | **M0i + pressure × geometry interactions** | **0.10897** | **0.244** | **0.0156** |
| M1 | spec-exact (pressure + zone + length/angle) | 0.10228 | 0.290 | 0.0327 |
| M2 | M0 + length/angle | 0.09358 | 0.350 | 0.0245 |
| M3 | M2 + height×length | 0.09278 | 0.356 | 0.0156 |

Validation and test agree to ~0.001 Brier throughout, so none of this is
overfitting. **M0x is the residual base for Stage 2.** The M3−M0x skill gap
(0.356 vs 0.244) means roughly **a third of the conventional model's apparent
skill comes from outcome leakage**, not football.

### M0x specification

M0i, plus interactions only — no new features:

```
under_pressure   ×  zone (8), pass_height (2), pass_body_part (5),
                    play_pattern (8), dist_to_goal, origin_x, origin_y
inv_presser_dist ×  pass_height (2), dist_to_goal
```

30 added columns, 152 → 182. The rationale is mechanical: a pooled additive logit
forces identical geometry coefficients on pressed and unpressed passes. If
geometry matters less once a defender is on you, the pooled model over-applies it
to pressed passes and spreads their predictions too wide — which is exactly the
over-dispersion the M0i diagnostic showed. Interacting pressure with geometry
lets the pressed subsample carry its own slope.

### Calibration within pressure strata (M0x, test only)

| quintile of predicted p | n | predicted | observed | diff |
|---|---|---|---|---|
| **pressed**, 1 | 32,188 | 0.4507 | 0.4519 | +0.0012 |
| pressed, 2 | 32,187 | 0.6902 | 0.6885 | −0.0017 |
| pressed, 3 | 32,188 | 0.8254 | 0.8234 | −0.0021 |
| pressed, 4 | 32,187 | 0.8889 | 0.8879 | −0.0010 |
| pressed, 5 | 32,188 | 0.9348 | 0.9290 | −0.0058 |
| **unpressed**, 1 | 158,046 | 0.4988 | 0.5003 | +0.0014 |
| unpressed, 2 | 158,046 | 0.8359 | 0.8327 | −0.0031 |
| unpressed, 3 | 158,046 | 0.9301 | 0.9255 | −0.0046 |
| unpressed, 4 | 158,046 | 0.9557 | 0.9606 | +0.0049 |
| unpressed, 5 | 158,047 | 0.9768 | 0.9791 | +0.0024 |

Pressed max |diff| falls 0.0609 → **0.0058**; unpressed falls 0.0120 → **0.0049**,
so pressed calibration was not bought at unpressed's expense. Aggregate
calibration is unchanged where it was already fine (pressed −0.0019, unpressed
+0.0002).

### Specification search, disclosed

One round, on 2026-08-13. The M0i pressed-quintile diagnostic was observed first;
the success criterion — *no systematic sign pattern across pressed quintiles, and
unpressed not degraded* — was fixed **before** the interaction model was fitted.
30 interaction columns were added in a single specification; no alternatives were
tried and none were discarded. Validation and test moved together (validation
Brier 0.11037 → 0.10983, skill 0.238 → 0.242, worst decile 0.0197 → 0.0169; test
as tabled above), which is the evidence that this is a real fix and not a test-set
artefact. An earlier smoke run of two specifications was inspected on a 300k
random subsample before the validation split existed; the redesign that followed
was driven by train-only diagnostics.

### ⚠️ Known limitation: residual level tilt in the top pressed quintile

M0x still overpredicts the easiest pressed passes by **0.6 pp** (quintile 5:
0.9348 predicted, 0.9290 observed). At n = 32,188 and p ≈ 0.93 the standard error
is ≈ 0.0014, so −0.0058 is about 4 SE — statistically detectable, not noise. It is
a level tilt confined to one cell, not the monotone slope it replaced. Stage 2
therefore stratifies by predicted-probability quintile: a residual signal that
appears only in quintile 5 is baseline misfit, not memory.

---

## Corrections ledger

Six results produced in this project have been corrected, retracted or
superseded under inspection. Two of them were headline numbers (entries 1 and 2);
entries 3 and 4 superseded a measurement and a model that later work was built
on; entries 5 and 6 were caught before the number left this repository. They are
kept here rather than deleted, because the limitations section in December is
only credible if the corrections are visible.

### 1. Carrier-to-presser distance — *corrected 2026-08-13*

**Was:** `presser_dist` computed from raw coordinates, giving a median
carrier-to-presser separation of **65.7 m**.
**Why wrong:** StatsBomb logs every event in the acting team's own attacking
frame. A `Pressure` event is performed by the defending team, so it is rotated
180° relative to the pass it acts on. Link integrity had been verified (symmetric
`related_events` on 95% of pressed passes) and gave false comfort — correct
*association* does not make coordinates *comparable*.
**Now:** mirrored to `(120 − x, 80 − y)`; median **3.9 m**. This silently
invalidated the primary Stage 1 pressure control and the test-3b spatial measure.
See caveat 0 and the validation policy.

### 2. Monotone turnover gradient — *retracted 2026-08-13*

**Was:** "turnover share rises monotonically with press-run length, 29.3% at
length 1 to 35.3% at length 4," computed under the pass-only clock.
**Why wrong:** passes and carries alternate along a possession, so run-length
parity determines the terminating event type — length 1/3/5 runs end on a carry
(80.9%/65.5%/56.4%), length 2/4/6 on a pass. Only carries can end in a foul
(11.91% against 0.26%). The apparent gradient was parity, not duration.
**Now:** run length is not used as the exposure axis at all. `press_elapsed_s`
and `press_n_pressers` are parity-free and are the primary axes; run length is
robustness only, conditioned on terminating event type.

### 3. Pass-only exposure clock — *superseded 2026-08-13*

**Was:** pressure history counted over passes (`passes_since_last_press` and
friends).
**Why wrong:** for **77.9%** of passes the immediately preceding ball event is a
carry, and carries are pressed at 35.8% against 16.0% for passes. 448,224 passes
(14.2% of the analysis sample) had the pass clock reporting *no press* where the
spine shows one; lag-1 pressure state differs on 25.2% of passes; accumulated
pressure is 2.86× higher on the spine. This was measurement error in the
treatment variable, not a gap in the hazard model.
**Now:** the spine (`spine.parquet`, Pass + Carry) carries the clock. The
pass-level columns are retained in `passes.parquet` so the two clocks can be
compared directly.

### 4. M0i pressed-quintile calibration — *superseded 2026-08-13*

**Was:** M0i as the Stage 2 residual base. Its aggregate pressed calibration read
+0.0020 and looked fine.
**Why wrong:** the aggregate was cancellation, not accuracy. By quintile of
predicted probability the error ran in a systematic slope:

| quintile of predicted p, pressed only | n | predicted | observed | diff |
|---|---|---|---|---|
| (0.058, 0.543] | 32,188 | 0.3997 | 0.4606 | **+0.0609** |
| (0.543, 0.783] | 32,187 | 0.6757 | 0.6794 | +0.0037 |
| (0.783, 0.893] | 32,188 | 0.8452 | 0.8276 | −0.0176 |
| (0.893, 0.934] | 32,187 | 0.9160 | 0.8851 | **−0.0309** |
| (0.934, 0.999] | 32,188 | 0.9538 | 0.9278 | **−0.0260** |

Since pressure history correlates with position in the predicted distribution,
Stage 2 on this base would have read baseline misfit as memory.
**Now:** M0x, above. Pressed max |diff| 0.0058, no monotone pattern.

### 5. Stage 2 fitted without the pre-registered ordinal control — *corrected 2026-08-16*

**Was:** the first Stage 2 run reported the elapsed × quintile table with
`pass_ord_in_poss` **stratified** but not **controlled**.
**Why wrong:** the pre-registration requires it "as a control, **and** results
additionally stratified by it" — both, and only stratification was implemented.
It is not a cosmetic omission: the two samples are imbalanced on possession
position (primary mean ordinal 7.17 against benchmark 2.47) and M0x contains no
ordinal term, so its benchmark residual runs +3.571 pp at ordinal 0–1 against
−1.165 pp at 2–3. Without the control the contrast is partly early-possession
against late-possession.
**Caught by:** the composition check comparing primary and benchmark means, run
before reporting.
**Now:** an adjusted regression on elapsed-bin + ordinal + quintile dummies is the
reported estimate. The effect survived it (shortest bin −6.380 adjusted against
−6.003 unadjusted), so the correction did not change the verdict — but it was
found by the written rule, not by the result looking wrong, which is the whole
argument for pre-registering.

### 6. Post-treatment features in the Tier 2 control block — *caught before use, 2026-08-17*

**Was:** the first draft of the threat-B geometry control used all twelve `ff_*`
columns, and reported that freeze-frame geometry adds **+7.06 skill points** and
absorbs roughly half the history effect.
**Why wrong:** four of those columns (`ff_lane_opp`, `ff_recv_opp_within_5`,
`ff_recv_visible_r5`, `ff_lane_visible`) are computed around `end_location`, which
on a failed pass is the interception point. They are post-treatment in exactly the
way `pass_length` is (caveat 4), so a control built from them absorbs the outcome
rather than the defensive state.
**Now:** origin-only geometry is the primary arm (+0.066 skill points); the
target-derived arm is reported only to bound the leakage. Listed here rather than
omitted because the erroneous number is the more impressive one, and because the
same defect had already been documented for `pass_length` one stage earlier — the
lesson did not transfer on its own.

---

## Stage 2 pre-registration (written 2026-08-13, before any fitting)

Nothing in this section has been estimated. It is recorded before the fact so
that a null result is publishable and a positive result is not a specification
search. No falsification test is specified here; those come after Stage 2.

> **Amendment 1 — 2026-08-16, before any fitting.** Four changes, all loosening or
> making explicit, none made with knowledge of any result: (a) the stoppage scope
> condition below was promoted from a housekeeping rule to a stated limit on the
> estimand; (b) H1 no longer requires strict bin-to-bin monotonic recovery, which
> is fragile to binning choices — attenuation is now tested as a trend; (c) the
> elapsed-time bin edges are pre-specified, set from the regressor's marginal
> distribution on the **train** split only, without reference to any outcome;
> (d) "survive every stratum" and the football-relevance floor are now defined in
> effect-size terms rather than significance, because at n ≈ 10⁶ everything is
> significant.

> **Amendment 2 — 2026-08-17. THE MEMORY CLAIM IS RETIRED.**
>
> ⚠️ **This amendment was written *after* Stage 2 and the mechanism tests, with
> full knowledge of the results.** It is not a pre-registration and must never be
> cited as one. It is a retirement notice. Everything above and below it stands
> unedited as written; this block records what the results did to it.
>
> **Retired:** the claim that pressure history has a causal residual effect on
> subsequent outcomes — "pressure is a possession-level state" in the strong,
> memory-trace sense. Three reasons, none individually fatal, jointly decisive.
>
> 1. **The clean band failed the floor.** The 2–8 s band was the defensible
>    evidence, being beyond any plausible annotation window. After origin-only
>    freeze-frame geometry it reads **−0.939 pp** — a *bounded null* on the
>    pre-registered scale (detectable, below the 1.0 pp relevance floor). Not a
>    practical null, which is reserved for < 0.5 pp; the distinction does not
>    rescue anything, but the ledger uses the term it registered.
> 2. **The surviving band is the unidentifiable one.** What clears the floor after
>    geometry control is the < 2 s band at −2.7 pp — precisely where the
>    annotation-resolution qualifier bites hardest: 66.7% of those passes fall
>    within 0.5 s of a pressure window closing, median gap +0.19 s. The surviving
>    effect lives in the one window where it cannot be distinguished from the
>    annotator having called time slightly early.
> 3. **Threat C, which this data cannot address at all.** StatsBomb 360 freeze
>    frames carry positions and no velocities. The geometry control therefore
>    knows where defenders *are*, not where they are *going*. A defence that is
>    compact and still closing is a different pressure state from one that is
>    compact and settled, and that difference is largest in the first two seconds
>    after a press ends, while defenders are still decelerating. The residual that
>    survives has exactly the shape of unmeasured defender velocity. This cannot
>    be ruled out with open event data plus static frames, by me or by anyone.
>    Recorded as a named threat, not a caveat — see the mechanism section.
>
> **Replacement claim, and the one the paper makes:**
>
> > **Event-level pressure annotation systematically under-measures defensive
> > pressure, and possession pressure history is a free correction for the
> > shortfall.**
>
> Models that treat pressure as an event attribute understate it; the shortfall
> persists for several seconds past the annotated flag, decaying on the same
> timescale as defensive geometry relaxes to benchmark; and possession history
> recovers roughly half of what freeze-frame geometry provides, at zero additional
> data cost. This is a measurement claim, not a causal one, and it is what was
> actually measured.
>
> **What the existing numbers now mean.** The uncontrolled −5.487 pp (< 2 s) and
> −2.645 pp (2–8 s) are no longer candidate causal effects. They are **the size of
> the measurement gap** — how much outcome-relevant defensive pressure the
> `under_pressure` flag fails to carry, expressed in completion probability. That
> is what they always were.
>
> **Explicitly not claimed:** that pressing has a lasting causal effect on the
> pressed team; that the residual is a psychological, physiological or tactical
> trace; that the < 2 s effect is anything other than jointly consistent with
> memory, annotation-boundary error, and defender velocity.
>
> **Consequences for what follows.** The Q1 non-attenuation diagnosis is
> **dropped** — it was diagnostic for a claim no longer being made. The
> falsification suite is retargeted at the measurement claim: spatial specificity
> (3b) and survivorship still apply and still matter, because a measurement gap
> should be spatially structured and must not be an artefact of possession
> selection. The test split remains sealed until the end.

### Residual

`resid = observed pass_success − M0x predicted probability`, M0x fitted on the
train split only and applied out of sample.

### Primary estimand

Residual as a function of **time since the press ended**, on passes where local
pressure is currently OFF but the possession has a pressure history.

```
sample     is_possession_team, not is_set_piece_restart, pass_success not null,
           under_pressure == False,
           events_since_last_press IS NOT NULL      (a prior press exists)
regressor  time_since_last_press_spine              (seconds since the last
                                                     pressed Pass/Carry)
benchmark  under_pressure == False AND
           events_since_last_press IS NULL          (no prior press in the segment)
```

⚠️ **One deliberate substitution, flagged for confirmation.** The instruction named
`time_since_last_press_s`, which is the *pass-level* clock column. Correction 3
above establishes that clock is blind to 77.9% of preceding ball events, so the
pre-registration uses the spine equivalent, `time_since_last_press_spine`. The
pass-level column will be reported as a robustness row so the two clocks can be
compared. Say the word if you intended the literal column.

### Scope condition: memory within uninterrupted play

"In the possession" is implemented as **within `segment_uid`** — possession ×
set-piece restart — because a stoppage dissolves the press and history must not
carry across it.

**This is a scope condition, not housekeeping, and it is not neutral.** Stoppage
accounts for **8.3%** of press-run exits (6.9% foul, 1.4% out of play), and fouls
won are *caused by* pressing. Segments that continue are therefore selected on the
press not having produced a whistle, which removes exactly the exit route pressing
generates. The estimand is **memory within uninterrupted play**; press episodes
terminated by the whistle lie outside it and the hypothesis is untestable across
them. The justification is that the press physically dissolves at a stoppage and
continuity of the tactical situation is the thing being measured — but the
selection is real and is named here rather than left for a referee to find.

### Direction, declared in advance

**Memory hypothesis (H1).** Mean residual is **negative in the shortest
elapsed-time bins** and **attenuates toward the benchmark as elapsed time grows**.
Attenuation is tested as a **trend**, not as a monotonicity check: no requirement
of strict bin-to-bin ordering, since bin-wise monotonicity is fragile to binning
choices and would let noise in a single bin reject a real effect.

**Null (H0).** Mean residual on the history-bearing sample does not depend on
`time_since_last_press_spine`: it is indistinguishable from the benchmark at short
elapsed times, and the trend slope is indistinguishable from zero.

### Pre-specified bins

Edges fixed on 2026-08-16 from the marginal distribution of the regressor on the
**train** split, with no reference to any outcome. Post-hoc binning is one of the
easier ways to manufacture a result, so they are frozen here:

```
[0,1)  [1,2)  [2,3)  [3,5)  [5,8)  [8,12)  [12,20)  [20,inf)   seconds
```

Train occupancy runs 5.9% to 16.5% per bin, smallest bin n = 41,882. Trend is
estimated on `log(1 + elapsed)` over the ungrouped data; the bins are for display
and for the sign check, not for the trend test.

### Effect-size floor

At n ≈ 10⁶ everything is significant, so relevance is defined before fitting.

**Reference scale.** The *contemporaneous* pressure effect is **8.35 pp** of
completion probability (test-set raw: pressed 0.7561, unpressed 0.8396). A memory
effect is a fraction of that.

| mean residual vs benchmark | verdict |
|---|---|
| < 0.5 pp | **practically null**, reported as such regardless of p-value |
| 0.5 – 1.0 pp | **bounded null** — detectable but below the relevance floor |
| ≥ 1.0 pp | **football-relevant**, claim available if the stratum rule passes |

1.0 pp is ~12% of the contemporaneous effect and compounds to ~4 pp of
possession-survival probability across a four-pass possession, which is the
smallest quantity a practitioner could act on.

### Decision rule, in effect-size terms

"Survives every stratum" is defined now, before any result is seen. A rule of the
form *p < 0.05 in all forty cells* would reject almost anything at these n, so the
criteria are about sign and magnitude:

1. **Magnitude.** Pooled |mean residual| in the two shortest bins ≥ **1.0 pp**.
2. **Sign consistency.** The estimate is negative in **≥ 4 of 5**
   predicted-probability quintiles and **≥ 3 of 4** `pass_ord_in_poss` strata.
3. **Magnitude consistency.** Among strata with the correct sign, the smallest
   |estimate| is **≥ 1/3 of the pooled |estimate|**. No single stratum carries it.
4. **Baseline-misfit discriminator.** Dropping predicted-probability **quintile 5**
   changes the pooled estimate by **< 50%**. M0x is known to overpredict the
   easiest pressed passes by 0.6 pp, so a signal concentrated in quintile 5 is
   baseline misfit, not memory.

All four must hold. If any fails, the result is reported as a null with the full
stratification shown. A null redirects the paper to the fallback thesis above,
which does not depend on this outcome.

### Secondary estimand

Accumulated exposure **within an ongoing run**, on the pressed subsample
(`under_pressure == True`):

- `press_elapsed_s` — wall-clock seconds under continuous pressure
- `press_n_pressers` — distinct defenders who have applied it

Both are parity-free. Note this is a **different sample** from the primary: the
primary is passes with pressure OFF, the secondary is passes with pressure ON.

`press_run_len_spine` is **robustness only**, and only ever conditioned on
terminating event type, per correction 2.

### Required in every specification

1. **`pass_ord_in_poss` as a control, and results additionally stratified by it.**
   Survivorship: a possession that has survived k passes under pressure is a
   selected possession. Turnover and escape select in opposite directions, so the
   bias is unsigned and cannot be claimed as conservative.
2. **Competition-season fixed effects.** Pressure-event density varies 184–468 per
   match across competition-seasons, adjacent seasons of the same competition
   differ by 2.8×, and the Pass:Carry mix varies 0.422–0.464, so any pooled rate
   is partly a composition statistic.
3. **Stratification by predicted-probability quintile.** This is the discriminator
   against the known top-quintile tilt: a signal present across quintiles is real;
   a signal concentrated in quintile 5 is baseline misfit.
4. **Match-level clustered standard errors, or a match-block bootstrap.** Passes
   within a match are not independent.

### Split discipline

Specification is developed on **validation only**. The **test split stays sealed**
and is touched once, at the end, for the final reported numbers. The split is the
same deterministic match hash used in Stage 1, so no pass that trained M0x can
appear in a Stage 2 test evaluation.

---

## Stage 2 result (validation, fitted 2026-08-16)

Fitted per the pre-registration above, on validation only. Test sealed. Primary
sample 262,292 passes, benchmark 226,768, over 742 matches.

All four pre-registered criteria pass. Pooled effect in the two shortest bins
−6.003 pp (clustered SE 0.190); negative in 5/5 quintiles and 4/4 ordinal strata;
minimum stratum above pooled/3; dropping Q5 moves it to −6.362 pp. Attenuation
trend +2.33 pp per log-second (SE 0.082, t = +28.4). **H1 supported**, subject to
the mechanism tests below, which change what the number means.

The discriminator came back clean: the effect is not concentrated in the tilted
top quintile, and Q2–Q5 all attenuate. Q1 does **not** attenuate (slope t = +1.58,
roughly −4 pp flat from 1 s to 20 s+). A persistent level offset that never decays
is not a memory signature, and this is recorded as unresolved.

### ⚠️ Known limitation: the benchmark is not neutral between samples

M0x carries a **+1.515 pp mean residual on the benchmark sample** (clustered SE
0.081) — it underpredicts unpressed passes in possessions that never contained a
press. Differencing removes the level, and the `pass_ord_in_poss` control absorbs
much of the cause, but the contrast still rests on a baseline that is not neutral
across the two samples being compared.

The mechanism is visible: the two samples are badly imbalanced on possession
position (primary mean ordinal 7.17 against benchmark 2.47), M0x contains no
ordinal term, and its benchmark residual runs +3.571 pp at ordinal 0–1 against
−1.165 pp at 2–3. Any specification that omits the ordinal control is therefore
partly measuring early-possession against late-possession. Recorded here because
it is easier to state now than to reconstruct in December.

---

## Stage 2 mechanism tests (validation, 2026-08-17)

`stage2_mechanism.py`. Two ways to produce the Stage 2 profile without any
memory. They are different threats and neither subsumes the other.

### Threat A — the press had not actually ended: **cleared**

"The press ended" is defined by the last ball event flagged `under_pressure`, but
a `Pressure` event is a window, not an instant (p50 0.683 s, p90 1.815 s, p99
4.136 s, max 35.4 s). A pass logged 0.4 s later and flagged unpressed could still
sit inside a live window — contemporaneous pressure, not memory.

`pressures.py` extracts all 1,394,692 Pressure events with their durations and
rebuilds the windows **independently of `related_events`**, so the test does not
lean on the same linkage twice. Instrument validated before use: passes the
annotator flagged pressed are 88.56% inside a live window (median gap −0.124 s),
unpressed passes 0.02% (median gap +14.9 s). The join fires.

| elapsed | n | live % | within 0.5 s of close | median gap |
|---|---|---|---|---|
| [0, 1) | 15,762 | 0.09 | 66.7 | +0.19 s |
| [1, 2) | 31,612 | 0.07 | 35.0 | +1.03 s |
| [2, 3) | 29,029 | 0.05 | 18.5 | +2.05 s |
| [5, 8) | 43,473 | 0.02 | 3.7 | +5.61 s |

Only 0.03% of the primary sample (68 passes) is inside a live window. Removing
them moves the shortest bin from −6.380 to −6.379 pp. A distance ladder (covering
presser within 5/10/15 m) changes nothing. **Threat A is not the explanation.**

The honest qualifier: 66.7% of the shortest bin falls within 0.5 s of a window
closing, median gap +0.19 s. The passes are outside the annotated window, but
only just. Whether a press is physically over 0.19 s after the annotator says so
is a question about the annotation's temporal resolution, not a defect this test
can settle.

### Threat B — unmeasured contemporaneous defensive state: **roughly half the effect**

M0x controls for pressure via the annotator flag and the linked presser's
distance. It does not control for the defending team's shape. Tier 2 freeze
frames give observed defender geometry at t, so this is testable. Coverage needed
is a frame at t only — the history variable is Tier 1 — so this is the
single-frame gate, not the frame-chain gate: 10.2% of validation passes have a
frame, 85.4% of those clear the 5 m origin-visibility margin (n = 51,213).

**The defence is in fact still compact after the press ends, and it relaxes on
the same timescale as the effect:**

| elapsed | n | nearest opponent | opp within 3 m | opp within 5 m |
|---|---|---|---|---|
| [0, 1) | 1,249 | 2.42 m | 0.841 | 1.367 |
| [1, 2) | 2,523 | 3.33 m | 0.651 | 1.175 |
| [2, 3) | 2,449 | 4.44 m | 0.473 | 0.954 |
| [3, 5) | 3,605 | 5.18 m | 0.384 | 0.839 |
| [5, 8) | 3,929 | 5.87 m | 0.274 | 0.710 |
| [8, 12) | 3,470 | 6.13 m | 0.241 | 0.630 |
| **benchmark** | 19,118 | **6.24 m** | **0.253** | **0.636** |

Geometry reaches benchmark levels at 8–12 s, which is where the residual reaches
zero. That is the mechanism, measured rather than assumed.

Adding origin-only geometry as controls (refit on the Tier 2 train subsample,
applied out of sample):

| specification | <2 s adj | 2–8 s adj | slope t |
|---|---|---|---|
| full validation, M0x | −5.487 | −2.645 | +28.43 |
| + live-window passes removed | −5.481 | −2.644 | +28.41 |
| Tier 2 subsample, M0x | −5.087 | −1.732 | +7.62 |
| **+ origin-only frame geometry at t** | **−2.715** | **−0.939** | **+3.97** |
| + geometry and live windows removed | −2.725 | −0.951 | +3.98 |

Geometry retains 53.4% of the <2 s effect and 54.2% of the 2–8 s effect. The
effect does not vanish, and the attenuation trend survives (t = +3.97), but the
2–8 s band lands at **−0.95 pp, below the pre-registered 1.0 pp relevance floor**.
The defensible claim is therefore the short window, not the long tail.

**Post-treatment leak caught in the control block.** `build._add_360` computes
`ff_lane_opp`, `ff_recv_opp_within_5`, `ff_recv_visible_r5` and `ff_lane_visible`
around `end_location`, which on a failed pass is the interception point. Those are
post-treatment in exactly the way `pass_length` is (caveat 4). Including them adds
+6.996 skill points against +0.066 for origin-only geometry — the gain is largely
the outcome predicting itself. The primary arm uses origin-only features; the
target-derived arm is reported only to bound the leakage.

**Where the control bites.** Origin-only geometry adds almost nothing in aggregate
(+0.066 skill points), which by itself would suggest a toothless control. The
aggregate is the wrong diagnostic: the geometry moves predictions specifically
where the estimand lives. Mean |Δp| is 0.0195 on the benchmark against 0.0304 on
the primary sample under 2 s, and restricted Brier improves by −0.00091 there
against +0.00004 on the benchmark.

### Threat C — unmeasured defender velocity: **cannot be addressed with this data**

Not testable here, and stated rather than buried, because it is the threat a
referee who works with tracking data will raise first.

StatsBomb 360 freeze frames carry **positions only, no velocities**. The threat-B
control therefore knows where defenders are, not where they are going. A defence
that is compact and still closing is a different pressure state from one that is
compact and settled, and the gap between them is largest in the first two seconds
after a press ends, while defenders are still decelerating. The residual that
survives threat B has precisely that shape: concentrated under 2 s, gone by 8 s.

No open-data test separates the two. Closing this would need tracking data, which
is outside the scope of this project by design. Threat C is the reason the causal
reading is retired rather than merely weakened — see Amendment 2.

### What this changes

The residual is **not** a memory trace, and this project no longer claims it is.

- Roughly half is contemporaneous defensive geometry the annotation fails to
  record (threat B, measured).
- Of the surviving half, the part above the relevance floor sits under 2 s, which
  is where annotation-boundary error is concentrated (threat A's qualifier) and
  where unmeasured defender velocity would be largest (threat C, untestable).
- The 2–8 s band — the only window clean of all three — is a **bounded null** at
  −0.939 pp.

The claim the evidence supports is a **measurement** claim: event-level pressure
annotation systematically under-measures defensive pressure, and possession
pressure history is a free Tier 1 correction recovering about half of what
freeze-frame geometry provides. The uncontrolled −5.487 pp and −2.645 pp are the
**size of that measurement gap**, not candidate causal effects. This is a claim
against how EPV/OBSO-class models ingest pressure, it needs no causal
interpretation, and anyone can check it on the open data. See Amendment 2.

### Caveats on the Tier 2 arm

- 77 validation matches, 24,159 primary passes. Power is limited; the 2–8 s band
  is estimated at t ≈ −1.3 to −2.0 per bin.
- The Tier 2 subsample shows a weaker effect than full validation *before* any
  geometry is added (−1.732 against −2.645 in the 2–8 s band). The 360 corpus is
  a non-random subset of competitions, so part of the drop is sample, not control.
- Freeze frames are truncated by `visible_area`. The 5 m origin-visibility gate
  handles the ball carrier; it does not guarantee the wider defensive shape was in
  frame.
- Positions only, no velocity. See threat C.

### Method note: validate the instrument before trusting its null

Threat A returns "no problem" (0.03% contamination). That is only evidence because
the same join was first shown capable of returning "problem": passes the annotator
flagged pressed come back **88.56% inside a live window**, against 0.02% for
unpressed. Without that step, a broken join and a clean result are the same
output. This belongs in the paper — a null from an unvalidated instrument is not a
null, and the check costs one table.

---

## Scope

This repository contains the pressure-memory analysis only. A separate computer
vision pipeline and match simulator exist and are **deliberately excluded**; no
result here depends on them.

Single author. Target venue: MIT Sloan Sports Analytics Conference (SSAC27).

