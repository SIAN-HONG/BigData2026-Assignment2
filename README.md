# BigData 2026 Assignment 2: predicting GitHub user activity

Predict, for each of the 4,465 test users, whether they perform each of six activities (`open_pr`,
`open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`) between 2026-11-22 00:00 and
11-29 00:00 Taiwan time (2026-11-21 16:00 to 11-28 16:00 UTC). The metric is
Score = (macro-F1 + micro-F1) / 2.

## Data Source

The data used in this project is sourced from **GH Archive** (https://www.gharchive.org/), a
publicly available archive of GitHub events. The hourly GH Archive files from 2025-12-01 00:00 UTC
to the prediction cutoff (2026-11-21 12:00 UTC) are turned into the 8 tables by the course's
`scripts/preprocess.py` (https://github.com/nycubigdata-2026fall/BigData2026-Assignment2), which
keeps the 9,000 cohort users and anonymizes the data. No other data is used. This project is for
educational use within the course.

## Method in one paragraph

The 8 tables are merged into one **point-in-time event log**. Each row carries the time the action
happened and the time the row became *known* (`first_seen_at` for PRs, issues and reviews;
`created_at` for comments and pushes; `merged_at`, `closed_at` and `dismissed_at` only once they are
earlier than the cutoff). Days on which GH Archive lost an activity are detected point-in-time (count
< 10 % of the median of the last 28 earlier non-outage days) and filled with that activity's events of
the same span whole weeks earlier (**outage imputation**).
For every daily cutoff t (12:00 UTC) we build features from rows known before t and labels for
[t+4h, t+4h+7d), the same offset as the test week. Users are selected with the rule that reproduces
`test_input.csv` exactly (>= 3 records in the 90 days before t − 17 d). The model averages XGBoost
(one model per activity, 3 seeds; the selected p4 uses shared parameters, depth 4, with early stopping) and two relational neural nets (a
user↔repository GNN and an MLP baseline without message passing). A **label-free decision rule**
then picks, for each activity, the number of positives that maximises the *expected* official score
under the model's own probabilities, which keeps working when the base rate drifts.

## Repository layout

| Path | Content |
|---|---|
| `download_gharchive.sh` | downloads the GH Archive hour files of a period |
| `src/run_preprocess.py` | runs the official `preprocess.py` with more worker processes (same output) |
| `src/common.py` | loading, unified event log, label and test-user rules |
| `src/features.py` | point-in-time feature groups T0 / T1 (temporal), R1 / R2 (relational); outage imputation |
| `src/build_dataset.py` | builds `artifacts/events.parquet`, `dataset[_imp].parquet` (one row per user and daily cutoff from 01-07; 305 cutoffs in the 11-15 snapshot) and `final_features[_imp].parquet` |
| `src/leakage_audit.py` | checks that features from the full tables equal features from tables truncated at t |
| `src/modeling.py` | training, time-based validation, official score, decision rules |
| `src/exp.py`, `src/gnn.py` | one rolling-validation experiment → `results/experiments.tsv`; GNN / MLP |
| `src/pipeline.py` | the final method (`fit_predict`) and `--backtest` of exactly that code path |
| `src/forward_decide.py` | the pre-registered selection rule (`docs/FORWARD_CHECK.md`) |
| `src/sim_push_outage.py` | simulation of missing pushes before the cutoff |
| `src/predict.py` | writes `output_<studentID>.csv` and checks it with the official `evaluate.py` |
| `src/graphs.py`, `src/framework_figure.py`, `src/figures.py` | Q1 graphs, Q4 framework figure, data and result figures |
| `configs/` | `final_config_p4_imp.json` (**selected for submission**: formal decision on the 11-17 snapshot, addendum 7), `final_config_p3_imp.json` (selected by addendum 6, superseded), `final_config_v1_baseline_imp.json` |
| `run_final.sh`, `run_formal.sh`, `backtest_new.sh`, `ablation_c15.sh` | submission, formal version decision (addendum 7), version comparison, corrected ablation (`ablation_new.sh`: earlier exploratory ablation) |
| `results/`, `figures/`, `slides/` | result tables, figures, slide builder |

`assignment2/`, `data/`, `artifacts/` and `.venv/` are not committed.

## How to run

```bash
# 0. environment (Python 3.11; an NVIDIA GPU is used by XGBoost and torch)
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# graphviz `dot` for the figures, e.g.:  conda create -p ./.gv -c conda-forge graphviz

# One command for the submission (steps 1-4 below); check nvidia-smi for an idle GPU first:
./run_final.sh <studentID> <gpu> configs/final_config_p4_imp.json 2026-11-21T12:00

# 1. course repository: preprocess.py, evaluate.py, test_input.csv
git clone https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git assignment2
git -C assignment2 checkout af9f335301d9badb482dfafa10c4534eabac845a      # the 11-07 revision we used
# 2. GH Archive hour files, 2025-12-01 .. cutoff (UTC; reruns only fetch missing hours)
#    stops with an error if any hour of [start, cutoff) is missing (GH Archive answers 404 until published)
./download_gharchive.sh ~/gharchive 2025-12-01 2026-11-21T12:00
# 3. the 8 tables, built by the official preprocess.py -> data/db
#    --raw must hold only the hours before the cutoff (run_final.sh links them into a temporary folder)
#    (src/run_preprocess.py runs the same script with 64 instead of 4 worker processes;
#     python assignment2/scripts/preprocess.py --raw <dir> --out data gives the same files)
.venv/bin/python src/run_preprocess.py --raw <dir with the hours before the cutoff> --out data --workers 64

cd src
# 4. samples + features for every daily cutoff, and features at the prediction cutoff
#    (both rewrite artifacts/events.parquet from data/db; --cutoff stops if the data do not reach it)
../.venv/bin/python build_dataset.py --jobs 16 --cutoff 2026-11-21T12:00
../.venv/bin/python build_dataset.py --impute --jobs 16 --cutoff 2026-11-21T12:00
../.venv/bin/python leakage_audit.py --impute          # must print LEAKAGE AUDIT PASSED
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python predict.py --student-id <studentID> \
    --config configs/final_config_p4_imp.json --cutoff 2026-11-21T12:00   # a missing config is an error
# 4b. manifest of what was used (course commit, hashes of code, tables, outputs and hour files, versions);
#     compare a rerun with ours:  python manifest.py --compare ../results/manifest.json <yours>.json
../.venv/bin/python manifest.py --raw ~/gharchive --cutoff 2026-11-21T12:00 \
    --config configs/final_config_p4_imp.json --hash-hours
# 5. optional: rolling validation of the final method, figures
CUDA_VISIBLE_DEVICES=0 ../.venv/bin/python pipeline.py --backtest --config configs/final_config_p4_imp.json
../.venv/bin/python graphs.py && ../.venv/bin/python framework_figure.py --config configs/final_config_p4_imp.json && ../.venv/bin/python figures.py
```

`run_final.sh` passes preprocess.py only the hour files before the cutoff. `predict.py` writes
`output_<studentID>.csv` in the project root and checks it with
`evaluate.py --pred ... --input assignment2/submission/test_input.csv`. With an earlier cutoff it
writes `output_<studentID>_rehearsal_<MMDD>.csv`, a rehearsal only.

## Validation protocol

* Time-based only. Validation cutoffs are Wednesdays at 12:00 UTC: 05-06, 05-27, 06-17, 07-08, 07-29,
  08-19, 09-09 … 10-07 (development weeks) and 10-14, 10-21, 10-28, 11-04 (fresh weeks; 11-04 added by
  addendum 7). A model validated at
  V trains only on cutoffs ≤ V − 8 days, so every training label window ends before V.
* 09-02 is reported but not averaged (pushes missing 09-02 … 09-04).
* Outage-aware score: an activity with an outage day (count < 10 % of the median of the last 28 earlier non-outage days, `common.outage_flags`) in a
  week's label window is left out of that week's score, as `evaluate.py --skip` does for the test week.
* Thresholds never see validation labels (label-free rule).
* The version to submit is chosen by a rule written before the results (`docs/FORWARD_CHECK.md`,
  addendum 5 for these tables): a candidate must not lose to v1 on the fresh weeks and must win on the
  mean of all weeks.

## Results by dataset snapshot

**Formal decision, 11-17 snapshot** (hours < 2026-11-17 12:00 UTC, `./run_formal.sh`, `results/f17_*_imp.json`,
`results/forward_check_f17.md`, `docs/FORWARD_CHECK.md` addendum 7; leakage audit passed at 5 cutoffs):

| version | outage-aware, 15 weeks | 10-14 * | 10-21 * | 10-28 * | 11-04 * | rule |
|---|---|---|---|---|---|---|
| v1: XGB depth 10 × 3 + GNN + MLP | 48.31 | 43.38 | 40.30 | 40.36 | 41.31 | baseline |
| p3: per-activity XGB parameters | 48.44 | 44.28 | 40.06 | 40.14 | 41.54 | passes |
| **p4: shallow XGB + NN 256** | 48.41 | 43.63 | 40.45 | 40.41 | 43.34 | passes, **chosen** |

\* fresh weeks, outage-aware. p3: fresh mean Δ +0.17, all-week mean Δ +0.13 (t = 0.91, 7 of 15 weeks better);
p4: fresh mean Δ +0.62, all-week mean Δ +0.09 (t = 0.49, 10 of 15 weeks better). Both pass; the tie-break
(larger fresh mean Δ) picks p4. p4's fresh Δ per week is +0.25, +0.15, +0.05, +2.03, so its lead comes from
11-04 (without it p3 and p4 are both +0.15); t compares each candidate with v1, not p4 with p3. The rule was
applied as written; this does not show that p4 is better in general.
This decision is final: later snapshots, including the submission run, do not reselect the version.

**Versions, 11-07 snapshot** (data up to 2026-11-07 11:00 UTC; `PREFIX=a6 ./backtest_new.sh`, `results/a6_*_imp.json`, `results/forward_check_a6.md`;
corrected outage detection, `docs/FORWARD_CHECK.md` addendum 6):

| version | outage-aware, 14 weeks | 10-14 * | 10-21 * | 10-28 * | rule |
|---|---|---|---|---|---|
| v1: XGB depth 10 × 3 + GNN + MLP | 48.75 | 43.57 | 40.36 | 39.90 | baseline |
| **p3: per-activity XGB parameters** | **48.98** | 44.00 | 40.23 | 40.61 | passes, selected by addendum 6 (superseded by the formal decision above) |
| p4: shallow XGB + NN 256 | 48.74 | 43.58 | 40.30 | 39.90 | does not pass |

\* fresh weeks, outage-aware. p3: fresh mean Δ +0.34, all-week mean Δ +0.23 (t = 1.77, 10 of 14 weeks
better). Before the detection fix (`results/forward_check_new.md`) p3 was chosen as well (+0.23 / +0.18).
The week matters much more than the version (p3: 20.95 on 08-19 to 63.24 on 05-06).

**Ablations, corrected, 11-15 snapshot** (`ablation_c15.sh`, `src/ablation_report.py`, `results/ablation_c15.md`):
XGBoost with the v1 tree parameters (depth 10, min_child_weight 50), CPU, 8 weeks 09-09 … 10-28, training
labels censored at each validation cutoff, `dataset_imp.parquet`, 3 seeds averaged in probability, joint
plug-in decisions, outage-aware scoring (push scored in 3 of the 8 weeks).

| feature groups | Score (mean of 8 weeks) |
|---|---|
| T0 counts 7 / 28 / 90 d | 45.56 |
| T0 + T1 rich temporal | 46.80 |
| T0 + T1 + R1 user-centric relational | 47.23 |
| T0 + T1 + R1 + R2 (all) | 47.15 |
| T0 + R1 + R2 (no T1) | 45.46 |

Paired by week: adding T1 to T0 +1.23 (5 of 8 weeks better; +0.23 without the largest week, 09-09);
adding R1 +0.44 (7 of 8; +0.35 without the largest week); adding R2 −0.08 (2 of 8). Gains are
conditional on the groups already present and do not add up. Ensemble members of the p3 backtest on the
same weeks (11-15 snapshot): trees 47.44, GNN 46.27, MLP baseline 46.30, ensemble 47.61, so the GNN's
message passing did not beat the MLP baseline here; this does not show that message passing is useless
in general.

**Decision rule, 11-15 snapshot** (`src/threshold_baseline.py`, `results/threshold_baseline_b15_p3_imp.md`, p3 backtest,
6 weeks 09-23 … 10-28): the label-free joint plug-in rule 45.00 against 41.33 for thresholds taken from
the latest past week whose labels are mature at the cutoff (better in 4 of 6 weeks; +0.84 without the
largest week, so most of the gain comes from one week).

The earlier exploratory ablation (`ablation_new.sh`, ledger rows `new_*`) did not censor the training
labels at the validation cutoff and used a threshold baseline that is not available at prediction time;
it is superseded by the tables above.

**Missing pushes before the cutoff** (`src/sim_push_outage.py`, 4 weeks 07-29, 09-23, 09-30, 10-07;
pushes of the N days before the cutoff deleted, change of the score of the other 5 activities;
`results/sim_push_outage_{new,a6}_*.csv`). Pushes have been missing since 10-27, so about 25 days
will be missing at the 11-21 cutoff:

| pushes missing | 3 d | 7 d | 14 d | 21 d | 28 d |
|---|---|---|---|---|---|
| no imputation | −0.31 | −13.96 | −7.05 | −2.00 | −2.11 |
| imputation, detection before the fix | −0.87 | −1.13 | −2.75 | −7.73 | −3.45 |
| **imputation, corrected detection** | −0.66 | −0.98 | −1.13 | −0.95 | −2.02 |

The first detection compared a day with the median of the 28 days before it, outage days included;
once more than half of them were missing the median fell to ~0 and later outage days were neither
detected nor filled. The corrected rule uses only earlier non-outage days (addendum 6). Without
imputation the loss shrinks again for very long holes, as the latest training cutoffs then see the
same empty push history, but it is large for 7–14 days.

**Rehearsal** at the 11-06 12:00 UTC cutoff (`predict.py`, p3 with corrected detection, test users):
official format check passes (4,465 users). Mean predicted probability of the 5 non-push activities: 2.9–7.3 %
(open_pr 7.3 %), against 12.4 % for open_pr without imputation (`results/rehearsal_1106/`). These are
mean probabilities, not evidence of calibration.

## Leakage prevention

* Every row is used only if its availability time is < t (row-time column; "known only later"
  columns only once they are earlier than t).
* `src/leakage_audit.py` rebuilds the tables as they would have been at t (rows removed, late columns
  nulled) and recomputes all features with outage imputation: 0 columns differ at 05-13, 09-23, 10-28,
  11-04 and 11-06 (`logs/a6_leakage_audit_5cut.log`).
* Limits of this check: it truncates the final tables; it does not rerun preprocess.py on the raw hours
  before t. preprocess.py keeps the latest login and repository owner, which the relation features (R2)
  use, so those two fields are the values at the end of the data, not at t. The outage-imputed events are
  used for the tabular features only; the GNN edges come from the raw event log.
* Training labels need a complete 7-day window: validation models train on cutoffs ≤ V − 8 d; the
  submission trains on cutoffs ≤ 11-13.

## Earlier work on the course's previous data releases

Before the course update of 11-07 the data came as daily releases of ready-made tables. All
development (`results/phase2/`, `reh/`, `fwd/`, ledger rows without the `new_` prefix) was done on those
releases; the course asked to use only the newly preprocessed data, so every number above was
recomputed on the GH Archive tables and the old numbers are not used for any decision.
