# BigData 2026 Assignment 2: predicting GitHub user activity

For each of the 4,465 test users, predict whether they perform each of six activities (`open_pr`,
`open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`) between 2026-11-22 00:00 and
11-29 00:00 Taiwan time (2026-11-21 16:00 to 11-28 16:00 UTC).

## Data Source

The data used in this project is sourced from **GH Archive** (https://www.gharchive.org/), a
publicly available archive of GitHub events. The hourly GH Archive files from 2025-12-01 00:00 UTC
to the prediction cutoff (2026-11-21 12:00 UTC, exclusive) are turned into the 8 tables by the course's
`scripts/preprocess.py` (https://github.com/nycubigdata-2026fall/BigData2026-Assignment2), which
keeps the 9,000 cohort users and anonymizes the data. No other data is used. This project is for
educational use within the course.

## Method

* **Point-in-time event log.** The 8 tables are merged into one event log; each row has the time of
  the action and the time it became known. Features at a cutoff t use only rows known before t.
* **Samples.** One cutoff per week, Saturday 12:00 UTC (20:00 Taiwan), the weekday and hour of the
  official cutoff, from 2026-03-21 (110.5 days after the data start; 107 are needed for the 90-day
  features and the test-user rule). Labels = activity in [t+4h, t+4h+7d), the same offset as the
  test week. Users are selected with the rule that reproduces `test_input.csv`.
* **Collection outages.** A day whose count of an activity is < 10 % of the median of the last 28
  earlier normal days is an outage; it is filled with that user's events of the day before the outage
  (features only, point-in-time).
* **52 features.** Temporal: counts in 7 / 28 / 90 days, days since the last action, 14-day decayed
  count, active weeks out of 12 (36). Cross-table, user-centric (7, 28-day counts unless noted):
  comments, reviews and line comments by others on the user's own issues / PRs; PRs the user merged as
  maintainer; the user's own PRs that were merged; own open PRs and issues (first seen in the last 90
  days). Cross-table, repositories and collaborators (9): number of repositories, share of activity in
  own repositories, number of collaborators, 7-day activity of the user's repositories (by type).
* **Model.** XGBoost only: one model per activity, depth 3, learning rate 0.01, 1,000 trees, subsample and
  colsample 0.7, min_child_weight 50, 3 seeds. The seeds' probabilities are averaged and a label-free decision
  rule chooses, for each activity, the number of positives that maximises the expected score.
* **Validation.** Rolling, the last 6 labelled Saturdays: a model validated at V trains only on cutoffs
  ≤ V − 14 days, whose label windows (+ 5 days for late rows) end before V. On these weeks we compared full
  and simplified feature sets, then an XGBoost grid (depth, learning rate, number of trees) and a deeper
  user–repo GNN. The final model scored 44.80 (six activities) and 42.06
  (without the activities that have a collection outage in the label window). The full 217-feature model
  scored higher on six activities (45.51); we chose the simpler one after seeing the results, so the
  comparison is not a significance test.

## Files

| Path | Content |
|---|---|
| `run_final.sh` | the whole pipeline: course repo, download, preprocessing, features, leakage audit, prediction, manifest |
| `download_gharchive.sh` | downloads the GH Archive hour files of a period |
| `configs/final_config_xgb_d3.json` | the submitted configuration |
| `src/run_preprocess.py` | runs the official `preprocess.py` with more worker processes (same output) |
| `src/common.py`, `src/features.py` | event log, labels, test-user rule; features and outage filling |
| `src/build_dataset.py`, `src/leakage_audit.py` | weekly training samples and features; point-in-time check |
| `src/modeling.py`, `src/gnn.py`, `src/pipeline.py`, `src/exp.py`, `src/exp2.py` | XGBoost, MLP, the final method and its rolling backtest |
| `src/predict.py`, `src/manifest.py` | writes and checks `output_<studentID>.csv`; records hashes and versions |

## How to run

Requirements: Linux, Python 3.11, git, wget, an NVIDIA GPU, about 200 GB of disk for the hour files.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# check nvidia-smi for an idle GPU, then:
./run_final.sh <studentID> <gpu_index> configs/final_config_xgb_d3.json 2026-11-21T12:00
```

`run_final.sh` clones the course repository at commit `af9f335`, downloads the missing hour files
into `$RAW` (default `~/gharchive`; it stops if any hour is missing), runs the official preprocessing
on exactly the hours before the cutoff, builds the features, runs the leakage audit, trains the model
and writes `output_<studentID>.csv`, which is checked with the course's `evaluate.py`. A cutoff earlier
than 2026-11-21T12:00 writes a rehearsal file `output_<studentID>_rehearsal_<MMDD>.csv`.
At the official cutoff, `predict.py` stops with an error if `assignment2/submission/test_input.csv` is missing,
malformed or not the pinned course file (4,465 users, checked by SHA-256), or if a test user has no features;
it never falls back to its own user selection. The written file is re-read and must have the header
`UserID` + 6 activities, only 0/1 values, and the test users in the same order.

Rolling backtest of the final configuration on tables built the same way (in `src/`):

```bash
CUDA_VISIBLE_DEVICES=<gpu> ../.venv/bin/python pipeline.py --backtest --config configs/final_config_xgb_d3.json --val last6
```
