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

The 8 tables are merged into one point-in-time event log (each row has the time of the action and the
time it became known). Days on which GH Archive lost an activity are detected and filled with the same
span whole weeks earlier (features only). For every daily cutoff t (12:00 UTC), features use only rows
known before t: temporal (T0, T1) and relational (R1 user-centric, R2 repository / collaborator).
Labels are activity in [t+4h, t+4h+7d), the same offset as the test week. The model averages the
probabilities of XGBoost (one model per activity, 3 seeds, depth 4, early stopping), a user–repository
GNN and an MLP; a label-free decision rule then chooses the positives of each activity.

## Files

| Path | Content |
|---|---|
| `run_final.sh` | the whole pipeline: course repo, download, preprocessing, features, prediction, manifest |
| `download_gharchive.sh` | downloads the GH Archive hour files of a period |
| `configs/final_config_p4_imp.json` | the submitted configuration |
| `src/run_preprocess.py` | runs the official `preprocess.py` with more worker processes (same output) |
| `src/common.py`, `src/features.py` | event log, labels, test-user rule; feature groups and outage imputation |
| `src/build_dataset.py`, `src/leakage_audit.py` | training samples and features; point-in-time check |
| `src/modeling.py`, `src/gnn.py`, `src/pipeline.py`, `src/exp.py`, `src/exp2.py` | XGBoost, GNN / MLP, final method |
| `src/predict.py`, `src/manifest.py` | writes and checks `output_<studentID>.csv`; records hashes and versions |

## How to run

Requirements: Linux, Python 3.11, an NVIDIA GPU, about 200 GB of disk for the hour files.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# check nvidia-smi for an idle GPU, then:
./run_final.sh <studentID> <gpu_index> configs/final_config_p4_imp.json 2026-11-21T12:00
```

`run_final.sh` clones the course repository at commit `af9f335`, downloads the missing hour files
into `$RAW` (default `~/gharchive`; it stops if any hour is missing), runs the official preprocessing
on exactly the hours before the cutoff, builds the features, runs the leakage audit, trains the model
and writes `output_<studentID>.csv`, which is checked with the course's `evaluate.py`. A cutoff earlier
than 2026-11-21T12:00 writes a rehearsal file `output_<studentID>_rehearsal_<MMDD>.csv`.
