# Configurations

**Submission:** `final_config_s3.json` (default of `run_final.sh`): 52 features (36 temporal + 16 cross-table),
XGBoost × 3 seeds + MLP, previous-day outage imputation (`dataset_impD.parquet`), weekly Saturday cutoffs,
joint plug-in decision. Chosen on 2026-11-19 from the 6-week rolling backtest (results/n19*_*.json).

**Experiments of that choice (same tables, 6 Saturdays 10-03 … 11-07):**
`n19s_S1_T0`, `n19s_S2_T`, `n19s_S3_TR` (identical model settings to the submission), `n19s_S4_TRG`;
`n19_{v1,p3,p4}_{W,D}` (full feature sets with the two imputation methods).

**Legacy (earlier design: daily cutoffs every 3 days, label censoring, copy imputation; kept so that old
commands and results can be rerun; not used for the submission):**
`../final_config.json` (v1, the default of `pipeline.load_cfg` when no `--config` is given),
`final_config_v1_baseline*.json`, `final_config_p2*.json`, `final_config_p3*.json`, `final_config_p4*.json`,
`cmp_p3_imp_olddet.json`, `random_search_xgb.json`.

Loading any configuration other than the submission one prints a one-line NOTE (`pipeline.load_cfg`);
nothing else changes.

The public GitHub repository contains only `final_config_s3.json` and `../final_config.json`; the experiment and
legacy configurations above stay in the local working copy.
