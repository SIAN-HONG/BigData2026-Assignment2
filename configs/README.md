# Configurations

**Submission:** `final_config_xgb_d3.json` (default of `run_final.sh`): the 52 S3 features (36 temporal + 16
cross-table), XGBoost only (depth 3, learning rate 0.01, 1,000 trees, subsample / colsample 0.7, min_child_weight 50,
3 seeds), previous-day outage imputation (`dataset_impD.parquet`), weekly Saturday cutoffs, joint plug-in decision.
Chosen on 2026-11-20 from the XGBoost grid (`src/grid_xgb.py`, results/grid_xgb*.csv, grid2_xgb_s*.csv) and the GNN
grid (`src/hgnn.py`, results/hgnn*_summary.csv); backtest results/n19x_D3.json (44.80 / outage-aware 42.06).

**Previous submission candidate:** `final_config_s3.json` (same features, XGBoost depth 10 × 3 + MLP; 44.59 / 42.00).

**Experiments of that choice (same tables, 6 Saturdays 10-03 … 11-07):**
`n19s_S1_T0`, `n19s_S2_T`, `n19s_S3_TR` (identical to `final_config_s3.json`), `n19x_D3` (identical to the submission), `n19s_S4_TRG`;
`n19_{v1,p3,p4}_{W,D}` (full feature sets with the two imputation methods).

**Legacy (earlier design: daily cutoffs every 3 days, label censoring, copy imputation; kept so that old
commands and results can be rerun; not used for the submission):**
`../final_config.json` (v1, the default of `pipeline.load_cfg` when no `--config` is given),
`final_config_v1_baseline*.json`, `final_config_p2*.json`, `final_config_p3*.json`, `final_config_p4*.json`,
`cmp_p3_imp_olddet.json`, `random_search_xgb.json`.

Loading any configuration other than the submission one prints a one-line NOTE (`pipeline.load_cfg`);
nothing else changes.

The public GitHub repository contains only `final_config_xgb_d3.json`, `final_config_s3.json` and `../final_config.json`; the experiment and
legacy configurations above stay in the local working copy.
