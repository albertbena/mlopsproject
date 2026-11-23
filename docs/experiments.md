# Experiment tracking and versioning

## What every MLflow run records

`src/taxi/train.py`, one run per training, in the experiment `nyc-taxi-demand`:

| Kind | Logged |
|---|---|
| Parameters | the model settings (`model_*`), the feature list, the train and validation dates, row counts |
| Metrics | `valid_mae`, `valid_rmse`, `valid_wape`, and the same for two naive baselines (same hour last week, same hour yesterday) |
| Code | tag `git_commit`: the commit the run comes from, with `-dirty` if files were modified |
| Data | `data_md5`: MD5 of `data/processed/zone_hour.parquet`, the same hash as in `dvc.lock`; `train_rows_hash`: a hash of the exact training rows |
| Artifacts | the model (with its input signature and an example), `reference_profile.json` for the drift monitor |

Every run registers a new version of the model `nyc-taxi-demand` in the **MLflow Model Registry**. The version
carries the same tags (`git_commit`, `data_md5`, `train_period`, `valid_mae`, `trigger`), so the registry alone
answers "which code and which data made the model we serve?". `GET /model` on the API returns them too.

The data is versioned by **DVC**: `dvc.yaml` describes the pipeline (download → aggregate → train) and `dvc.lock`
the hash of each input and output. Changing a month in `params.yaml` or a line of `aggregate.py` makes
`dvc repro` re-run only what depends on it.

## Reference model

`HistGradientBoostingRegressor` (scikit-learn) with a Poisson loss, since pickups are counts. It is deliberately
simple: what we grade is how it is operated.

Measured with `dvc repro` (`reports/train_metrics.json`), validation 16 to 29 February 2020:

| Model | MAE (pickups) | RMSE | WAPE |
|---|---|---|---|
| **Gradient boosting (v1)** | **4.43** | **14.45** | **13.2%** |
| Same hour last week (lag 168) | 6.84 | 24.51 | 20.4% |
| Same hour yesterday (lag 24) | 8.78 | 34.02 | 26.2% |

The model has 35% less error than the best naive baseline. WAPE (total absolute error divided by total pickups) is
the metric we compare across periods: when demand falls, the MAE in pickups falls with it, the WAPE does not.

## Browse it

```bash
make mlflow-ui      # http://localhost:5000: runs, metrics, tags, and the registry with its aliases
make status         # champion and previous versions, from the command line
```
