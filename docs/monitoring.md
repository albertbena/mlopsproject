# Drift monitoring

`src/taxi/monitor.py` compares a window of recent hours (7 days by default) with the **reference** the
champion was validated on. The reference is saved with each model run (`reference_profile.json` in MLflow), so
every version is judged against its own training conditions.

```bash
make monitor START=2020-03-16      # writes reports/monitoring/2020-03-16_v<version>.json
```

## The checks

| Family | Check | Threshold (`params.yaml`) |
|---|---|---|
| Data quality | a zone with at least 5 pickups an hour in the reference has **no pickup at all** in the window | `min_zone_mean: 5` |
| Data quality | hours missing from the window | any |
| Data drift | PSI of the target (`pickups`) and of the main input (`lag_1`) against the reference deciles | `psi_threshold: 0.2` |
| Performance | WAPE of the champion on the window ÷ its WAPE on the reference | `wape_ratio_threshold: 1.5` |

PSI (Population Stability Index) = Σ (actual − expected) × ln(actual / expected) over the bins. Below 0.1 is
stable, 0.1 to 0.2 a moderate shift, above 0.2 a significant one.

Performance is checked with **WAPE, not MAE**: in March 2020 the MAE of the reference model *fell* (from 4.55 to
2.80 pickups) because demand fell, while its relative error went from 12.9% to 54.1%.

## What the report says to do

| Status | Action | Why |
|---|---|---|
| `ok` | `none` | nothing crossed a threshold |
| `drift` | `retrain` | the world changed and the data is fine: learn the new world |
| `data_quality` | `hold` | the **feed** is broken: retraining would teach the model the bug. Fix the data first |

A data quality problem wins over drift. The retraining trigger ([retraining.md](retraining.md)) only acts on
`retrain`.

## Measured on real weeks (reference model v1)

From the incident replay (`reports/incident/`):

| Week | PSI pickups | WAPE | WAPE ratio | Status |
|---|---|---|---|---|
| 23 to 29 Feb (control) | 0.0004 | 12.9% | 0.97 | ok |
| 9 to 15 Mar | 0.011 | 22.6% | **1.70** | drift → retrain |
| 16 to 22 Mar | **0.318** | 54.1% | **4.09** | drift → retrain |

The performance check fires a week before the distribution check: on 9 to 15 March, demand had fallen by a third
but its shape (deciles) had not changed yet.

## Limits

- The performance check needs the actual pickups, so it runs an hour late at best. Here we replay history, so the
  labels are always there.
- A model's reference is its own validation period. A model trained on broken data has a broken reference and may
  judge itself "ok" (it happened in the incident, step 11). That is why the data quality check comes first and
  why a rollback can be decided on lineage, not on the model's own score.
