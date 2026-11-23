# Retraining trigger

`src/taxi/retrain.py` reads a monitoring report and decides.

```bash
make retrain REPORT=reports/monitoring/2020-03-16_v1.json
```

## Rules

1. **Only on `action: retrain`.** On `hold` (data quality) or `none`, it does nothing and says why. `--force`
   overrides it, and the run is tagged `trigger: forced` in MLflow so that it can be found later.
2. **Train on the recent past.** The challenger is trained on the last 14 days of the window minus the last 3
   (`retrain.window_days`, `retrain.holdout_days`). After a break like March 2020, old data describes a world that
   no longer exists.
3. **Compare on the same holdout.** Challenger and champion are both scored on the last 3 days, which neither was
   trained on.
4. **Promote only if better.** The challenger becomes champion only if its holdout MAE is lower; the old champion
   becomes `previous` (see [rollback.md](rollback.md)). Otherwise the challenger stays in the registry, unused.

Every decision is saved (`reports/retrain/`), with both scores.

## Measured (incident replay)

| Alert | Challenger trained on | Holdout | Champion MAE | Challenger MAE | Decision |
|---|---|---|---|---|---|
| 9 to 15 Mar | 2 to 12 Mar | 13 to 15 Mar | 5.31 (v1) | 6.55 (v2) | **keep v1** |
| 16 to 22 Mar | 9 to 19 Mar | 20 to 22 Mar | 1.76 (v1) | 0.99 (v3) | **promote v3** |

The first retraining did *not* help: the 11 days before 13 March were still mostly normal, and v2 learned them.
Rule 4 kept v1. A week later the challenger had seen the collapse, and it cut the error by 44%.

## What would make us change it

- Retraining every week costs about 10 seconds here; with more data, a schedule (or a minimum gap between two
  retrainings) would be needed.
- A 3-day holdout is small: a holiday in it could decide the comparison. With more history we would compare over
  a week.
