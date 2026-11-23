# Incident report: the week the city stopped, and the feed that went quiet

_Simulated on real NYC TLC data (January to April 2020) plus one injected fault. Every number below comes from
`reports/incident/`, written by `make incident` (`src/taxi/incident.py`). The replay takes about 30 seconds and
runs in its own MLflow database (`mlflow_incident.db`), so it never touches the registry the API serves from._

## Summary

| | |
|---|---|
| **What broke** | 1) Real drift: taxi demand fell by 34%, then 77% (COVID-19, March 2020). 2) Injected fault: three busy zones reported no pickup for a week. |
| **Detected by** | 1) the performance check (WAPE 1.70× the reference), then the PSI. 2) the data quality check (busy zones silent). |
| **Reaction** | 1) Retraining trigger: first challenger rejected, second promoted. 2) Trigger refused to retrain; a forced retrain was promoted anyway, then **rolled back in 0.04 s**. |
| **Lesson** | Retrain on drift, never on a broken feed; decide rollbacks on lineage, because a model trained on bad data judges itself by bad data. |

## Timeline

| Step | Week (2020) | What happened | Serving |
|---|---|---|---|
| 1 | 8 Jan to 29 Feb | v1 trained and validated on the reference period: MAE 4.43, WAPE 13.2% (naive "same hour last week": MAE 6.84). Promoted to champion. | v1 |
| 2 | 23 to 29 Feb | Control week: PSI 0.0004, WAPE 12.9% (0.97× the reference). **ok**. | v1 |
| 3 | 9 to 15 Mar | Demand −34%. PSI still low (0.011), but WAPE 22.6% = **1.70×** the reference > 1.5. **drift → retrain**. | v1 |
| 4 | | Challenger v2 trained on 2 to 12 March, holdout 13 to 15 March: v2 MAE **6.55**, v1 MAE **5.31**. v2 is worse: **v1 kept**. | v1 |
| 5 | 16 to 22 Mar | Demand −77%. PSI of pickups **0.318** > 0.2, WAPE 54.1% = **4.09×**. **drift → retrain**. | v1 |
| 6 | | Challenger v3 trained on 9 to 19 March, holdout 20 to 22 March: v3 MAE **0.99**, v1 MAE **1.76**. **v3 promoted**, v1 becomes `previous`. | v3 |
| 7 | 23 to 29 Mar | **Injected fault:** zones 140, 236 and 238 (Upper East Side and Upper West Side, the three busiest that week) report 0 pickups all week. Monitoring: **data_quality → hold**, "3 busy zone(s) with no pickup at all: [140, 236, 238]". | v3 |
| 8 | | Retraining trigger on that report: **no_retrain** ("monitoring action is 'hold'"). | v3 |
| 9 | | Human override: `retrain --force` on the broken week. v4 trained on 16 to 26 March, holdout 27 to 29 March, both broken. On that broken holdout v4 looks better (MAE 0.495 vs 0.519), so it is **promoted**. | v4 |
| 10 | 30 Mar to 5 Apr | The feed is fixed. On clean data, v4: MAE 0.603, WAPE **45.5%**; v3: MAE 0.580, WAPE **43.8%**. v4's own monitoring says "ok" (1.04×), because its reference is the broken holdout. | v4 |
| 11 | | **Rollback** to v3, the last model trained on clean data: `registry.rollback(to=3)`, 0.037 s. v4 is tagged `rolled_back_at` and becomes `previous`. | v3 |
| 12 | 30 Mar to 5 Apr | Monitoring with v3: **ok**, no silent zone. | v3 |

## Root causes

1. **The world changed** (steps 3 to 6). A model trained on January and February had never seen an empty city.
   Its relative error quadrupled. That is drift: retraining on recent data is the right answer, and it worked once
   the recent data contained the new world (step 6).
2. **The feed broke** (steps 7 to 11). Three zones reporting zero is not a change in the city; it is a bug
   upstream. Retraining on it was a human error, made possible by `--force`.

## What worked

- The **performance check caught the drift a week before the PSI did** (step 3), because demand fell before
  its shape changed.
- The trigger **did not promote blindly**: in step 4 the challenger was worse, and v1 stayed.
- The **data quality check stopped the automatic retrain** (step 8): the broken week never reached a model
  through the normal path.
- The rollback was **one command, 0.037 s, and reversible**: v4 is still in the registry, tagged with why it
  was rolled back.

## What did not, and what we changed or would change

- **A model judges itself by its own reference.** v4 measured against its broken holdout said "ok" (step 10).
  The rollback was decided on **lineage** (v4's `trigger: forced` tag and its `monitoring_window` that failed the
  quality check), not on v4's score. Next step: compare every new champion with `previous` on the first clean
  week, automatically.
- **`--force` should leave a trace before it acts.** Today it is tagged in MLflow after the fact; we would
  require a reason (as `rollback` does) and refuse to promote a forced model without a second person.
- The holdout of a forced retrain is as broken as its training data. A fixed clean "canary" set (for example
  the last clean week) would have shown v4 was worse before it was promoted.

## Reproduce

```bash
make incident          # rebuilds reports/incident/ from scratch (needs `make pipeline` first, for the data)
```
