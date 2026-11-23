# NYC taxi demand: the MLOps layer

Predict the number of yellow taxi pickups in each New York zone for the next hour, and operate that model so that
anyone can say **which data and which code trained it, whether it still works, and how to undo it**.

The model is deliberately simple (gradient boosting on lags and calendar features). The project is the layer
around it: experiment tracking, data and model versioning, a tested API, drift monitoring, a retraining trigger,
a rollback procedure, and a simulated incident.

Data: **NYC Taxi & Limousine Commission (TLC) Trip Record Data**,
<https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page> (yellow taxis, January to June 2020).

## Quick start (fresh clone)

Needs Python 3.11 to 3.13, `make` and about 500 MB of disk space.

```bash
git clone <this repo> && cd mlops-nyc-taxi
make setup        # .venv, pinned dependencies, and a .env with a random admin token
make pipeline     # DVC: download 6 TLC files (250 MB), aggregate, train, register v1 as champion (~1 min)
make serve        # API on http://localhost:8000 (interactive docs on /docs)
```

In another terminal:

```bash
curl localhost:8000/health
curl localhost:8000/model
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d '{"zone_id": 237, "hour": "2020-02-20T18:00:00"}'
```

Then the operations:

```bash
make test                       # 30 unit and integration tests, no download needed
make monitor START=2020-03-16   # drift report for one week
make incident                   # replay the full incident (drift, retrain, broken feed, rollback)
make mlflow-ui                  # runs, metrics, tags and the registry on http://localhost:5000
make demo                       # live demo of the whole loop: predict, drift, retrain, swap, roll back (docs/demo.md)
make demo-ui                    # the same demo as a web page with buttons, on http://localhost:8501/demo
make present                    # the defence: the slides with the live demo inside (presentation/)
make help                       # every command
```

## What is where

| Requirement | How | Files |
|---|---|---|
| Experiment tracking | each run logs params, metrics, baselines, the **git commit** and the **data hash** to MLflow | `src/taxi/train.py`, `src/taxi/tracking.py`, [docs/experiments.md](docs/experiments.md) |
| Data versioning | download script + DVC pipeline; raw files never committed, hashes in `dvc.lock` | `src/taxi/data/`, `dvc.yaml`, `dvc.lock`, [docs/data.md](docs/data.md) |
| Model versioning | MLflow Model Registry, aliases `champion` and `previous` | `src/taxi/registry.py` |
| API + integration tests | FastAPI serving `@champion`, reload without restart; tests against a real temporary registry | `src/taxi/api/`, `tests/test_api.py` |
| Drift monitoring | data quality, PSI, relative error (WAPE) against each model's own reference | `src/taxi/monitor.py`, [docs/monitoring.md](docs/monitoring.md) |
| Retraining trigger | acts on a monitoring report; promotes a challenger only if it beats the champion | `src/taxi/retrain.py`, [docs/retraining.md](docs/retraining.md) |
| Rollback | one command, reversible, logged | `make rollback`, [docs/rollback.md](docs/rollback.md) |
| Incident report | real drift (March 2020) + an injected broken feed, detected and handled | `src/taxi/incident.py`, `reports/incident/`, [docs/incident_report.md](docs/incident_report.md) |

Results in one line: v1 has a WAPE of **13.2%** on the reference period (naive baseline 20.4%). In March 2020 the
monitor flags the drift, the trigger replaces v1 with a model that cuts the error by 44%, refuses to retrain on
an injected broken feed, and a forced bad model is rolled back in 0.04 s.

## Rules we follow

- **Split by time, never at random.** Reference period: January and February 2020. Drift period: March to June.
- **Secrets live in `.env`**, never in Git (`.env.example` lists them; `make setup` generates the admin token).
- **Raw data is never committed**: `make pipeline` (`dvc repro`) rebuilds it from the TLC.

## Configuration

| File | What |
|---|---|
| `params.yaml` | months, periods, features, model settings, monitoring thresholds, retraining window |
| `.env` | `MLFLOW_TRACKING_URI` (default: local `mlflow.db`), `ADMIN_TOKEN` for `POST /admin/reload` |

## API

| Method | Path | |
|---|---|---|
| GET | `/health` | status and served version |
| GET | `/model` | version, run id, git commit, data hash, training period of the champion |
| POST | `/predict` | `{"zone_id": 237, "hour": "2020-02-20T18:00:00"}` → predicted pickups (and the actual value when known) |
| POST | `/admin/reload` | reload the champion after a promotion or a rollback (header `X-Admin-Token`) |

## Team

| Member | Part |
|---|---|
| Jules Faucher | project setup, data download and aggregation, DVC pipeline |
| Eduardo Pepe | features, training and MLflow tracking, experiment results |
| Rudy Causel | model registry, retraining trigger, rollback procedure |
| Adrien Lory | FastAPI service, integration tests, Makefile and CI |
| Abdel BenAbid | drift monitoring, incident simulation and report |
