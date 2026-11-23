# Every command of the project. Run `make help` for the list.
PY      := .venv/bin/python
PORT    ?= 8000
START   ?= 2020-03-16

.PHONY: help setup pipeline serve test monitor retrain status rollback incident mlflow-ui clean

help:  ## this list
	@grep -E '^[a-z-]+:.*## ' Makefile | awk -F':.*## ' '{printf "  make %-10s %s\n", $$1, $$2}'

setup:  ## virtualenv, pinned dependencies, and a .env with a fresh admin token
	python3 -m venv .venv
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -r requirements.txt
	$(PY) -m pip install -q -e .
	@test -f .env || ( sed "s/^ADMIN_TOKEN=$$/ADMIN_TOKEN=$$($(PY) -c 'import secrets; print(secrets.token_urlsafe(32))')/" .env.example > .env && echo ".env created" )

pipeline:  ## download the TLC data, aggregate it, train and register the reference model (DVC)
	PATH="$(CURDIR)/.venv/bin:$$PATH" .venv/bin/dvc repro

serve:  ## start the API on http://localhost:$(PORT) (docs on /docs)
	.venv/bin/uvicorn taxi.api.app:app --host 0.0.0.0 --port $(PORT)

test:  ## unit and integration tests (no download, temporary MLflow)
	$(PY) -m pytest

monitor:  ## drift report for 7 days from START (default 2020-03-16) against the champion
	$(PY) -m taxi.monitor --start $(START)

retrain:  ## retraining trigger on a monitoring report: make retrain REPORT=reports/monitoring/....json
	$(PY) -m taxi.retrain $(REPORT)

status:  ## which version is champion, which is previous
	$(PY) -m taxi.registry status

rollback:  ## serve the previous version again and reload the API: make rollback REASON="..."
	$(PY) -m taxi.registry rollback --reason "$(or $(REASON),manual rollback)"
	@. ./.env; curl -fsS -X POST -H "X-Admin-Token: $$ADMIN_TOKEN" http://localhost:$(PORT)/admin/reload \
	  && echo || echo "API not running: it will serve the new champion at its next start"

incident:  ## replay the simulated incident in its own MLflow database (reports/incident/)
	$(PY) -m taxi.incident

mlflow-ui:  ## browse runs and the registry on http://localhost:5000
	.venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

clean:  ## remove local MLflow state (keeps the data)
	rm -f mlflow.db mlflow_incident.db reports/registry_log.jsonl
	rm -rf mlruns reports/monitoring reports/retrain
