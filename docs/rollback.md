# Rollback procedure

The API serves `models:/nyc-taxi-demand@champion`. Rolling back means moving the alias, never deleting or
rebuilding a model: every version stays in the MLflow registry with its run, its data hash and its commit.

## When

- The champion was trained on data that failed a data quality check (lineage: its `trigger` and
  `monitoring_window` tags, or its `data_md5`).
- The champion performs worse than the previous version on a clean window.
- Anything unexplained after a promotion. Rolling back is cheap and reversible; investigate afterwards.

## How (about one minute)

```bash
make status                                  # 1. who is champion, who is previous
make rollback REASON="v4 trained on the broken feed of 23-29 March"
                                             # 2. previous -> champion, and the API reloads it
curl localhost:8000/health                   # 3. check the served version
```

To go back to a specific version instead of `previous`:

```bash
.venv/bin/python -m taxi.registry rollback --to 3 --reason "..."
curl -X POST -H "X-Admin-Token: $ADMIN_TOKEN" localhost:8000/admin/reload
```

What happens (`src/taxi/registry.py`):

1. `champion` moves to the target version (default: the `previous` alias).
2. The version we leave becomes `previous`, and gets the tags `rolled_back_at` and `rollback_reason`.
   A rollback can therefore be undone with the same command.
3. The move is appended to `reports/registry_log.jsonl`.
4. `POST /admin/reload` (header `X-Admin-Token`, from `.env`) loads the new champion. If loading fails, the API
   keeps serving the model it had and answers 503 with the reason. If the API is down, it loads the champion at
   its next start.

## Measured

In the incident replay, moving the alias took **0.037 s** (`reports/incident/12_rollback.json`). The integration
test `tests/test_api.py::test_rollback_then_reload_switches_the_served_version` checks the whole path:
promote, reload, rollback, reload, and the version returned by `/predict`.

## After a rollback

- Write down why in the incident report, with the version numbers.
- Fix the cause (the data, the code), then let the retraining trigger produce a new challenger. Never promote the
  rolled-back version again without a new reason.
