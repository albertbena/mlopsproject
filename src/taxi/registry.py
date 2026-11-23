"""The MLflow Model Registry: which version serves (alias "champion"), and how to go back.

- promote(v): the current champion becomes "previous", v becomes "champion".
- rollback(): "previous" becomes "champion" again, and the version we leave becomes "previous"
  (so a rollback can itself be undone). Nothing is deleted: every version stays in the registry.

Every move is appended to reports/registry_log.jsonl. The API serves models:/<name>@champion, so after a
move it only needs POST /admin/reload (or a restart).

    python -m taxi.registry status
    python -m taxi.registry promote 3 --reason "..."
    python -m taxi.registry rollback --reason "..."        # or --to 2
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from taxi.config import model_name, path, tracking_uri

CHAMPION = "champion"
PREVIOUS = "previous"


def client() -> MlflowClient:
    mlflow.set_tracking_uri(tracking_uri())
    return MlflowClient()


def _alias(alias: str) -> str | None:
    try:
        return str(client().get_model_version_by_alias(model_name(), alias).version)
    except MlflowException:
        return None


def champion() -> str | None:
    return _alias(CHAMPION)


def previous() -> str | None:
    return _alias(PREVIOUS)


def tag_version(version: str, tags: dict) -> None:
    c = client()
    for k, v in tags.items():
        c.set_model_version_tag(model_name(), str(version), k, str(v))


def _log(event: dict) -> None:
    event = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **event}
    log = Path(os.environ.get("REGISTRY_LOG", path("reports") / "registry_log.jsonl"))
    log.parent.mkdir(exist_ok=True)
    with open(log, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def promote(version: str, reason: str) -> dict:
    """Make `version` the champion; the old champion becomes "previous"."""
    c = client()
    version = str(version)
    c.get_model_version(model_name(), version)  # fails clearly if the version does not exist
    old = champion()
    if old == version:
        return {"action": "promote", "champion": version, "previous": previous(), "changed": False}
    if old is not None:
        c.set_registered_model_alias(model_name(), PREVIOUS, old)
    c.set_registered_model_alias(model_name(), CHAMPION, version)
    tag_version(version, {"promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "promotion_reason": reason})
    event = {"action": "promote", "champion": version, "previous": old, "reason": reason, "changed": True}
    _log(event)
    return event


def rollback(reason: str, to: str | None = None) -> dict:
    """Serve `to` (default: the "previous" alias) again. The version we leave is tagged and kept."""
    c = client()
    current = champion()
    target = str(to) if to else previous()
    if target is None:
        raise RuntimeError("nothing to roll back to: no 'previous' version and no --to given")
    if target == current:
        raise RuntimeError(f"version {target} is already the champion")
    c.get_model_version(model_name(), target)
    c.set_registered_model_alias(model_name(), CHAMPION, target)
    if current is not None:
        c.set_registered_model_alias(model_name(), PREVIOUS, current)
        tag_version(current, {"rolled_back_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "rollback_reason": reason})
    event = {"action": "rollback", "champion": target, "previous": current, "reason": reason, "changed": True}
    _log(event)
    return event


def status() -> dict:
    c = client()
    versions = sorted(c.search_model_versions(f"name='{model_name()}'"), key=lambda v: int(v.version))
    return {
        "model": model_name(),
        CHAMPION: champion(),
        PREVIOUS: previous(),
        "versions": [{"version": v.version, "run_id": v.run_id,
                      **{k: v.tags[k] for k in ("trigger", "train_period", "valid_mae", "data_md5") if k in v.tags}}
                     for v in versions],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("promote")
    p.add_argument("version")
    p.add_argument("--reason", required=True)
    r = sub.add_parser("rollback")
    r.add_argument("--to", default=None)
    r.add_argument("--reason", required=True)
    a = ap.parse_args()
    if a.cmd == "status":
        out = status()
    elif a.cmd == "promote":
        out = promote(a.version, a.reason)
    else:
        out = rollback(a.reason, a.to)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
