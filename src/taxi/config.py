"""Settings: params.yaml for the pipeline, .env for the machine (MLflow URI, admin token).

Secrets only ever come from the environment (.env, never committed). Nothing here has a secret default.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")


@lru_cache(maxsize=1)
def params() -> dict:
    """The content of params.yaml (path overridable with TAXI_PARAMS, for the tests)."""
    path = Path(os.environ.get("TAXI_PARAMS", ROOT / "params.yaml"))
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(relative: str) -> Path:
    """A path from params.yaml, relative to the repository root."""
    p = Path(relative)
    return p if p.is_absolute() else ROOT / p


def tracking_uri() -> str:
    """MLflow backend: a local SQLite file by default, so the Model Registry works with no server."""
    return os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{ROOT / 'mlflow.db'}")


def model_name() -> str:
    return os.environ.get("MODEL_NAME", params()["model"]["name"])


def admin_token() -> str:
    """Token for POST /admin/reload. Empty means the endpoint is disabled."""
    return os.environ.get("ADMIN_TOKEN", "").strip()


def data_path() -> Path:
    return Path(os.environ.get("TAXI_DATA", path(params()["data"]["processed"])))
