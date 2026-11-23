"""Shared fixtures: a small synthetic zone-hour dataset and an isolated MLflow registry per test module.

Nothing here downloads data or touches the real mlflow.db: every module gets its own temporary directory.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

ZONES = {1: 20.0, 2: 10.0, 3: 6.0}  # mean pickups an hour; all three count as "busy" (min_zone_mean = 5)


def synthetic(start="2020-01-01", end="2020-04-01", seed=0, march_factor=1.0) -> pd.DataFrame:
    """Complete hourly grid with a daily cycle; March can be scaled down to simulate a demand collapse."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    rows = []
    for zone, base in ZONES.items():
        cycle = 1 + 0.8 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 6) / 24)
        lam = base * cycle * np.where(hours >= pd.Timestamp("2020-03-01"), march_factor, 1.0)
        rows.append(pd.DataFrame({"zone_id": zone, "hour": hours, "pickups": rng.poisson(lam)}))
    df = pd.concat(rows, ignore_index=True)
    df["zone_id"] = df["zone_id"].astype("int16")
    df["pickups"] = df["pickups"].astype("int32")
    return df


@contextmanager
def isolated(tmp, **extra):
    """Point MLflow, the data file and the registry log to `tmp` for the duration of the block."""
    env = {
        "MLFLOW_TRACKING_URI": f"sqlite:///{tmp / 'mlflow.db'}",
        "TAXI_DATA": str(tmp / "zone_hour.parquet"),
        "REGISTRY_LOG": str(tmp / "registry_log.jsonl"),
        "ADMIN_TOKEN": "test-token",
        "MODEL_NAME": "nyc-taxi-demand-test",
        **extra,
    }
    old_env = {k: os.environ.get(k) for k in env}
    old_cwd = os.getcwd()
    os.environ.update(env)
    os.chdir(tmp)  # MLflow writes its artifacts under ./mlruns
    try:
        yield
    finally:
        os.chdir(old_cwd)
        for k, v in old_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@pytest.fixture(scope="module")
def registry_with_two_versions(tmp_path_factory):
    """A registry with v1 (reference, champion) and v2 (trained on other days, not promoted)."""
    tmp = tmp_path_factory.mktemp("mlflow")
    with isolated(tmp):
        from taxi import registry
        from taxi.train import fit_and_register, load_features

        data = synthetic()
        data.to_parquet(tmp / "zone_hour.parquet", index=False)
        feats = load_features(data)
        v1 = fit_and_register(feats, ("2020-01-08", "2020-02-15"), ("2020-02-16", "2020-02-29"), run_name="reference")
        registry.promote(v1.version, reason="test reference")
        v2 = fit_and_register(feats, ("2020-01-15", "2020-02-22"), ("2020-02-23", "2020-02-29"), run_name="other")
        yield {"tmp": tmp, "data": data, "v1": v1.version, "v2": v2.version}
