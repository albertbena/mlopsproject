"""Drift monitoring: compare a window of recent hours with the reference the champion was validated on.

Three families of checks, each with a threshold in params.yaml (monitoring):
- data quality: a zone that was busy in the reference and has no pickup at all, or missing hours.
  A broken feed must not be "learned": the action is "hold" (fix the data), never "retrain".
- drift: Population Stability Index (PSI) of the target (pickups) and of the main input (lag_1),
  against decile bins of the reference.
- performance: the champion's WAPE (total absolute error / total pickups) on the window divided by its WAPE
  on the reference. WAPE, not MAE: when demand collapses the MAE in pickups falls with it, the relative error does not.

    python -m taxi.monitor --start 2020-03-16            # 7 days from the 16th, against the champion
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from taxi.config import data_path, model_name, params, path, tracking_uri

DRIFT_COLUMNS = ["pickups", "lag_1"]
EPS = 1e-4


def _edges(values: pd.Series) -> list[float]:
    qs = np.unique(np.quantile(values.dropna(), np.linspace(0.1, 0.9, 9)))
    return [-np.inf, *qs.tolist(), np.inf]


def _proportions(values: pd.Series, edges: list[float]) -> list[float]:
    counts, _ = np.histogram(values.dropna(), bins=edges)
    total = counts.sum()
    return (counts / total).tolist() if total else [0.0] * len(counts)


def psi(expected: list[float], actual: list[float]) -> float:
    """Population Stability Index: sum over bins of (a - e) * ln(a / e)."""
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    return float(np.sum((a - e) * np.log(a / e)))


def reference_profile(frame: pd.DataFrame, pred, mae: float) -> dict:
    """What the monitor needs to remember about the reference period (logged with each model run)."""
    min_mean = params()["monitoring"]["min_zone_mean"]
    zone_mean = frame.groupby("zone_id")["pickups"].mean()
    edges = {c: _edges(frame[c]) for c in DRIFT_COLUMNS}
    return {
        "start": str(frame["hour"].min()), "end": str(frame["hour"].max()),
        "mae": float(mae),
        "wape": float(np.abs(np.asarray(pred) - frame["pickups"].to_numpy()).sum() / frame["pickups"].sum()),
        "mean_pickups": float(frame["pickups"].mean()),
        "mean_prediction": float(np.mean(pred)),
        "edges": {c: [str(x) if not np.isfinite(x) else x for x in e] for c, e in edges.items()},
        "proportions": {c: _proportions(frame[c], edges[c]) for c in DRIFT_COLUMNS},
        "busy_zones": {str(z): round(float(m), 2) for z, m in zone_mean.items() if m >= min_mean},
    }


def _edges_from_profile(profile: dict, column: str) -> list[float]:
    return [float(x) for x in profile["edges"][column]]


def load_model(version: str | None = None):
    """The model of `version` (default: the champion), its version, run id and reference profile."""
    mlflow.set_tracking_uri(tracking_uri())
    client = mlflow.MlflowClient()
    mv = (client.get_model_version(model_name(), version) if version
          else client.get_model_version_by_alias(model_name(), "champion"))
    model = mlflow.sklearn.load_model(f"models:/{model_name()}/{mv.version}")
    profile = mlflow.artifacts.load_dict(f"runs:/{mv.run_id}/reference_profile.json")
    return model, str(mv.version), mv.run_id, profile


def check(model, profile: dict, window: pd.DataFrame, feature_cols: list[str], expected_hours: int) -> dict:
    """Run every check on a window that already has its features. Returns the report (status + action)."""
    cfg = params()["monitoring"]
    hours = int(window["hour"].nunique())
    zone_sum = window.groupby("zone_id")["pickups"].sum()
    silent = sorted(int(z) for z in profile["busy_zones"] if zone_sum.get(int(z), 0) == 0)

    drift = {c: round(psi(profile["proportions"][c],
                          _proportions(window[c], _edges_from_profile(profile, c))), 4) for c in DRIFT_COLUMNS}
    scored = window.dropna(subset=feature_cols)
    pred = model.predict(scored[feature_cols].astype("float64"))
    abs_err = np.abs(pred - scored["pickups"].to_numpy())
    mae = float(abs_err.mean())
    total = float(scored["pickups"].sum())
    wape = float(abs_err.sum() / total) if total else float("inf")
    wape_ratio = wape / profile["wape"] if profile["wape"] else float("inf")

    reasons = []
    if silent:
        reasons.append(f"{len(silent)} busy zone(s) with no pickup at all: {silent[:10]}")
    if hours < expected_hours:
        reasons.append(f"{expected_hours - hours} hour(s) missing from the window")
    quality_ok = not reasons
    for c, v in drift.items():
        if v > cfg["psi_threshold"]:
            reasons.append(f"PSI of {c} = {v:.3f} > {cfg['psi_threshold']}")
    if wape_ratio > cfg["wape_ratio_threshold"]:
        reasons.append(f"WAPE {wape:.1%} is {wape_ratio:.2f}x the reference WAPE {profile['wape']:.1%}")

    if not quality_ok:
        status, action = "data_quality", "hold"   # fix the feed first: retraining would learn the bug
    elif reasons:
        status, action = "drift", "retrain"
    else:
        status, action = "ok", "none"
    return {
        "status": status, "action": action, "reasons": reasons,
        "window": {"start": str(window["hour"].min()), "end": str(window["hour"].max()), "hours": hours,
                   "rows": len(window)},
        "mean_pickups": round(float(window["pickups"].mean()), 3),
        "reference_mean_pickups": round(profile["mean_pickups"], 3),
        "psi": drift,
        "mae": round(mae, 4), "reference_mae": round(profile["mae"], 4),
        "wape": round(wape, 4), "reference_wape": round(profile["wape"], 4), "wape_ratio": round(wape_ratio, 3),
        "silent_zones": silent,
        "thresholds": {k: cfg[k] for k in ("psi_threshold", "wape_ratio_threshold", "min_zone_mean")},
    }


def monitor(start: str, days: int | None = None, version: str | None = None,
            data: pd.DataFrame | None = None, save: bool = True) -> dict:
    """Check `days` days from `start` against a model version (default: the champion) and save the report."""
    from taxi.features import between, feature_names  # local import: train imports this module
    from taxi.train import load_features

    days = days or params()["monitoring"]["window_days"]
    end = (pd.Timestamp(start) + pd.Timedelta(days=days - 1)).strftime("%Y-%m-%d")
    feats = load_features(data)
    window = between(feats, start, end)
    p = params()["features"]
    model, v, run_id, profile = load_model(version)
    report = {"model_version": v, "run_id": run_id,
              **check(model, profile, window, feature_names(p["lags"], p["rolling_hours"]), days * 24)}
    if save:
        out = path("reports") / "monitoring" / f"{start}_v{v}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        report["saved_to"] = str(out.relative_to(path(".")))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first day of the window, YYYY-MM-DD")
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--version", default=None, help="model version (default: the champion)")
    ap.add_argument("--data", type=Path, default=None, help="another zone-hour Parquet file (e.g. an injected one)")
    a = ap.parse_args()
    data = pd.read_parquet(a.data) if a.data else None
    print(json.dumps(monitor(a.start, a.days, a.version, data), indent=2))


if __name__ == "__main__":
    main()
