"""Train the demand model, log everything to MLflow and register it in the Model Registry.

Each run logs its parameters, its metrics (and those of two naive baselines), the git commit, the hash of the
data file and of the exact training rows, and a reference profile used later by the drift monitor.

    python -m taxi.train            # the reference model; it becomes the champion if there is none yet
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingRegressor

from taxi import registry, tracking
from taxi.config import data_path, model_name, params, path
from taxi.features import add_features, between, feature_names
from taxi.monitor import reference_profile


@dataclass
class TrainResult:
    version: str
    run_id: str
    metrics: dict


def metrics(y_true, y_pred) -> dict:
    """MAE and RMSE in pickups; WAPE = total absolute error / total pickups (comparable across periods)."""
    y_true = np.asarray(y_true, dtype=float)
    err = np.asarray(y_pred, dtype=float) - y_true
    total = y_true.sum()
    return {
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err ** 2).mean())),
        "wape": float(np.abs(err).sum() / total) if total else float("nan"),
    }


def load_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    p = params()["features"]
    if df is None:
        df = pd.read_parquet(data_path())
    return add_features(df, p["lags"], p["rolling_hours"])


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    p = params()["features"]
    cols = feature_names(p["lags"], p["rolling_hours"])
    frame = frame.dropna(subset=cols)
    return frame[cols].astype("float64"), frame["pickups"]


def fit_and_register(feats: pd.DataFrame, train: tuple[str, str], valid: tuple[str, str], *,
                     run_name: str, tags: dict | None = None, data_md5: str = "") -> TrainResult:
    """Fit on the train days, score on the valid days, log the run and register a new model version."""
    cfg = params()["model"]
    tracking.setup()
    X_tr, y_tr = xy(between(feats, *train))
    X_va, y_va = xy(between(feats, *valid))
    if X_tr.empty or X_va.empty:
        raise ValueError(f"no rows to train on {train} or validate on {valid}")

    model = HistGradientBoostingRegressor(
        loss=cfg["loss"], max_iter=cfg["max_iter"], learning_rate=cfg["learning_rate"],
        max_leaf_nodes=cfg["max_leaf_nodes"], random_state=cfg["random_state"],
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_va)
    m = {f"valid_{k}": v for k, v in metrics(y_va, pred).items()}
    for lag in (168, 24):
        m.update({f"baseline_lag{lag}_{k}": v for k, v in metrics(y_va, X_va[f"lag_{lag}"]).items()})

    commit = tracking.git_commit()
    rows_hash = tracking.frame_hash(between(feats, *train)[["zone_id", "hour", "pickups"]])
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params({
            **{f"model_{k}": v for k, v in cfg.items() if k != "name"},
            "features": ",".join(X_tr.columns),
            "train_start": train[0], "train_end": train[1],
            "valid_start": valid[0], "valid_end": valid[1],
            "train_rows": len(X_tr), "valid_rows": len(X_va),
            "data_md5": data_md5 or "n/a", "train_rows_hash": rows_hash,
        })
        mlflow.set_tags({"git_commit": commit, "data_md5": data_md5 or "n/a", "train_rows_hash": rows_hash,
                         **(tags or {})})
        mlflow.log_metrics(m)
        mlflow.log_dict(reference_profile(between(feats, *valid), pred, m["valid_mae"]), "reference_profile.json")
        info = mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=model_name(),
            signature=infer_signature(X_va.head(50), pred[:50]), input_example=X_va.head(3),
        )
    version = str(info.registered_model_version)
    registry.tag_version(version, {"git_commit": commit, "data_md5": data_md5 or "n/a",
                                   "train_period": f"{train[0]}..{train[1]}", "valid_mae": f"{m['valid_mae']:.3f}",
                                   **(tags or {})})
    return TrainResult(version=version, run_id=run.info.run_id, metrics=m)


def main() -> None:
    p = params()["periods"]
    feats = load_features()
    md5 = tracking.file_md5(data_path())
    result = fit_and_register(feats, tuple(p["train"]), tuple(p["valid"]), run_name="reference",
                              tags={"trigger": "reference"}, data_md5=md5)
    if registry.champion() is None:
        registry.promote(result.version, reason="first model, trained on the reference period")
    out = {"model_version": result.version, "run_id": result.run_id, "data_md5": md5,
           **{k: round(v, 4) for k, v in result.metrics.items()}}
    path("reports").mkdir(exist_ok=True)
    (path("reports") / "train_metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
