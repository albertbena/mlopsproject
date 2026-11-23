"""FastAPI service: serves the champion model of the MLflow Model Registry.

The model is loaded once at start-up (models:/<name>@champion). After a promotion or a rollback,
POST /admin/reload swaps it without a restart; the endpoint needs the X-Admin-Token header (ADMIN_TOKEN
in .env) and is disabled when no token is set.

The recent hours a prediction needs are read from the processed zone-hour file (data/processed), which plays
the role of a feature store here: a client only sends a zone and an hour.

    uvicorn taxi.api.app:app --port 8000
"""
from __future__ import annotations

import hmac
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow
import pandas as pd
from fastapi import FastAPI, Header, HTTPException

from taxi import __version__
from taxi.api.schemas import Health, ModelInfo, Prediction, PredictRequest
from taxi.config import admin_token, data_path, model_name, params, tracking_uri
from taxi.features import add_features, feature_names


@dataclass
class Served:
    model: object
    version: str
    run_id: str
    tags: dict


class State:
    served: Served | None = None
    features: pd.DataFrame | None = None
    error: str | None = None
    lock = threading.Lock()


state = State()


def load_champion() -> Served:
    mlflow.set_tracking_uri(tracking_uri())
    mv = mlflow.MlflowClient().get_model_version_by_alias(model_name(), "champion")
    model = mlflow.sklearn.load_model(f"models:/{model_name()}/{mv.version}")
    return Served(model=model, version=str(mv.version), run_id=mv.run_id, tags=dict(mv.tags))


def load_features() -> pd.DataFrame:
    p = params()["features"]
    df = add_features(pd.read_parquet(data_path()), p["lags"], p["rolling_hours"])
    return df.set_index(["zone_id", "hour"]).sort_index()


def reload() -> None:
    """Load the champion and the features; on failure keep serving what was loaded before."""
    with state.lock:
        try:
            served = load_champion()
            features = state.features if state.features is not None else load_features()
        except Exception as err:  # noqa: BLE001 - reported by /health, never as a stack trace
            state.error = f"{type(err).__name__}: {err}"[:300]
            raise
        state.served, state.features, state.error = served, features, None


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        reload()
    except Exception:  # noqa: BLE001 - the service starts, /health says why it cannot predict
        pass
    yield


app = FastAPI(title="NYC taxi demand", version=__version__, lifespan=lifespan)


def _served() -> Served:
    if state.served is None or state.features is None:
        raise HTTPException(503, detail=f"no model loaded: {state.error or 'run the pipeline first (make pipeline)'}")
    return state.served


@app.get("/health", response_model=Health)
def health() -> Health:
    s = state.served
    return Health(status="ok" if s and state.features is not None else "unavailable",
                  model_version=s.version if s else None, error=state.error)


@app.get("/model", response_model=ModelInfo)
def model_info() -> ModelInfo:
    s = _served()
    return ModelInfo(name=model_name(), version=s.version, run_id=s.run_id,
                     git_commit=s.tags.get("git_commit"), data_md5=s.tags.get("data_md5"),
                     train_period=s.tags.get("train_period"), valid_mae=s.tags.get("valid_mae"))


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest) -> Prediction:
    s = _served()
    p = params()["features"]
    cols = feature_names(p["lags"], p["rolling_hours"])
    hour = pd.Timestamp(req.hour).tz_localize(None).floor("h")
    try:
        row = state.features.loc[(req.zone_id, hour)]
    except KeyError:
        raise HTTPException(404, detail=f"no data for zone {req.zone_id} at {hour}") from None
    x = row[cols]
    if x.isna().any():
        raise HTTPException(422, detail=f"not enough history before {hour} (the model needs 168 hours)")
    y = float(s.model.predict(x.to_frame().T.astype("float64"))[0])
    return Prediction(zone_id=req.zone_id, hour=hour.to_pydatetime(), predicted_pickups=round(max(y, 0.0), 2),
                      actual_pickups=int(row["pickups"]), model_version=s.version)


@app.post("/admin/reload", response_model=Health)
def admin_reload(x_admin_token: str | None = Header(default=None)) -> Health:
    expected = admin_token()
    if not expected:
        raise HTTPException(403, detail="disabled: set ADMIN_TOKEN in .env")
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
        raise HTTPException(401, detail="missing or wrong X-Admin-Token")
    try:
        reload()
    except Exception:  # noqa: BLE001
        raise HTTPException(503, detail=f"reload failed, still serving the previous model: {state.error}") from None
    return health()
