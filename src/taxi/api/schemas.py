"""Request and response bodies of the API (validated by FastAPI: a bad body gets a 422, not a crash)."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    zone_id: int = Field(ge=1, le=263, description="TLC taxi zone (PULocationID)", examples=[237])
    hour: datetime = Field(description="the hour to predict, New York local time", examples=["2020-02-20T18:00:00"])


class Prediction(BaseModel):
    zone_id: int
    hour: datetime
    predicted_pickups: float
    actual_pickups: int | None = Field(default=None, description="what really happened, when the data has it")
    model_version: str


class Health(BaseModel):
    status: str
    model_version: str | None = None
    error: str | None = None


class ModelInfo(BaseModel):
    name: str
    version: str
    run_id: str
    git_commit: str | None = None
    data_md5: str | None = None
    train_period: str | None = None
    valid_mae: str | None = None
