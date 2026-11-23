"""Features for "pickups in this zone in the next hour": the recent hours of the zone and the calendar.

Everything is computed from hours strictly before the one predicted (shift >= 1), so nothing leaks from
the target. The input must be a complete hourly grid per zone (see taxi.data.aggregate).
"""
from __future__ import annotations

import pandas as pd

CALENDAR = ["hour_of_day", "day_of_week", "is_weekend"]


def feature_names(lags: list[int], rolling_hours: int) -> list[str]:
    return [f"lag_{l}" for l in lags] + [f"rolling_mean_{rolling_hours}"] + CALENDAR


def check_hourly_grid(df: pd.DataFrame) -> None:
    """Fail if a zone has a missing or duplicated hour: the lags would silently point to the wrong hour."""
    counts = df.groupby("zone_id")["hour"].agg(["count", "min", "max"])
    expected = ((counts["max"] - counts["min"]) / pd.Timedelta(hours=1)).astype(int) + 1
    bad = counts.index[counts["count"] != expected]
    if len(bad):
        raise ValueError(f"incomplete hourly series for zone(s) {list(bad[:5])}")


def add_features(df: pd.DataFrame, lags: list[int], rolling_hours: int) -> pd.DataFrame:
    """Return a copy of (zone_id, hour, pickups) with the lag, rolling and calendar columns added."""
    check_hourly_grid(df)
    out = df.sort_values(["zone_id", "hour"]).reset_index(drop=True)
    by_zone = out.groupby("zone_id", sort=False)["pickups"]
    for lag in lags:
        out[f"lag_{lag}"] = by_zone.shift(lag)
    out[f"rolling_mean_{rolling_hours}"] = by_zone.transform(
        lambda s: s.shift(1).rolling(rolling_hours, min_periods=rolling_hours).mean()
    )
    out["hour_of_day"] = out["hour"].dt.hour
    out["day_of_week"] = out["hour"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    return out


def between(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Rows whose hour falls on the days [start, end], both included. Splits are by time, never at random."""
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) + pd.Timedelta(days=1)
    return df[(df["hour"] >= lo) & (df["hour"] < hi)]
