"""Turn the raw trips into one row per zone and hour: (zone_id, hour, pickups).

Hours with no pickup are written as 0, so every zone has a complete hourly series: the lag features can then
shift by rows. Trips dated outside their file's month (the TLC files contain a few) are dropped.

    python -m taxi.data.aggregate
"""
from __future__ import annotations

import json
import sys

import pandas as pd
import pyarrow.parquet as pq

from taxi.config import params, path
from taxi.data.download import file_name

PICKUP = "tpep_pickup_datetime"
ZONE = "PULocationID"
REQUIRED = {PICKUP, ZONE}


def check_schema(columns: list[str], name: str) -> list[str]:
    """Fail on a missing column we need; return a warning for each column we do not know (e.g. a new fee)."""
    missing = REQUIRED - set(columns)
    if missing:
        raise ValueError(f"{name}: missing column(s) {sorted(missing)}")
    known = {
        "VendorID", PICKUP, "tpep_dropoff_datetime", "passenger_count", "trip_distance", "RatecodeID",
        "store_and_fwd_flag", ZONE, "DOLocationID", "payment_type", "fare_amount", "extra", "mta_tax",
        "tip_amount", "tolls_amount", "improvement_surcharge", "total_amount", "congestion_surcharge",
        "airport_fee", "Airport_fee",
    }
    return [f"{name}: new column {c!r}" for c in columns if c not in known]


def month_counts(file, month: str, max_zone: int) -> pd.DataFrame:
    """Pickups per (zone, hour) for one monthly file."""
    columns = pq.ParquetFile(file).schema_arrow.names
    for warning in check_schema(columns, file.name):
        print(f"  warning: {warning}", file=sys.stderr)
    trips = pd.read_parquet(file, columns=[PICKUP, ZONE])
    start = pd.Timestamp(f"{month}-01")
    end = start + pd.offsets.MonthBegin(1)
    trips = trips[(trips[PICKUP] >= start) & (trips[PICKUP] < end)]
    trips = trips[(trips[ZONE] >= 1) & (trips[ZONE] <= max_zone)]
    hours = trips[PICKUP].dt.floor("h")
    counts = trips.groupby([trips[ZONE].rename("zone_id"), hours.rename("hour")]).size().rename("pickups")
    return counts.reset_index()


def complete_grid(counts: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, max_zone: int) -> pd.DataFrame:
    """Every zone 1..max_zone times every hour in [start, end), missing pairs filled with 0 pickups."""
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    grid = pd.MultiIndex.from_product([range(1, max_zone + 1), hours], names=["zone_id", "hour"])
    out = counts.set_index(["zone_id", "hour"])["pickups"].reindex(grid, fill_value=0).reset_index()
    out["zone_id"] = out["zone_id"].astype("int16")
    out["pickups"] = out["pickups"].astype("int32")
    return out


def main() -> None:
    cfg = params()["data"]
    raw = path(cfg["raw_dir"])
    months = cfg["months"]
    max_zone = int(cfg["max_zone_id"])
    parts = []
    for month in months:
        part = month_counts(raw / file_name(month), month, max_zone)
        print(f"{month}: {int(part['pickups'].sum()):,} pickups")
        parts.append(part)
    start = pd.Timestamp(f"{months[0]}-01")
    end = pd.Timestamp(f"{months[-1]}-01") + pd.offsets.MonthBegin(1)
    df = complete_grid(pd.concat(parts, ignore_index=True), start, end, max_zone)

    out = path(cfg["processed"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    summary = {
        "rows": len(df),
        "zones": int(df["zone_id"].nunique()),
        "hours": int(df["hour"].nunique()),
        "first_hour": str(df["hour"].min()),
        "last_hour": str(df["hour"].max()),
        "pickups": int(df["pickups"].sum()),
    }
    path("reports").mkdir(exist_ok=True)
    (path("reports") / "data_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
