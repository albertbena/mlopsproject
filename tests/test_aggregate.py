import pandas as pd
import pytest

from taxi.data.aggregate import check_schema, complete_grid, month_counts


def test_missing_hours_become_zero_pickups():
    counts = pd.DataFrame({"zone_id": [1], "hour": [pd.Timestamp("2020-01-01 01:00")], "pickups": [4]})
    out = complete_grid(counts, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01 03:00"), max_zone=2)
    assert len(out) == 2 * 3
    assert out["pickups"].sum() == 4
    assert out.set_index(["zone_id", "hour"]).loc[(2, pd.Timestamp("2020-01-01 02:00")), "pickups"] == 0


def test_a_new_column_is_a_warning_and_a_missing_one_an_error():
    warnings = check_schema(["tpep_pickup_datetime", "PULocationID", "cbd_congestion_fee"], "f.parquet")
    assert warnings == ["f.parquet: new column 'cbd_congestion_fee'"]
    with pytest.raises(ValueError, match="PULocationID"):
        check_schema(["tpep_pickup_datetime", "fare_amount"], "f.parquet")


def test_trips_outside_the_month_and_unknown_zones_are_dropped(tmp_path):
    trips = pd.DataFrame({
        "tpep_pickup_datetime": pd.to_datetime(["2020-01-05 10:15", "2020-01-05 10:45", "2019-12-31 23:50",
                                                "2020-02-01 00:05", "2020-01-06 08:00"]),
        "PULocationID": [5, 5, 5, 5, 265],
    })
    f = tmp_path / "yellow_tripdata_2020-01.parquet"
    trips.to_parquet(f)
    out = month_counts(f, "2020-01", max_zone=263)
    assert out.to_dict("records") == [{"zone_id": 5, "hour": pd.Timestamp("2020-01-05 10:00"), "pickups": 2}]
