import pandas as pd
import pytest

from taxi.features import add_features, between, check_hourly_grid, feature_names

from conftest import synthetic


def test_lags_point_to_past_hours_of_the_same_zone():
    df = synthetic(end="2020-01-15")
    out = add_features(df, [1, 24, 168], 24)
    z = out[out["zone_id"] == 2].set_index("hour")
    t = pd.Timestamp("2020-01-10 18:00")
    assert z.loc[t, "lag_1"] == z.loc[t - pd.Timedelta(hours=1), "pickups"]
    assert z.loc[t, "lag_24"] == z.loc[t - pd.Timedelta(hours=24), "pickups"]
    assert z.loc[t, "lag_168"] == z.loc[t - pd.Timedelta(hours=168), "pickups"]


def test_no_feature_uses_the_hour_it_predicts():
    df = synthetic(end="2020-01-15")
    out = add_features(df, [1], 24)
    z = out[out["zone_id"] == 1].set_index("hour")
    t = pd.Timestamp("2020-01-12 09:00")
    window = z.loc[t - pd.Timedelta(hours=24): t - pd.Timedelta(hours=1), "pickups"]
    assert z.loc[t, "rolling_mean_24"] == pytest.approx(window.mean())


def test_first_hours_have_no_lag_rather_than_a_wrong_one():
    out = add_features(synthetic(end="2020-01-15"), [168], 24)
    first_week = out[out["hour"] < pd.Timestamp("2020-01-08")]
    assert first_week["lag_168"].isna().all()


def test_a_missing_hour_is_refused():
    df = synthetic(end="2020-01-05")
    df = df.drop(df.index[10])
    with pytest.raises(ValueError, match="incomplete hourly series"):
        check_hourly_grid(df)


def test_split_by_days_includes_both_ends():
    df = synthetic(end="2020-01-10")
    part = between(df, "2020-01-02", "2020-01-03")
    assert part["hour"].min() == pd.Timestamp("2020-01-02 00:00")
    assert part["hour"].max() == pd.Timestamp("2020-01-03 23:00")


def test_feature_names_match_the_columns():
    out = add_features(synthetic(end="2020-01-10"), [1, 2], 24)
    assert set(feature_names([1, 2], 24)) <= set(out.columns)
