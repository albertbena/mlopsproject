import numpy as np
import pandas as pd

from taxi.features import between, feature_names
from taxi.monitor import check, load_model, psi
from taxi.train import load_features

from conftest import isolated, synthetic

COLS = feature_names([1, 2, 3, 24, 168], 24)


def test_psi_is_zero_for_the_same_distribution_and_grows_with_the_shift():
    same = psi([0.25] * 4, [0.25] * 4)
    small = psi([0.25] * 4, [0.3, 0.25, 0.25, 0.2])
    big = psi([0.25] * 4, [0.7, 0.1, 0.1, 0.1])
    assert same == 0 and 0 < small < 0.1 < big


def _report(registry_with_two_versions, data, start="2020-03-02"):
    with isolated(registry_with_two_versions["tmp"]):
        model, _, _, profile = load_model()
        window = between(load_features(data), start, (pd.Timestamp(start) + pd.Timedelta(days=6)).strftime("%Y-%m-%d"))
        return check(model, profile, window, COLS, 7 * 24)


def test_a_normal_week_is_ok(registry_with_two_versions):
    r = _report(registry_with_two_versions, synthetic())
    assert r["status"] == "ok" and r["action"] == "none", r["reasons"]


def test_a_demand_collapse_is_drift_and_asks_for_retraining(registry_with_two_versions):
    r = _report(registry_with_two_versions, synthetic(march_factor=0.2))
    assert r["status"] == "drift" and r["action"] == "retrain"
    assert r["psi"]["pickups"] > 0.2


def test_a_silent_zone_is_a_data_problem_not_a_reason_to_retrain(registry_with_two_versions):
    data = synthetic()
    data.loc[(data["zone_id"] == 1) & (data["hour"] >= pd.Timestamp("2020-03-02")), "pickups"] = 0
    r = _report(registry_with_two_versions, data)
    assert r["status"] == "data_quality" and r["action"] == "hold"
    assert r["silent_zones"] == [1]
