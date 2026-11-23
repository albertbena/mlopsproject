from taxi.retrain import plan_dates, retrain

from conftest import isolated, synthetic


def test_dates_hold_out_the_last_days_of_the_window():
    d = plan_dates("2020-03-15 23:00:00")
    assert d == {"train": ("2020-03-02", "2020-03-12"), "holdout": ("2020-03-13", "2020-03-15")}


def test_hold_and_none_never_retrain(registry_with_two_versions):
    with isolated(registry_with_two_versions["tmp"]):
        for action in ("hold", "none"):
            out = retrain({"action": action, "window": {"start": "x", "end": "2020-03-15"}}, save=False)
            assert out["decision"] == "no_retrain"


def test_drift_trains_a_challenger_and_compares_it_on_the_same_holdout(registry_with_two_versions):
    with isolated(registry_with_two_versions["tmp"]):
        from taxi import registry
        before = registry.champion()
        report = {"action": "retrain", "reasons": ["test"],
                  "window": {"start": "2020-03-09 00:00:00", "end": "2020-03-15 23:00:00"}}
        out = retrain(report, data=synthetic(march_factor=0.2), save=False)
        assert out["decision"] in ("promote", "keep_champion")
        assert out["champion"]["version"] == before
        assert out["holdout"] == ("2020-03-13", "2020-03-15")
        expected = out["challenger"]["version"] if out["decision"] == "promote" else before
        assert registry.champion() == expected
