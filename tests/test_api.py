"""Integration tests: the real FastAPI app, a real MLflow registry (temporary), real model files."""
import pytest
from fastapi.testclient import TestClient

from conftest import isolated


@pytest.fixture(scope="module")
def client(registry_with_two_versions):
    with isolated(registry_with_two_versions["tmp"]):
        from taxi.api.app import app, state
        state.served, state.features, state.error = None, None, None
        with TestClient(app) as c:
            yield c


def test_health_names_the_champion(client, registry_with_two_versions):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": registry_with_two_versions["v1"], "error": None}


def test_model_info_says_which_code_and_data_trained_it(client):
    info = client.get("/model").json()
    assert info["train_period"] == "2020-01-08..2020-02-15"
    assert info["git_commit"] and info["data_md5"]


def test_predict_returns_a_count_and_the_actual_value(client):
    r = client.post("/predict", json={"zone_id": 1, "hour": "2020-02-20T18:00:00"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["predicted_pickups"] >= 0
    assert isinstance(body["actual_pickups"], int)


@pytest.mark.parametrize("payload, code", [
    ({"zone_id": 0, "hour": "2020-02-20T18:00:00"}, 422),        # zone out of range
    ({"zone_id": 1, "hour": "not a date"}, 422),
    ({"zone_id": 4, "hour": "2020-02-20T18:00:00"}, 404),        # zone with no data
    ({"zone_id": 1, "hour": "2021-01-01T00:00:00"}, 404),        # hour with no data
    ({"zone_id": 1, "hour": "2020-01-02T00:00:00"}, 422),        # not enough history for the lags
])
def test_bad_requests_get_a_clear_status(client, payload, code):
    assert client.post("/predict", json=payload).status_code == code


def test_reload_needs_the_admin_token(client):
    assert client.post("/admin/reload").status_code == 401
    assert client.post("/admin/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401


def test_rollback_then_reload_switches_the_served_version(client, registry_with_two_versions):
    r = registry_with_two_versions
    with isolated(r["tmp"]):
        from taxi import registry
        registry.promote(r["v2"], reason="test")
        assert client.get("/health").json()["model_version"] == r["v1"]  # not reloaded yet
        assert client.post("/admin/reload", headers={"X-Admin-Token": "test-token"}).json()["model_version"] == r["v2"]
        registry.rollback(reason="test")
        client.post("/admin/reload", headers={"X-Admin-Token": "test-token"})
        assert client.get("/health").json()["model_version"] == r["v1"]
        assert client.post("/predict", json={"zone_id": 2, "hour": "2020-02-21T08:00:00"}).json()["model_version"] == r["v1"]
