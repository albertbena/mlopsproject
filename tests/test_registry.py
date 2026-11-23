import pytest

from conftest import isolated


def test_promote_keeps_the_old_champion_as_previous(registry_with_two_versions):
    r = registry_with_two_versions
    with isolated(r["tmp"]):
        from taxi import registry
        event = registry.promote(r["v2"], reason="test")
        assert (registry.champion(), registry.previous()) == (r["v2"], r["v1"])
        assert event["changed"] is True


def test_rollback_restores_previous_and_can_be_undone(registry_with_two_versions):
    r = registry_with_two_versions
    with isolated(r["tmp"]):
        from taxi import registry
        registry.rollback(reason="test")
        assert (registry.champion(), registry.previous()) == (r["v1"], r["v2"])
        registry.rollback(reason="undo")
        assert registry.champion() == r["v2"]
        registry.rollback(reason="back to v1", to=r["v1"])
        assert registry.champion() == r["v1"]


def test_rollback_to_the_champion_itself_is_refused(registry_with_two_versions):
    r = registry_with_two_versions
    with isolated(r["tmp"]):
        from taxi import registry
        with pytest.raises(RuntimeError, match="already the champion"):
            registry.rollback(reason="test", to=registry.champion())


def test_every_move_is_logged(registry_with_two_versions):
    log = (registry_with_two_versions["tmp"] / "registry_log.jsonl").read_text().splitlines()
    assert any('"action": "rollback"' in line for line in log)
