import pytest

from churn.registry import (
    NoRollbackTarget,
    apply_production,
    empty_registry,
    next_version,
    previous_version,
)


def test_versions_increment():
    registry = empty_registry()
    assert next_version(registry) == "1"
    registry["versions"]["1"] = {}
    assert next_version(registry) == "2"


def test_previous_version_walks_history():
    registry = empty_registry()
    registry["versions"] = {"1": {}, "2": {}}
    apply_production(registry, "1")
    apply_production(registry, "2")
    assert previous_version(registry) == "1"


def test_rollback_requires_another_version():
    registry = empty_registry()
    registry["versions"] = {"1": {}}
    apply_production(registry, "1")
    with pytest.raises(NoRollbackTarget):
        previous_version(registry)
