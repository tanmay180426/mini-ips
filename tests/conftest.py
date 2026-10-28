"""Shared test isolation for implicit dashboard database paths."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_implicit_dashboard_databases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Route dashboard defaults into a disposable per-test directory."""
    monkeypatch.setenv("MINI_IDS_DB_PATH", str(tmp_path / "default_test.db"))
    monkeypatch.setenv("MINI_IDS_DEMO_DB_PATH", str(tmp_path / "demo_test.db"))
