"""Never let routine tests write to a developer's configured cloud database."""
import pytest


@pytest.fixture(autouse=True)
def local_database_by_default(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.delenv('TIGER_DATABASE_URL', raising=False)
