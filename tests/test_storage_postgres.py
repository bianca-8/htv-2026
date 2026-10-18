"""Storage contract and opt-in real PostgreSQL round trip in an isolated schema."""
import json
import os
import sqlite3
import sys
import types
import uuid
from contextlib import contextmanager

import pytest
from backend import storage
from backend.migrate_postgres import migrate
from backend.session import Investigation


def test_sqlite_round_trip_and_event_dedup(monkeypatch, tmp_path):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.setenv('MARGINALIA_DB', str(tmp_path / 'research.db'))
    inv = Investigation('test question')
    event = {'n': 1, 'ts': 12.5, 'type': 'human', 'text': 'test question'}
    storage.save(inv, event)
    storage.save(inv, event)
    assert storage.snapshots()[0]['direction'] == 'test question'
    assert storage.events(inv.id) == [event]


def test_invalid_database_configuration_never_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv('DATABASE_URL', 'mysql://invalid')
    path = tmp_path / 'not-created.db'
    monkeypatch.setenv('MARGINALIA_DB', str(path))
    with pytest.raises(ValueError, match='PostgreSQL'):
        storage.snapshots()
    assert not path.exists()


def test_postgres_tls_and_jsonb_queries(monkeypatch):
    calls = []
    class DB:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def execute(self, query, params=None):
            calls.append((query, params))
            return self
        def __iter__(self): return iter([])
    def connect(url, **kwargs):
        assert kwargs == {'connect_timeout': 10, 'sslmode': 'verify-full', 'sslrootcert': '/ca.pem'}
        return DB()
    monkeypatch.setitem(sys.modules, 'psycopg', types.SimpleNamespace(connect=connect))
    monkeypatch.setenv('DATABASE_URL', 'postgresql://host/db?sslmode=disable')
    monkeypatch.setenv('PGSSLROOTCERT', '/ca.pem')
    inv = Investigation('quoted question: \' hello')
    storage.save(inv, {'n': 1, 'ts': 1})
    assert any('JSONB' in sql for sql, _ in calls)
    assert any('ON CONFLICT(id) DO UPDATE' in sql for sql, _ in calls)
    assert any('ON CONFLICT(id,n) DO NOTHING' in sql for sql, _ in calls)
    assert not any(inv.direction in sql for sql, _ in calls)
    storage.events(inv.id)
    assert calls[-1] == ('SELECT payload FROM events WHERE id=%s ORDER BY n', (inv.id,))


def test_migration_rejects_nonempty_target(monkeypatch, tmp_path):
    source = tmp_path / 'source.db'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE investigations (id TEXT, snapshot TEXT, updated REAL)')
        db.execute('CREATE TABLE events (id TEXT, n INTEGER, payload TEXT)')
    class Target:
        def execute(self, sql, params=None): return self
        def fetchone(self): return (True,)
    @contextmanager
    def target(): yield Target()
    monkeypatch.setenv('DATABASE_URL', 'postgresql://host/db')
    monkeypatch.setattr(storage, 'connect', target)
    with pytest.raises(ValueError, match='empty target'):
        migrate(source)
    assert source.exists()


@pytest.mark.skipif(not os.environ.get('TEST_POSTGRES_URL'), reason='Set TEST_POSTGRES_URL for a disposable PostgreSQL database')
def test_live_postgres_roundtrip_and_migration(monkeypatch, tmp_path):
    import psycopg
    from psycopg import sql
    url = os.environ['TEST_POSTGRES_URL']
    schema = 'test_marginalia_' + uuid.uuid4().hex
    original_connect = psycopg.connect
    options = {'sslmode': 'verify-full', 'connect_timeout': 10, 'sslrootcert': os.environ.get('PGSSLROOTCERT') or __import__('certifi').where()}
    if os.environ.get('PGSSLROOTCERT'):
        options['sslrootcert'] = os.environ['PGSSLROOTCERT']
    with original_connect(url, **options) as admin:
        admin.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
    try:
        monkeypatch.delenv('DATABASE_URL', raising=False)
        source = tmp_path / 'local.db'
        monkeypatch.setenv('MARGINALIA_DB', str(source))
        inv = Investigation('Unicode research — calibration')
        event = {'n': 1, 'ts': 1.5, 'type': 'human', 'text': inv.direction}
        storage.save(inv, event)
        def isolated_connect(dsn, **kwargs):
            kwargs['options'] = f'-c search_path={schema}'
            return original_connect(dsn, **kwargs)
        monkeypatch.setattr(psycopg, 'connect', isolated_connect)
        monkeypatch.setenv('DATABASE_URL', url)
        assert migrate(source) == (1, 1)
        assert storage.snapshots()[0]['direction'] == inv.direction
        assert storage.events(inv.id) == [event]
        inv.direction = 'Updated'
        storage.save(inv, {'n': 2, 'ts': 2.5, 'type': 'human'})
        storage.save(inv, {'n': 2, 'ts': 2.5, 'type': 'human'})
        assert storage.snapshots()[0]['direction'] == 'Updated'
        assert [e['n'] for e in storage.events(inv.id)] == [1, 2]
        with pytest.raises(ValueError, match='empty target'):
            migrate(source)
    finally:
        with original_connect(url, **options) as admin:
            admin.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))


def test_snapshot_and_event_rollback_together(monkeypatch, tmp_path):
    monkeypatch.setenv('MARGINALIA_DB', str(tmp_path / 'rollback.db'))
    inv = Investigation('Original')
    storage.save(inv, {'n': 1, 'ts': 1})
    inv.direction = 'Must roll back'
    with pytest.raises(ValueError):
        storage.save(inv, {'n': 2, 'ts': 2, 'bad': float('nan')})
    assert storage.snapshots()[0]['direction'] == 'Original'
    assert [e['n'] for e in storage.events(inv.id)] == [1]


def test_tiger_url_uses_postgres_with_public_ca(monkeypatch):
    import certifi
    monkeypatch.setenv('TIGER_DATABASE_URL', 'postgresql://tsdbadmin:secret@service.project.tsdb.cloud.timescale.com:5432/tsdb?sslmode=require')
    monkeypatch.delenv('PGSSLROOTCERT', raising=False)
    assert storage.postgres_enabled()
    assert storage.database_url().startswith('postgresql://tsdbadmin:')
    assert storage.postgres_options() == {'connect_timeout': 10, 'sslmode': 'verify-full', 'sslrootcert': certifi.where()}


def test_conflicting_cloud_urls_fail_without_revealing_secrets(monkeypatch):
    monkeypatch.setenv('DATABASE_URL', 'postgresql://user:secret1@vultr/db')
    monkeypatch.setenv('TIGER_DATABASE_URL', 'postgresql://user:secret2@tiger/db')
    with pytest.raises(ValueError) as exc:
        storage.database_url()
    assert 'secret' not in str(exc.value)
    assert 'different databases' in str(exc.value)


def test_tiger_custom_ca_precedence(monkeypatch):
    monkeypatch.setenv('TIGER_DATABASE_URL', 'postgresql://host/tsdb?sslrootcert=%2Furl-ca.pem')
    monkeypatch.delenv('PGSSLROOTCERT', raising=False)
    assert storage.postgres_options()['sslrootcert'] == '/url-ca.pem'
    monkeypatch.setenv('PGSSLROOTCERT', '/env-ca.pem')
    assert storage.postgres_options()['sslrootcert'] == '/env-ca.pem'
