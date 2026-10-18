"""Durable SQLite/PostgreSQL investigation snapshots and append-only event history."""
import json
import os
import sqlite3
import certifi
from urllib.parse import parse_qs, urlsplit
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def database_url():
    generic = os.environ.get('DATABASE_URL', '').strip()
    tiger = os.environ.get('TIGER_DATABASE_URL', '').strip()
    if generic and tiger and generic != tiger:
        raise ValueError('Set only one of DATABASE_URL or TIGER_DATABASE_URL; they point to different databases.')
    url = generic or tiger
    if url and not url.startswith(('postgresql://', 'postgres://')):
        raise ValueError('DATABASE_URL or TIGER_DATABASE_URL must be a PostgreSQL connection URL.')
    return url


def postgres_enabled():
    return bool(database_url())


def postgres_options():
    # Tiger Cloud can use a public CA. Vultr/custom CAs can be supplied explicitly.
    query = parse_qs(urlsplit(database_url()).query)
    root = os.environ.get('PGSSLROOTCERT') or query.get('sslrootcert', [certifi.where()])[-1]
    return {'connect_timeout': 10, 'sslmode': 'verify-full', 'sslrootcert': root}


@contextmanager
def connect():
    if postgres_enabled():
        try:
            import psycopg
        except ImportError:
            raise RuntimeError('Install requirements.txt to enable PostgreSQL.') from None
        # Verify the database certificate and hostname. Never silently fall back.
        with psycopg.connect(database_url(), **postgres_options()) as db:
            db.execute('CREATE TABLE IF NOT EXISTS investigations (id TEXT PRIMARY KEY, snapshot JSONB NOT NULL, updated DOUBLE PRECISION NOT NULL)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT NOT NULL, n BIGINT NOT NULL, payload JSONB NOT NULL, PRIMARY KEY(id,n))')
            yield db
    else:
        path = Path(os.environ.get('MARGINALIA_DB', str(ROOT / '.cache' / 'research.sqlite3')))
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=10)
        try:
            with db:
                db.execute('CREATE TABLE IF NOT EXISTS investigations (id TEXT PRIMARY KEY, snapshot TEXT NOT NULL, updated REAL NOT NULL)')
                db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT, n INTEGER, payload TEXT, PRIMARY KEY(id,n))')
                yield db
        finally:
            db.close()


def decode(value):
    return json.loads(value) if isinstance(value, str) else value


def write_snapshot(db, investigation_id, snapshot, updated):
    if postgres_enabled():
        db.execute('INSERT INTO investigations (id,snapshot,updated) VALUES (%s,%s::jsonb,%s) '
                   'ON CONFLICT(id) DO UPDATE SET snapshot=excluded.snapshot, updated=excluded.updated',
                   (investigation_id, snapshot, updated))
    else:
        db.execute('INSERT OR REPLACE INTO investigations VALUES (?,?,?)', (investigation_id, snapshot, updated))


def write_event(db, investigation_id, n, payload):
    if postgres_enabled():
        db.execute('INSERT INTO events (id,n,payload) VALUES (%s,%s,%s::jsonb) ON CONFLICT(id,n) DO NOTHING',
                   (investigation_id, n, payload))
    else:
        db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?)', (investigation_id, n, payload))


def save(investigation, event):
    snapshot = {
        'direction': investigation.direction, 'query_override': investigation.query_override,
        'id': investigation.id, 'phase': investigation.phase, 'paused': investigation.paused,
        'rules': sorted(investigation.rules), 'dropped': sorted(investigation.dropped),
        'metric': investigation.metric, 'precision_floor': investigation.precision_floor,
        'papers': [p.as_dict() for p in investigation.papers],
        'research': investigation.research, 'seq': investigation.seq,
    }
    with connect() as db:
        write_snapshot(db, investigation.id, json.dumps(snapshot, allow_nan=False), event['ts'])
        write_event(db, investigation.id, event['n'], json.dumps(event, allow_nan=False))


def snapshots():
    with connect() as db:
        return [decode(row[0]) for row in db.execute('SELECT snapshot FROM investigations ORDER BY updated DESC')]


def events(investigation_id):
    with connect() as db:
        placeholder = '%s' if postgres_enabled() else '?'
        return [decode(row[0]) for row in db.execute(f'SELECT payload FROM events WHERE id={placeholder} ORDER BY n', (investigation_id,))]
