"""Copy local SQLite history into an empty configured PostgreSQL database."""
import argparse
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from . import storage


def migrate(source):
    if not storage.postgres_enabled():
        raise ValueError('Set DATABASE_URL or TIGER_DATABASE_URL to your PostgreSQL database first.')
    source = Path(source).resolve(strict=True)
    with closing(sqlite3.connect(source.as_uri() + '?mode=ro', uri=True)) as local:
        local.execute('BEGIN')
        snapshots = local.execute('SELECT id,snapshot,updated FROM investigations').fetchall()
        events = local.execute('SELECT id,n,payload FROM events ORDER BY id,n').fetchall()
    # Validate all JSON before opening the target transaction.
    for _, payload, _ in snapshots:
        json.loads(payload)
    for _, _, payload in events:
        json.loads(payload)
    with storage.connect() as target:
        target.execute('LOCK TABLE investigations, events IN ACCESS EXCLUSIVE MODE')
        if target.execute('SELECT EXISTS(SELECT 1 FROM investigations) OR EXISTS(SELECT 1 FROM events)').fetchone()[0]:
            raise ValueError('Migration requires an empty target database; existing records were not changed.')
        for investigation_id, snapshot, updated in snapshots:
            storage.write_snapshot(target, investigation_id, snapshot, updated)
        for investigation_id, n, payload in events:
            storage.write_event(target, investigation_id, n, payload)
    return len(snapshots), len(events)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=storage.ROOT / '.cache' / 'research.sqlite3')
    args = parser.parse_args()
    try:
        investigations, events = migrate(args.source)
    except Exception:
        # Driver error text can contain connection details. Keep credentials out of logs.
        parser.exit(1, 'Migration failed. Check the source file, PostgreSQL connection, CA certificate, and that the target is empty. No partial migration was committed.\n')
    print(f'Migrated {investigations} investigations and {events} events. SQLite source preserved.')


if __name__ == '__main__':
    main()
