"""Opt-in real OS containment tests: RUN_SANDBOX_TESTS=1 python -m pytest tests/test_sandbox.py."""
import asyncio
import json
import os
from pathlib import Path

import pytest
from backend.runner import execute, RunCancelled
from backend.workflow import prepare_data, bench_script, scored_payload
from backend.data_ground import Grounding
from backend.methods import Path as MethodPath

pytestmark = pytest.mark.skipif(os.environ.get('RUN_SANDBOX_TESTS') != '1', reason='Requires OS sandbox access outside a nested agent sandbox.')


def test_numpy_sklearn_and_local_artifacts_work_without_inherited_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'should-never-be-in-child')
    script = '''import json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
x = np.array([[0.], [1.], [2.], [3.]])
m = LogisticRegression().fit(x, [0,0,1,1])
open('table.csv', 'w').write('probability\\n' + str(m.predict_proba([[1.]])[0,1]))
print(json.dumps({'ok': True, 'secret_present': 'GEMINI_API_KEY' in os.environ}))
'''
    result = asyncio.run(execute(script, tmp_path / 'run', lambda: False))
    assert result == {'ok': True, 'secret_present': False}
    assert (tmp_path / 'run' / 'table.csv').exists()


def test_sandbox_blocks_external_reads_writes_and_network(tmp_path):
    secret = tmp_path / 'secret.txt'
    secret.write_text('private')
    (tmp_path / 'run').mkdir()
    (tmp_path / 'run' / 'inputs.npz').write_bytes(b'protected input')
    script = f'''import json, socket
checks = {{}}
try:
    open('inputs.npz', 'wb').write(b'changed')
    checks['input_protected'] = False
except OSError:
    checks['input_protected'] = True
try:
    open({str(secret)!r}).read()
    checks['read_blocked'] = False
except OSError:
    checks['read_blocked'] = True
try:
    open({str(tmp_path / 'outside.txt')!r}, 'w').write('bad')
    checks['write_blocked'] = False
except OSError:
    checks['write_blocked'] = True
try:
    socket.create_connection(('127.0.0.1', 9), timeout=.2)
    checks['network_blocked'] = False
except PermissionError:
    checks['network_blocked'] = True
print(json.dumps(checks))
'''
    checks = asyncio.run(execute(script, tmp_path / 'run', lambda: False))
    assert checks == {'read_blocked': True, 'write_blocked': True, 'network_blocked': True, 'input_protected': True}
    assert not (tmp_path / 'outside.txt').exists()


def test_cancellation_terminates_process_and_retains_logs(tmp_path):
    async def scenario():
        cancel = False
        async def later():
            nonlocal cancel
            await asyncio.sleep(.3)
            cancel = True
        task = asyncio.create_task(later())
        with pytest.raises(RunCancelled):
            await execute('import time\nprint("started", flush=True)\ntime.sleep(10)\n', tmp_path / 'run', lambda: cancel)
        await task
        assert 'started' in (tmp_path / 'run' / 'stdout.txt').read_text()
    asyncio.run(scenario())


def test_bench_runs_from_prepared_table_and_produces_valid_scores(tmp_path):
    ground = Grounding('public', 'We use Wisconsin breast cancer data.', 'Study', 'https://example.org/p',
                       dataset_name='Wisconsin breast cancer', loader='breast_cancer')
    data = prepare_data(ground, tmp_path / 'inputs')
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'inputs.npz').write_bytes((tmp_path / 'inputs' / 'inputs.npz').read_bytes())
    path = MethodPath('logreg', 'Logistic regression', ['logreg'], [], [])
    payload = asyncio.run(execute(bench_script(path, ground), folder, lambda: False))
    scores = scored_payload(payload, path, None)
    assert data['n_rows'] == 569
    assert 0 <= scores['metrics']['average_precision'] <= 1
    assert (folder / 'dataset.csv').exists()
