"""Verify the real executor as the deployment user before a live demo."""
import asyncio
import json
from pathlib import Path
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backend.runner import execute

SCRIPT = '''import json,numpy as np
try:
    open('/etc/marginalia.env').read()
    raise RuntimeError('Sandbox exposed credentials')
except (PermissionError,FileNotFoundError):
    pass
try:
    import socket
    s=socket.socket()
    s.settimeout(1)
    s.connect(('1.1.1.1',443))
    raise RuntimeError('Sandbox allowed outbound network')
except OSError:
    pass
finally:
    if 's' in locals():s.close()
assert 'GEMINI_API_KEY' not in __import__('os').environ
assert 'TIGER_DATABASE_URL' not in __import__('os').environ
assert 'DATABASE_URL' not in __import__('os').environ
print(json.dumps({'sum':float(np.sum([1,2,3]))}))
'''


async def main():
    root = Path(__file__).resolve().parents[1] / 'experiments'
    root.mkdir(exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix='deployment-check-', dir=root))
    try:
        result = await execute(SCRIPT, folder, lambda: False, timeout=15)
        assert result == {'sum': 6.0}, result
        print('Sandbox passed: NumPy runs; credential files, inherited keys, and outbound network are unavailable.')
    finally:
        shutil.rmtree(folder)


if __name__ == '__main__':
    asyncio.run(main())
