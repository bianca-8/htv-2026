"""Fail-closed OS sandbox for experiment scripts. No inherited credentials or network."""
import asyncio
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time


class RunCancelled(RuntimeError):
    pass


def sandbox_command(folder: Path):
    python = str(Path(sys.executable).absolute())
    if platform.system() == 'Darwin' and Path('/usr/bin/sandbox-exec').exists():
        # Read only Python and OS runtime files, plus this run's inputs.
        roots = {'/System', '/usr', '/Library/Frameworks', '/Library/Apple', str(Path(sys.prefix).resolve()),
                 str(Path(sys.base_prefix).resolve()), str(folder.resolve())}
        reads = ' '.join('(subpath ' + json.dumps(p) + ')' for p in sorted(roots))
        profile = ('(version 1)(deny default)(allow process-exec (literal ' + json.dumps(python) + '))'
                   '(allow process-exec (subpath ' + json.dumps(str(Path(sys.base_prefix).resolve())) + '))'
                   '(allow sysctl-read)(allow mach-lookup)(allow file-read-metadata)'
                   '(allow file-read* (literal "/private/etc/localtime") (subpath "/dev/fd") (literal "/dev/random"))'
                   '(allow file-read* ' + reads + ' (literal "/") (literal "/dev/urandom") (literal "/dev/null"))'
                   '(allow file-write* (subpath ' + json.dumps(str(folder.resolve())) + ') (literal "/dev/null"))'
                   '(deny file-write* ' + ' '.join('(literal ' + json.dumps(str(folder.resolve() / name)) + ')'
                     for name in ('inputs.npz', 'experiment.py', 'launcher.py', 'plan.json', 'environment.json')) + ')')
        return ['/usr/bin/sandbox-exec', '-p', profile, python, '-I', 'launcher.py']
    if platform.system() == 'Linux' and shutil.which('bwrap'):
        command = ['bwrap', '--unshare-all', '--die-with-parent', '--new-session', '--proc', '/proc', '--dev', '/dev']
        for root in sorted({'/usr', '/lib', '/lib64', str(Path(sys.prefix).resolve()), str(Path(sys.base_prefix).resolve())}):
            if Path(root).exists():
                command += ['--ro-bind', root, root]
        command += ['--bind', str(folder.resolve()), '/run']
        for name in ('inputs.npz', 'experiment.py', 'launcher.py', 'plan.json', 'environment.json'):
            if (folder / name).is_file():
                command += ['--ro-bind', str((folder / name).resolve()), '/run/' + name]
        return command + ['--chdir', '/run', '--setenv', 'HOME', '/run', '--setenv', 'TMPDIR', '/run', python, '-I', 'launcher.py']
    raise RuntimeError('No supported OS sandbox. Install bubblewrap on Linux or run on macOS. Execution was blocked.')


LAUNCHER = '''import os, resource, runpy
resource.setrlimit(resource.RLIMIT_CPU, (120, 120))
resource.setrlimit(resource.RLIMIT_FSIZE, (32 * 1024 * 1024, 32 * 1024 * 1024))
resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
if hasattr(resource, 'RLIMIT_AS') and __import__('sys').platform != 'darwin':
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
os.environ['MPLCONFIGDIR'] = os.getcwd()
runpy.run_path('experiment.py', run_name='__main__')
'''


async def execute(script: str, folder: Path, cancelled, timeout=180):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'experiment.py').write_text(script)
    (folder / 'launcher.py').write_text(LAUNCHER)
    env = {'PATH': '/usr/bin:/bin', 'HOME': str(folder.resolve()), 'TMPDIR': str(folder.resolve()),
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1', 'VECLIB_MAXIMUM_THREADS': '1',
           'PYTHONHASHSEED': '11', 'PYTHONDONTWRITEBYTECODE': '1'}
    if cancelled():
        raise RunCancelled('Cancelled before execution.')
    command = sandbox_command(folder)
    start = time.monotonic()
    with (folder / 'stdout.txt').open('wb') as stdout, (folder / 'stderr.txt').open('wb') as stderr:
        process = await asyncio.create_subprocess_exec(*command, cwd=folder, env=env, stdout=stdout, stderr=stderr,
                                                       start_new_session=True)
        try:
            while process.returncode is None:
                if cancelled():
                    raise RunCancelled('Cancelled by the researcher; partial output is retained.')
                if sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink()) > 64 * 1024 * 1024:
                    raise RuntimeError('Experiment exceeded the 64 MB artifact budget.')
                if time.monotonic() - start > timeout:
                    raise RuntimeError(f'Experiment exceeded {timeout} seconds.')
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.2)
                except asyncio.TimeoutError:
                    pass
        finally:
            if process.returncode is None:
                os.killpg(process.pid, signal.SIGKILL)
                await process.wait()
    output = (folder / 'stdout.txt').read_text(errors='replace')
    error = (folder / 'stderr.txt').read_text(errors='replace')
    if process.returncode:
        raise RuntimeError(f'Experiment exited {process.returncode}: {error[-2000:]}')
    if not output.strip():
        raise RuntimeError('Experiment produced no result.')
    try:
        payload = json.loads(output.strip().splitlines()[-1], parse_constant=lambda x: (_ for _ in ()).throw(ValueError(x)))
    except (ValueError, TypeError) as exc:
        raise RuntimeError('Experiment must finish with a valid JSON result; raw output is retained.') from exc
    if not isinstance(payload, dict):
        raise RuntimeError('Experiment result must be an object.')
    return payload
