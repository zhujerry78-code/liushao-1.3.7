"""Fresh, atomic acceptance evidence and bounded ownership-based process cleanup."""
from __future__ import annotations

import hashlib
import json
import os
import ctypes
from ctypes import wintypes
from pathlib import Path
import subprocess
import tempfile
import time
from datetime import datetime, timezone

import psutil


class _WindowsJob:
    """An unnamed kernel container inherited by every child, including orphaned ones."""

    def __init__(self):
        self.kernel = ctypes.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'CreateJobObjectW': ([wintypes.LPVOID, wintypes.LPCWSTR], wintypes.HANDLE),
            'SetInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                         wintypes.DWORD], wintypes.BOOL),
            'AssignProcessToJobObject': ([wintypes.HANDLE, wintypes.HANDLE], wintypes.BOOL),
            'QueryInformationJobObject': ([wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID,
                                           wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)],
                                          wintypes.BOOL),
            'IsProcessInJob': ([wintypes.HANDLE, wintypes.HANDLE,
                                ctypes.POINTER(wintypes.BOOL)], wintypes.BOOL),
            'OpenProcess': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'OpenThread': ([wintypes.DWORD, wintypes.BOOL, wintypes.DWORD], wintypes.HANDLE),
            'GetProcessIdOfThread': ([wintypes.HANDLE], wintypes.DWORD),
            'ResumeThread': ([wintypes.HANDLE], wintypes.DWORD),
            'TerminateJobObject': ([wintypes.HANDLE, wintypes.UINT], wintypes.BOOL),
            'CloseHandle': ([wintypes.HANDLE], wintypes.BOOL),
        }
        for name, (arguments, result) in signatures.items():
            function = getattr(self.kernel, name)
            function.argtypes, function.restype = arguments, result
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())

        class BasicLimits(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', wintypes.DWORD), ('minimum_working_set', ctypes.c_size_t),
                        ('maximum_working_set', ctypes.c_size_t), ('active_process_limit', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [('basic', BasicLimits), ('io_counters', ctypes.c_ulonglong * 6),
                        ('process_memory', ctypes.c_size_t), ('job_memory', ctypes.c_size_t),
                        ('peak_process_memory', ctypes.c_size_t), ('peak_job_memory', ctypes.c_size_t)]

        limits = ExtendedLimits()
        # Microsoft winnt.h member order; verify the ABI on the running Python.
        self.layout = {'pointer_bytes': ctypes.sizeof(ctypes.c_size_t),
                       'basic_limit_bytes': ctypes.sizeof(BasicLimits),
                       'extended_limit_bytes': ctypes.sizeof(ExtendedLimits)}
        if (self.layout['pointer_bytes'] == 8
                and (self.layout['basic_limit_bytes'], self.layout['extended_limit_bytes'])
                != (64, 144)):
            self.close()
            raise RuntimeError(f'Unexpected Windows Job ABI: {self.layout}')
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits),
                                                   ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def assign(self, process: subprocess.Popen) -> None:
        if not self.kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def resume(self, process: subprocess.Popen) -> None:
        # CREATE_SUSPENDED leaves exactly the initial thread, before user code.
        # Popen closes its initial thread handle; reopen only that verified thread
        # and use the documented ResumeThread API (no native undocumented API).
        threads = psutil.Process(process.pid).threads()
        if process.poll() is not None or len(threads) != 1:
            raise RuntimeError('Cannot identify the suspended initial process thread')
        handle = self.kernel.OpenThread(0x0002 | 0x0800, False, threads[0].id)
        if not handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if (self.kernel.GetProcessIdOfThread(handle) != process.pid
                    or process.poll() is not None or not self.contains(process.pid)):
                raise RuntimeError('Suspended thread does not belong to the owned root')
            previous = self.kernel.ResumeThread(handle)
            if previous == 0xFFFFFFFF:
                raise ctypes.WinError(ctypes.get_last_error())
            if previous != 1:
                raise RuntimeError(f'Unexpected initial thread suspend count: {previous}')
        finally:
            if not self.kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def pids(self) -> list[int]:
        capacity = 64
        for _ in range(12):
            buffer = ctypes.create_string_buffer(8 + ctypes.sizeof(ctypes.c_size_t) * capacity)
            returned = wintypes.DWORD()
            ok = self.kernel.QueryInformationJobObject(self.handle, 3, buffer,
                                                       len(buffer), ctypes.byref(returned))
            assigned = wintypes.DWORD.from_buffer(buffer, 0).value
            listed = wintypes.DWORD.from_buffer(buffer, 4).value
            if ok and listed == assigned and listed <= capacity:
                return list((ctypes.c_size_t * listed).from_buffer(buffer, 8))
            if not ok and ctypes.get_last_error() != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(ctypes.get_last_error())
            capacity = max(capacity * 2, assigned + 8)
        raise RuntimeError('Owned Job process list did not stabilize')

    def contains(self, pid: int) -> bool:
        handle = self.kernel.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error in (87, 1168):  # PID exited between query and OpenProcess.
                return False
            raise ctypes.WinError(error)
        try:
            present = wintypes.BOOL()
            if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(present)):
                raise ctypes.WinError(ctypes.get_last_error())
            return bool(present.value)
        finally:
            if not self.kernel.CloseHandle(handle):
                raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self) -> None:
        if not self.kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self) -> None:
        if self.handle is not None:
            if not self.kernel.CloseHandle(self.handle):
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle = None


def start_owned_process(argv, **kwargs):
    """Assign a suspended root to its Job before any user code can spawn children.

    The acceptance gate currently targets Windows. Unsupported containment or any
    assignment/resume failure raises instead of falling back to racy polling.
    """
    if os.name != 'nt':
        raise RuntimeError('Reliable acceptance process containment requires Windows Jobs')
    job = _WindowsJob()
    process = None
    assigned = False
    try:
        flags = kwargs.pop('creationflags', 0) | 0x4  # CREATE_SUSPENDED
        process = subprocess.Popen(argv, creationflags=flags, **kwargs)
        job.assign(process)
        assigned = True
        owner = OwnedProcessTree(process, job=job)
        job.resume(process)
        return process, owner
    except BaseException:
        try:
            if process is not None:
                if assigned:
                    job.terminate()
                else:
                    process.kill()  # Popen's original process handle, never a PID lookup.
                process.wait(timeout=5)
        finally:
            job.close()
        raise


def atomic_json(path: Path, value: dict, *, exclusive: bool = False) -> None:
    """Replace a report only after complete UTF-8 JSON has reached the local file."""
    path = Path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            # A complete file becomes visible only if the destination is absent.
            # Both names are in the same directory/filesystem; link never overwrites.
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def create_run_directory(output: Path) -> Path:
    """Reserve a new directory; never remove or reuse previous evidence."""
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Use a fresh output directory: {output}')
    output.mkdir(parents=True, exist_ok=False)
    return output.resolve()


def source_snapshot(root: Path, execution_files=None) -> dict:
    """Bind HEAD, tree, index/worktree state and exact executing source bytes."""
    root = Path(root).resolve()
    if not (root / '.git').exists() and (root / 'SOURCE_MANIFEST.json').is_file():
        manifest_bytes = (root / 'SOURCE_MANIFEST.json').read_bytes()
        manifest = json.loads(manifest_bytes.decode('utf-8'))
        tracked = sorted(manifest.get('files', {}).keys())
        executable_suffixes = {'.py', '.cmd', '.bat', '.json', '.yml', '.yaml', '.toml', '.txt'}
        names = {str(name) for name in execution_files or ()}
        names.update(name for name in tracked if name and Path(name).suffix.lower() in executable_suffixes)
        def _digest(name):
            p = root / name
            return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
        tracked_sha = {name: _digest(name) for name in tracked if name}
        dirty = [name for name, exp in manifest.get('files', {}).items()
                 if name != 'SOURCE_MANIFEST.json' and tracked_sha.get(name) != exp]
        return {'head': manifest.get('source_commit', 'source-snapshot'),
                'tree': hashlib.sha256(manifest_bytes).hexdigest(),
                'index_entries_sha256': hashlib.sha256(manifest_bytes).hexdigest(),
                'tracked_status': '\0'.join(dirty),
                'tracked_sha256': tracked_sha,
                'execution_sha256': {str(name): _digest(name) for name in sorted(names)}}

    def git(*argv):
        return subprocess.check_output(['git', *argv], cwd=root, stderr=subprocess.PIPE)

    tracked = git('ls-files', '-z').decode('utf-8').split('\0')
    untracked = git('ls-files', '--others', '--exclude-standard', '-z').decode('utf-8').split('\0')
    executable_suffixes = {'.py', '.cmd', '.bat', '.json', '.yml', '.yaml', '.toml', '.txt'}
    names = {str(name) for name in execution_files or ()}
    names.update(name for name in tracked + untracked
                 if name and Path(name).suffix.lower() in executable_suffixes)

    def digest(name):
        path = root / name
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None

    return {'head': git('rev-parse', 'HEAD').decode().strip(),
            'tree': git('rev-parse', 'HEAD^{tree}').decode().strip(),
            'index_entries_sha256': hashlib.sha256(git('ls-files', '--stage', '-z')).hexdigest(),
            'tracked_status': git('status', '--porcelain=v1', '-z', '--untracked-files=no')
                              .decode('utf-8'),
            'tracked_sha256': {name: digest(name) for name in sorted(set(tracked)) if name},
            'execution_sha256': {str(name): digest(name) for name in sorted(names)}}


def compare_snapshots(before: dict, after: dict) -> dict:
    required = {'head', 'tree', 'tracked_status', 'tracked_sha256', 'execution_sha256'}
    missing = {'before': sorted(required - before.keys()),
               'after': sorted(required - after.keys())}
    changed = [key for key in sorted(set(before) | set(after))
               if before.get(key) != after.get(key)]
    return {'passed': not changed and not any(missing.values()), 'changes': changed,
            'missing_fields': missing}


class AcceptanceRun:
    """Persist unfinished immediately; only explicit finalization can produce success."""

    def __init__(self, output: Path, root: Path, *, execution_files=None,
                 report_name: str = 'summary.json'):
        self.output = create_run_directory(output)
        self.root = Path(root)
        self.execution_files = execution_files
        self.path = self.output / report_name
        self.report = {'status': 'unfinished', 'passed': False,
                       'started_utc': datetime.now(timezone.utc).isoformat()}
        self.checkpoint()
        self.report['source_before'] = source_snapshot(self.root, execution_files)
        self.checkpoint()

    def checkpoint(self) -> None:
        atomic_json(self.path, self.report)

    def finish(self, passed: bool) -> dict:
        self.report['passed'] = False
        try:
            self.report['source_after'] = source_snapshot(self.root, self.execution_files)
            self.report['source_integrity'] = compare_snapshots(
                self.report['source_before'], self.report['source_after'])
        except Exception as error:
            self.report['source_integrity'] = {'passed': False,
                                              'error': f'{type(error).__name__}: {error}'}
        self.report['passed'] = bool(passed and self.report['source_integrity']['passed'])
        self.report['status'] = 'completed' if self.report['passed'] else 'failed'
        self.report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        self.checkpoint()
        return self.report


class OwnedProcessTree:
    """Verify PID identities inside a Job that retains even unobserved orphans."""

    def __init__(self, process: subprocess.Popen, *, job=None):
        self.process = process
        self.job = job
        if job is not None:
            self.job_layout = job.layout
        self.identities = {}
        self.identity_history = []
        self.observations = []
        self.errors = []
        if job is None:
            self.errors.append('Missing reliable process container; start with start_owned_process')
        self._remember(process.pid)
        self.capture()

    def _current_member(self, pid: int) -> bool | None:
        """Exclude departed historical PIDs before opening any process handle."""
        job = getattr(self, 'job', None)
        if job is None:
            return True
        try:
            return pid in job.pids()
        except (OSError, RuntimeError) as error:
            self.errors.append(f'Cannot query owned Job membership for PID {pid}: {error}')
            return None

    def _departed_after_error(self, pid: int, operation: str, error: Exception) -> bool:
        """A second kernel membership query must prove a raced PID is no longer ours."""
        if getattr(self, 'job', None) is None or self._current_member(pid) is not False:
            return False
        self.observations.append({'pid': pid, 'operation': operation,
                                  'reason': 'left_job_during_observation',
                                  'error': f'{type(error).__name__}: {error}'})
        return True

    def _remember(self, pid: int, observed=None):
        if self._current_member(pid) is not True:
            return None
        try:
            owned = observed if observed is not None else psutil.Process(pid)
            created = owned.create_time()
            job = getattr(self, 'job', None)
            if job is not None:
                # A PID can legitimately represent a later process in the same
                # Job. Confirm membership between two reads of its identity.
                if not job.contains(pid):
                    return None
                confirmed = psutil.Process(pid)
                if confirmed.create_time() != created:
                    return None  # Exited/reused during observation; capture again.
                owned = confirmed
            elif pid in self.identities and self.identities[pid] != created:
                self.errors.append(f'PID identity changed: {pid}')
                return None
            if self.identities.get(pid) != created:
                self.identity_history.append({'pid': pid, 'create_time': created})
            self.identities[pid] = created
            return owned
        except psutil.NoSuchProcess:
            return None
        except (psutil.Error, OSError) as error:
            if not self._departed_after_error(pid, 'identify', error):
                self.errors.append(f'Cannot identify owned PID {pid}: {error}')
            return None

    def _matching(self, pid: int):
        if self._current_member(pid) is not True:
            return None
        try:
            job = getattr(self, 'job', None)
            if job is not None and not job.contains(pid):
                return None  # The old process exited; an external reuse is not ours.
            process = psutil.Process(pid)
            if process.create_time() != self.identities[pid]:
                if job is None:
                    self.errors.append(f'PID identity changed; no signal sent: {pid}')
                return None
            if process.status() == psutil.STATUS_ZOMBIE:
                return None
            return process
        except psutil.NoSuchProcess:
            return None
        except (psutil.Error, OSError) as error:
            if not self._departed_after_error(pid, 'inspect', error):
                self.errors.append(f'Cannot inspect owned PID {pid}: {error}')
            return None

    def capture(self) -> None:
        if getattr(self, 'job', None) is not None:
            try:
                for pid in self.job.pids():
                    self._remember(pid)
            except (psutil.Error, OSError, RuntimeError) as error:
                self.errors.append(f'Cannot enumerate owned Job: {error}')
            return
        # Follow only relationships from previously confirmed owned processes.
        for pid in list(self.identities):
            process = self._matching(pid)
            if process is None:
                continue
            try:
                for child in process.children(recursive=True):
                    self._remember(child.pid, child)
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as error:
                self.errors.append(f'Cannot inspect descendants of {pid}: {error}')

    def _alive(self):
        if getattr(self, 'job', None) is not None:
            self.capture()
        return [process for pid in self.identities if (process := self._matching(pid))]

    def owns_tcp_listener(self, host: str, port: int) -> bool:
        """Require every matching IPv4 listener to have a verified identity in this Job."""
        if getattr(self, 'job', None) is None:
            self.errors.append('Cannot verify TCP listener without an owned process Job')
            return False
        self.capture()
        if self.errors:
            return False
        addresses = {host, '0.0.0.0'} if host == '127.0.0.1' else {host}
        try:
            # An owned wildcard socket alone is insufficient: another process may
            # own the more specific address. Never signal any PID found here.
            listeners = [connection for connection in psutil.net_connections(kind='tcp4')
                         if connection.status == psutil.CONN_LISTEN and connection.laddr
                         and connection.laddr.ip in addresses and connection.laddr.port == port]
        except (psutil.Error, OSError, RuntimeError) as error:
            self.errors.append(f'Cannot inspect TCP listener ownership: {error}')
            return False
        if not listeners:
            return False
        # Recheck membership and creation time after socket enumeration, which can
        # itself race process exit or PID reuse. Unknown owners cannot count as ours.
        return (all(connection.pid in self.identities
                    and self._matching(connection.pid) is not None
                    for connection in listeners) and not self.errors)

    def cleanup(self, timeout: float = 5.0, kill_timeout: float = 3.0) -> dict:
        self.capture()
        for method, budget in [('terminate', timeout), ('kill', kill_timeout)]:
            self.capture()
            # Signal confirmed descendants before the root. Each signal rechecks identity.
            for pid in reversed(list(self.identities)):
                process = self._matching(pid)
                if process is not None:
                    try:
                        getattr(process, method)()
                    except psutil.NoSuchProcess:
                        pass
                    except psutil.Error as error:
                        self.errors.append(f'{method} failed for owned PID {pid}: {error}')
            if method == 'kill' and getattr(self, 'job', None) is not None:
                try:
                    # The kernel owns membership; this also covers new children
                    # between enumeration and a signal, without a PID reuse race.
                    self.job.terminate()
                except OSError as error:
                    self.errors.append(f'Cannot terminate owned Job: {error}')
            deadline = time.monotonic() + budget
            while self._alive() and time.monotonic() < deadline:
                time.sleep(min(.05, max(0, deadline - time.monotonic())))
            if not self._alive():
                break
        self.process.poll()  # Reap the direct child on platforms that need waitpid.
        remaining = [{'pid': process.pid, 'create_time': self.identities[process.pid]}
                     for process in self._alive()]
        if getattr(self, 'job', None) is not None:
            try:
                remaining_ids = {item['pid'] for item in remaining}
                remaining.extend({'pid': pid, 'create_time': self.identities.get(pid)}
                                 for pid in self.job.pids() if pid not in remaining_ids)
            except (OSError, RuntimeError) as error:
                self.errors.append(f'Cannot verify final owned Job state: {error}')
            try:
                self.job.close()
            except OSError as error:
                self.errors.append(f'Cannot close owned Job: {error}')
            self.job = None
        owned = getattr(self, 'identity_history', [
            {'pid': pid, 'create_time': created} for pid, created in self.identities.items()])
        return {'passed': not remaining and not self.errors,
                'containment': 'windows_job' if hasattr(self, 'job_layout') else 'unconfirmed',
                'job_layout': getattr(self, 'job_layout', None),
                'owned': owned, 'pid_reuses': len(owned) - len(self.identities),
                'observations': getattr(self, 'observations', []),
                'remaining': remaining, 'errors': list(dict.fromkeys(self.errors))}


def run_owned_command(argv: list[str], *, cwd: Path, log_path: Path,
                      timeout: float, env=None) -> dict:
    """Run one bounded command, retaining logs and ownership/cleanup evidence on failure."""
    started = time.monotonic()
    result = {'argv': argv, 'log': str(log_path), 'passed': False, 'returncode': None,
              'timed_out': False}
    process = owner = None
    interruption = None
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as log:
        try:
            process, owner = start_owned_process(
                argv, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            deadline = time.monotonic() + timeout
            while process.poll() is None:
                owner.capture()
                if time.monotonic() >= deadline:
                    result['timed_out'] = True
                    result['error'] = f'Command timed out after {timeout} seconds'
                    break
                time.sleep(.025)
        except OSError as error:
            result['error'] = f'{type(error).__name__}: {error}'
        except (KeyboardInterrupt, SystemExit) as error:
            interruption = error
            result['error'] = f'{type(error).__name__}: {error}'
            result['interrupted'] = True
        finally:
            if owner is not None:
                result['cleanup'] = owner.cleanup()
            else:
                result['cleanup'] = {'passed': process is None, 'owned': [], 'remaining': [],
                                     'errors': []}
            if process is not None:
                result['returncode'] = process.poll()
    result['seconds'] = round(time.monotonic() - started, 3)
    result['passed'] = (result['returncode'] == 0 and not result['timed_out']
                        and 'error' not in result and result['cleanup']['passed'])
    atomic_json(log_path.with_suffix(log_path.suffix + '.json'), result)
    if interruption is not None:
        raise interruption
    return result
