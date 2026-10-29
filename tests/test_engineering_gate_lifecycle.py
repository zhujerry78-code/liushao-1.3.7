"""Acceptance cannot turn stale, partial or drifting evidence into a green run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import psutil
import pytest

from scripts import engineering_ci as ci


def fixture_snapshot(*args, **kwargs):
    return {'head': 'fixture', 'tree': 'fixture', 'tracked_status': '',
            'tracked_sha256': {'app.py': 'fixture'},
            'execution_sha256': {'app.py': 'fixture'}}


@pytest.mark.parametrize('content', [b'<broken', b'<testsuites/>',
                                      b'<testsuites><testsuite tests="1" errors="1">'
                                      b'<testcase name="x"/></testsuite></testsuites>'])
def test_invalid_or_inconsistent_junit_cannot_pass(tmp_path, content):
    path = tmp_path / 'pytest.xml'
    path.write_bytes(content)
    result = ci.test_counts(path)
    assert result.get('successful') is False


def test_junit_failure_summary_exposes_only_case_identity(tmp_path):
    path = tmp_path / 'pytest.xml'
    payload = 'PRIVATE_FIXTURE_PAYLOAD'
    path.write_text(
        '<testsuite tests="4" failures="1" errors="1" skipped="1">'
        f'<testcase name="test_failure[case-a]" classname="tests.test_example">'
        f'<failure message="{payload}" type="{payload}">{payload}</failure>'
        f'<system-out>{payload}</system-out><system-err>{payload}</system-err>'
        f'<properties><property name="locals" value="{payload}"/></properties>'
        '</testcase><testcase name="test_setup_error" classname="tests.test_example">'
        f'<error message="{payload}">{payload}</error></testcase>'
        '<testcase name="test_ok" classname="tests.test_example"/>'
        '<testcase name="test_skip" classname="tests.test_example">'
        '<skipped message="fixture skipped"/></testcase></testsuite>',
        encoding='utf-8',
    )

    result = ci.test_counts(path)

    assert result['valid'] and not result['successful']
    assert (result['total'], result['passed'], result['failures'], result['errors']) == (
        4, 1, 1, 1,
    )
    assert result['failed_cases'] == [
        {'name': 'test_failure[case-a]', 'class': 'tests.test_example', 'kind': 'failure'},
        {'name': 'test_setup_error', 'class': 'tests.test_example', 'kind': 'error'},
    ]
    assert payload not in json.dumps(result)


def test_successful_junit_summary_has_no_failed_cases(tmp_path):
    path = tmp_path / 'pytest.xml'
    path.write_text('<testsuite tests="1"><testcase name="test_ok"/></testsuite>',
                    encoding='utf-8')

    result = ci.test_counts(path)

    assert result['successful'] and result['passed'] == 1
    assert result['failed_cases'] == []


def test_existing_output_is_never_reused(tmp_path, monkeypatch):
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'summary.json'
    marker.write_text('{"passed": true}', encoding='utf-8')
    monkeypatch.setattr(sys, 'argv', ['engineering_ci', '--output', str(output)])
    with pytest.raises((FileExistsError, SystemExit)):
        ci.main()
    assert marker.read_text(encoding='utf-8') == '{"passed": true}'


def test_output_snapshot_initializes_unfinished_and_detects_execution_drift(tmp_path):
    from scripts.engineering_run import AcceptanceRun
    repo = tmp_path / 'repo'
    repo.mkdir()
    subprocess.run(['git', 'init', '-q'], cwd=repo, check=True)
    source = repo / 'runner.py'
    source.write_text('print(1)\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'runner.py'], cwd=repo, check=True)
    subprocess.run(['git', '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.com',
                    'commit', '-qm', 'fixture'], cwd=repo, check=True)
    run = AcceptanceRun(tmp_path / 'evidence', repo)
    initial = json.loads((run.output / 'summary.json').read_text(encoding='utf-8'))
    assert initial['status'] == 'unfinished' and initial['passed'] is False
    source.write_text('print(2)\n', encoding='utf-8')
    report = run.finish(True)
    assert not report['passed'] and report['status'] == 'failed'
    assert not report['source_integrity']['passed']


def test_changed_index_is_detected_even_when_status_and_worktree_match(tmp_path):
    from scripts.engineering_run import compare_snapshots, source_snapshot
    subprocess.run(['git', 'init', '-q'], cwd=tmp_path, check=True)
    source = tmp_path / 'runner.py'
    source.write_text('print(1)\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'runner.py'], cwd=tmp_path, check=True)
    subprocess.run(['git', '-c', 'user.name=fixture', '-c', 'user.email=fixture@example.com',
                    'commit', '-qm', 'fixture'], cwd=tmp_path, check=True)
    source.write_text('print(2)\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'runner.py'], cwd=tmp_path, check=True)
    source.write_text('print(4)\n', encoding='utf-8')
    before = source_snapshot(tmp_path)
    source.write_text('print(3)\n', encoding='utf-8')
    subprocess.run(['git', 'add', 'runner.py'], cwd=tmp_path, check=True)
    source.write_text('print(4)\n', encoding='utf-8')
    after = source_snapshot(tmp_path)
    assert before['tracked_status'] == after['tracked_status']
    assert before['execution_sha256'] == after['execution_sha256']
    result = compare_snapshots(before, after)
    assert not result['passed'] and result['changes'] == ['index_entries_sha256']


def test_timeout_stops_only_owned_parent_and_child_and_retains_logs(tmp_path):
    from scripts.engineering_run import run_owned_command
    child_file = tmp_path / 'child.json'
    script = ("import subprocess,sys,time,json,pathlib; "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
              f"pathlib.Path({str(child_file)!r}).write_text(json.dumps(p.pid)); "
              "print('fixture started',flush=True); time.sleep(60)")
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
    try:
        result = run_owned_command([sys.executable, '-c', script], cwd=tmp_path,
                                   log_path=tmp_path / 'command.log', timeout=1)
        assert not result['passed'] and result['timed_out']
        assert result['cleanup']['passed'] and not result['cleanup']['remaining']
        child_pid = json.loads(child_file.read_text())
        assert not psutil.pid_exists(child_pid)
        assert unrelated.poll() is None
        assert 'fixture started' in (tmp_path / 'command.log').read_text()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_fast_exiting_parent_cannot_leave_an_unobserved_child_green(tmp_path):
    from scripts.engineering_run import run_owned_command
    script = ("import subprocess,sys,time,psutil; time.sleep(.05); "
              "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(2)']); "
              "print(p.pid,psutil.Process(p.pid).create_time(),flush=True)")
    survivors = []
    try:
        for attempt in range(4):
            log = tmp_path / f'fast-exit-{attempt}.log'
            result = run_owned_command([sys.executable, '-c', script], cwd=tmp_path,
                                       log_path=log, timeout=5)
            pid_text, created_text = log.read_text().strip().split()
            pid, created = int(pid_text), float(created_text)
            try:
                child = psutil.Process(pid)
                alive = child.create_time() == created and child.is_running()
            except psutil.NoSuchProcess:
                alive = False
            if alive:
                survivors.append(child)
            assert not (result['passed'] and alive), result
            assert not alive, 'Owned orphan must actually be stopped before runner returns'
    finally:
        for child in survivors:
            try:
                descendants = child.children(recursive=True)
                for owned in reversed(descendants + [child]):
                    if owned.is_running():
                        owned.kill()
            except psutil.NoSuchProcess:
                pass


@pytest.mark.parametrize('content', [
    b'<testsuite tests="1"><testcase name="ok"/><error message="collection failed"/></testsuite>',
    b'<testsuites><testsuite tests="0"/><testcase name="orphan"/></testsuites>',
    b'<testsuite tests="1"><system-out><testcase name="not-a-result"/></system-out></testsuite>',
])
def test_structurally_invalid_junit_cannot_report_success(tmp_path, content):
    path = tmp_path / 'invalid-structure.xml'
    path.write_bytes(content)
    assert ci.test_counts(path)['successful'] is False


def test_job_assignment_failure_stops_suspended_root_without_running_code(tmp_path, monkeypatch):
    from scripts import engineering_run
    marker = tmp_path / 'must-not-run.txt'
    roots = []

    def fail_assignment(job, process):
        roots.append(process)
        raise OSError('fixture Job assignment failure')

    monkeypatch.setattr(engineering_run._WindowsJob, 'assign', fail_assignment)
    with pytest.raises(OSError, match='fixture Job assignment failure'):
        engineering_run.start_owned_process([sys.executable, '-c',
                                             f'open({str(marker)!r}, "w").write("ran")'])
    assert len(roots) == 1 and roots[0].poll() is not None
    assert not marker.exists()


def test_killing_runner_closes_job_and_stops_owned_children(tmp_path):
    from scripts import engineering_run
    repo = Path(ci.__file__).resolve().parents[1]
    ready = tmp_path / 'ready.json'
    children = tmp_path / 'children.json'
    evidence = tmp_path / 'hard-kill-job'
    child_script = (
        "import subprocess,sys,time,json,os,psutil,pathlib; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
        f"pathlib.Path({str(children)!r}).write_text(json.dumps("
        "[[os.getpid(),psutil.Process().create_time()],"
        "[p.pid,psutil.Process(p.pid).create_time()]])); time.sleep(60)")
    script = (
        "import sys,time,json,pathlib,os,psutil; "
        "from scripts.engineering_run import AcceptanceRun,start_owned_process; "
        f"run=AcceptanceRun({str(evidence)!r},{str(repo)!r}); "
        f"process,owner=start_owned_process([sys.executable,'-c',{child_script!r}]); "
        f"pathlib.Path({str(ready)!r}).write_text(json.dumps("
        "[os.getpid(),psutil.Process().create_time()])); time.sleep(60)")
    launcher = subprocess.Popen([sys.executable, '-c', script], cwd=repo,
                                env={**os.environ, 'PYTHONPATH': str(repo)})
    runner = None
    child_identities = []
    try:
        deadline = time.monotonic() + 10
        while (not ready.is_file() or not children.is_file()) and time.monotonic() < deadline:
            time.sleep(.05)
        assert ready.is_file() and children.is_file()
        runner_pid, runner_created = json.loads(ready.read_text())
        runner = psutil.Process(runner_pid)
        assert runner.create_time() == runner_created
        child_identities = json.loads(children.read_text())
        runner.kill()  # Kill the actual interpreter, not a venv launcher parent.
        launcher.wait(timeout=10)
        deadline = time.monotonic() + 5
        alive = []
        while time.monotonic() < deadline:
            alive = []
            for pid, created in child_identities:
                try:
                    process = psutil.Process(pid)
                    if process.create_time() == created and process.is_running():
                        alive.append(pid)
                except psutil.NoSuchProcess:
                    pass
            if not alive:
                break
            time.sleep(.05)
        assert not alive, f'Owned Job survived loss of its sole owner handle: {alive}'
        report = json.loads((evidence / 'summary.json').read_text(encoding='utf-8'))
        assert report['status'] == 'unfinished' and report['passed'] is False
    finally:
        if runner is not None and runner.is_running():
            runner.kill()
        if launcher.poll() is None:
            launcher.kill()
            launcher.wait(timeout=10)
        for pid, created in child_identities:
            try:
                process = psutil.Process(pid)
                if process.create_time() == created:
                    process.kill()
            except psutil.NoSuchProcess:
                pass


def test_cleanup_failure_cannot_pass_a_zero_exit_command(tmp_path, monkeypatch):
    from scripts import engineering_run
    monkeypatch.setattr(engineering_run.OwnedProcessTree, 'cleanup', lambda *a, **kw: {
        'passed': False, 'remaining': [{'pid': 123}], 'errors': ['fixture cleanup failure']})
    result = engineering_run.run_owned_command([sys.executable, '-c', 'pass'], cwd=tmp_path,
                                             log_path=tmp_path / 'command.log', timeout=5)
    assert result['returncode'] == 0 and not result['passed']


def test_early_stage_exception_keeps_failed_complete_report(tmp_path, monkeypatch):
    monkeypatch.setenv('LIUSHAO_SOURCE_MANIFEST_SHA256', '0'*64)
    from scripts import engineering_run
    monkeypatch.setattr(engineering_run, 'source_snapshot', fixture_snapshot)
    monkeypatch.setattr(ci, 'locked_environment', lambda: (_ for _ in ()).throw(
        RuntimeError('fixture early stage failure')))
    monkeypatch.setattr(sys, 'argv', ['engineering_ci', '--output', str(tmp_path / 'run')])
    assert ci.main() == 1
    report = json.loads((tmp_path / 'run' / 'summary.json').read_text(encoding='utf-8'))
    assert report['status'] == 'failed' and report['passed'] is False
    assert 'fixture early stage failure' in report['runner_error']
    assert report['missing_stages']


@pytest.mark.parametrize(('junit', 'exit_code', 'expected'), [
    (None, 0, False), (b'<broken', 0, False), (b'<testsuites/>', 0, False),
    (b'<testsuite tests="1" failures="1">'
     b'<testcase name="test_failure" classname="tests.test_example">'
     b'<failure message="PRIVATE_GATE_PAYLOAD">PRIVATE_GATE_PAYLOAD</failure>'
     b'</testcase></testsuite>',
     0, False),
    (b'<testsuite tests="1"><testcase/></testsuite>', 3, False),
    (b'<testsuite tests="1"><testcase/></testsuite>', 0, True),
])
def test_full_gate_requires_valid_junit_and_zero_exit(tmp_path, monkeypatch, capsys,
                                                     junit, exit_code, expected):
    monkeypatch.setenv('LIUSHAO_SOURCE_MANIFEST_SHA256', '0'*64)
    from scripts import engineering_run
    monkeypatch.setattr(engineering_run, 'source_snapshot', fixture_snapshot)
    monkeypatch.setattr(ci, 'locked_environment', lambda: {'passed': True})
    monkeypatch.setattr(ci, 'protected_files', lambda: {'passed': True})
    monkeypatch.setattr(ci, 'startup_check', lambda *a: {'passed': True})

    def command(argv, log, timeout=600):
        if '-m' in argv and 'pytest' in argv and junit is not None:
            (log.parent / 'pytest.xml').write_bytes(junit)
        # Deliberately inconsistent flag verifies the gate checks the actual exit, too.
        return {'passed': True, 'returncode': exit_code if 'pytest' in argv else 0}

    monkeypatch.setattr(ci, 'run_command', command)
    monkeypatch.setattr(sys, 'argv', ['engineering_ci', '--output', str(tmp_path / 'run')])
    assert ci.main() == (0 if expected else 1)
    report = json.loads((tmp_path / 'run' / 'summary.json').read_text(encoding='utf-8'))
    assert report['passed'] is expected
    stdout = capsys.readouterr().out
    assert json.loads(stdout)['test_counts'] == report['test_counts']
    assert 'PRIVATE_GATE_PAYLOAD' not in stdout


def test_atomic_failure_preserves_last_whole_report(tmp_path, monkeypatch):
    from scripts import engineering_run
    path = tmp_path / 'summary.json'
    engineering_run.atomic_json(path, {'status': 'unfinished'})
    monkeypatch.setattr(engineering_run.os, 'replace', lambda *a: (_ for _ in ()).throw(
        OSError('fixture replace failure')))
    with pytest.raises(OSError, match='replace failure'):
        engineering_run.atomic_json(path, {'passed': True})
    assert json.loads(path.read_text()) == {'status': 'unfinished'}
    assert sorted(item.name for item in tmp_path.iterdir()) == ['summary.json']


def test_missing_snapshot_fields_cannot_finish_green(tmp_path, monkeypatch):
    from scripts import engineering_run
    monkeypatch.setattr(engineering_run, 'source_snapshot', lambda *a, **kw: {'head': 'fixture'})
    run = engineering_run.AcceptanceRun(tmp_path / 'incomplete', tmp_path)
    report = run.finish(True)
    assert not report['passed']
    assert 'tree' in report['source_integrity']['missing_fields']['before']


def test_cleanup_does_not_signal_a_reused_pid(monkeypatch):
    from scripts import engineering_run
    signals = []
    fake = SimpleNamespace(create_time=lambda: 2, terminate=lambda: signals.append('terminate'),
                           kill=lambda: signals.append('kill'))
    monkeypatch.setattr(engineering_run.psutil, 'Process', lambda pid: fake)
    owner = object.__new__(engineering_run.OwnedProcessTree)
    owner.identities, owner.errors = {12345: 1}, []
    owner.process = SimpleNamespace(poll=lambda: 0)
    result = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert not result['passed'] and result['errors']
    assert signals == []


@pytest.fixture
def reused_job_pid(monkeypatch):
    """Deterministically reuse one PID without relying on OS allocation timing."""
    from scripts import engineering_run
    state = {'created': 1, 'member': True, 'signals': []}
    pid = 12345

    def signal(name):
        state['signals'].append((name, state['created']))
        state['member'] = False

    process = SimpleNamespace(pid=pid, create_time=lambda: state['created'],
                              status=lambda: 'running',
                              terminate=lambda: signal('terminate'), kill=lambda: signal('kill'))
    monkeypatch.setattr(engineering_run.psutil, 'Process', lambda observed_pid: process)
    job = SimpleNamespace(layout={'fixture': True},
                          pids=lambda: [pid] if state['member'] else [],
                          contains=lambda observed_pid: state['member'],
                          terminate=lambda: signal('job terminate'), close=lambda: None)
    root = SimpleNamespace(pid=pid, poll=lambda: 0)
    return engineering_run.OwnedProcessTree(root, job=job), state


def test_same_job_pid_reuse_records_both_generations_and_cleans_latest(reused_job_pid):
    owner, state = reused_job_pid
    state['created'] = 2
    owner.capture()
    assert owner.identities[12345] == 2 and not owner.errors
    report = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert report['passed'] and not report['remaining']
    assert report['owned'] == [{'pid': 12345, 'create_time': 1},
                               {'pid': 12345, 'create_time': 2}]
    assert report['pid_reuses'] == 1
    assert state['signals'] == [('terminate', 2)]


def test_pid_reused_outside_job_is_ignored_without_signals_or_false_failure(reused_job_pid):
    owner, state = reused_job_pid
    state.update(created=2, member=False)
    report = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert report['passed'] and not report['errors'] and not report['remaining']
    assert state['signals'] == []
    assert report['owned'] == [{'pid': 12345, 'create_time': 1}]


def test_new_generation_requires_another_job_observation_before_signalling(reused_job_pid):
    owner, state = reused_job_pid
    state['created'] = 2
    assert owner._matching(12345) is None
    assert not owner.errors and not state['signals']
    assert owner.identities[12345] == 1
    owner.capture()
    assert owner.identities[12345] == 2


def test_unverifiable_job_membership_still_fails_closed(reused_job_pid, monkeypatch):
    owner, state = reused_job_pid

    def inaccessible(pid):
        raise psutil.AccessDenied(pid)

    monkeypatch.setattr(owner.job, 'contains', inaccessible)
    report = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert not report['passed'] and report['errors']
    assert not any(name in {'terminate', 'kill'} for name, _ in state['signals'])


def test_departed_historical_pid_is_not_opened_when_external_reuse_denies_access(
    reused_job_pid, monkeypatch,
):
    owner, state = reused_job_pid
    state.update(created=2, member=False)
    attempted = []

    def denied(pid):
        attempted.append(pid)
        raise OSError(5, 'fixture external process denies access')

    monkeypatch.setattr(owner.job, 'contains', denied)
    report = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert report['passed'] and not report['errors'] and not report['remaining']
    assert attempted == [] and state['signals'] == []


@pytest.mark.parametrize('operation', ['_remember', '_matching'])
def test_process_query_error_is_only_observation_after_job_proves_pid_departed(
    reused_job_pid, monkeypatch, operation,
):
    owner, state = reused_job_pid

    def exited_during_query(pid):
        state['member'] = False
        raise OSError(5, 'fixture PID exited and was reused outside the Job')

    monkeypatch.setattr(owner.job, 'contains', exited_during_query)
    assert getattr(owner, operation)(12345) is None
    assert not owner.errors and not state['signals']
    assert owner.observations[-1]['pid'] == 12345
    assert owner.observations[-1]['reason'] == 'left_job_during_observation'
    assert 'fixture PID exited' in owner.observations[-1]['error']


def test_failed_authoritative_job_query_cannot_match_or_signal_a_pid(
    reused_job_pid, monkeypatch,
):
    owner, state = reused_job_pid

    def unavailable():
        raise OSError(5, 'fixture Job query unavailable')

    monkeypatch.setattr(owner.job, 'pids', unavailable)
    assert owner._matching(12345) is None
    assert owner.errors and 'fixture Job query unavailable' in owner.errors[-1]
    assert not state['signals']


def test_remaining_process_fails_bounded_cleanup(monkeypatch):
    from scripts import engineering_run
    fake = SimpleNamespace(pid=12345, terminate=lambda: None, kill=lambda: None)
    owner = object.__new__(engineering_run.OwnedProcessTree)
    owner.identities, owner.errors = {12345: 1}, []
    owner.process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(owner, 'capture', lambda: None)
    monkeypatch.setattr(owner, '_matching', lambda pid: fake)
    monkeypatch.setattr(owner, '_alive', lambda: [fake])
    start = time.monotonic()
    result = owner.cleanup(timeout=.01, kill_timeout=.01)
    assert not result['passed'] and result['remaining'] == [{'pid': 12345, 'create_time': 1}]
    assert time.monotonic() - start < 1


def test_interrupt_preserves_command_cleanup_evidence(tmp_path, monkeypatch):
    from scripts import engineering_run
    original = engineering_run.OwnedProcessTree.capture
    calls = 0

    def interrupt_once(owner):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt('fixture interrupt')
        return original(owner)

    monkeypatch.setattr(engineering_run.OwnedProcessTree, 'capture', interrupt_once)
    with pytest.raises(KeyboardInterrupt, match='fixture interrupt'):
        engineering_run.run_owned_command([sys.executable, '-c', 'import time; time.sleep(60)'],
                                         cwd=tmp_path, log_path=tmp_path / 'command.log', timeout=5)
    result = json.loads((tmp_path / 'command.log.json').read_text())
    assert not result['passed'] and result['interrupted']
    assert result['cleanup']['passed'] and not result['cleanup']['remaining']


def test_hard_termination_can_only_leave_unfinished_evidence(tmp_path):
    # This child owns no server or grandchildren. Simulate loss of the runner itself.
    repo = Path(ci.__file__).resolve().parents[1]
    output = tmp_path / 'hard-stop'
    script = ('from scripts.engineering_run import AcceptanceRun; import time; '
              f'run=AcceptanceRun({str(output)!r}, {str(repo)!r}); time.sleep(60)')
    process = subprocess.Popen([sys.executable, '-c', script], cwd=repo,
                               env={**os.environ, 'PYTHONPATH': str(repo)})
    try:
        deadline = time.monotonic() + 10
        while not (output / 'summary.json').is_file() and time.monotonic() < deadline:
            time.sleep(.05)
        assert (output / 'summary.json').is_file()
        process.kill()
        process.wait(timeout=10)
        report = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
        assert report['status'] == 'unfinished' and report['passed'] is False
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
