"""Run the bounded engineering gate without training or writing original results.

Run from the repository root: python -m scripts.engineering_ci
The Streamlit health check is only a startup check, never a model or browser E2E claim.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from scripts.engineering_run import AcceptanceRun, run_owned_command, start_owned_process


ROOT = Path(__file__).resolve().parents[1]
BASELINE = '8fc59879e2f498c32915836310af468b4272cfdf'
TEXT_SUFFIXES = {'.py', '.json', '.md', '.xml', '.txt', '.yml', '.yaml'}
TEMP_ROOT = ROOT / 'test-results/engineering/tmp'


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.check_output(['git', *args], cwd=ROOT, stderr=subprocess.PIPE)


def locked_environment() -> dict:
    """Check exact package versions as well as agreement of direct and locked pins."""
    def pins(name: str) -> dict[str, str]:
        result = {}
        for line in (ROOT / name).read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            package, version = line.split('==', 1)
            result[re.sub(r'[-_.]+', '-', package).lower()] = version
        return result

    locked, direct = pins('requirements-lock.txt'), pins('requirements.txt')
    mismatches = []
    installed = {}
    for package, expected in locked.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            actual = None
        installed[package] = actual
        if actual != expected:
            mismatches.append({'package': package, 'expected': expected, 'actual': actual})
    direct_mismatches = [name for name, version in direct.items() if locked.get(name) != version]
    python_supported = sys.version_info[:2] == (3, 12)
    return {'passed': not mismatches and not direct_mismatches and python_supported,
            'python_3_12': python_supported, 'locked_packages': len(locked),
            'installed_versions': installed, 'version_mismatches': mismatches,
            'direct_lock_mismatches': direct_mismatches,
            'lock_sha256': sha256((ROOT / 'requirements-lock.txt').read_bytes())}


def protected_files() -> dict:
    """Use a release manifest only with an independently supplied trusted digest."""
    from scripts.release_source_guard import verify_source
    return verify_source(ROOT, os.environ.get('LIUSHAO_SOURCE_MANIFEST', str(ROOT / 'SOURCE_MANIFEST.json')),
                         os.environ.get('LIUSHAO_SOURCE_MANIFEST_SHA256'))


def run_command(argv: list[str], log_path: Path, timeout: int = 600) -> dict:
    temporary = log_path.parent / 'tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONUTF8': '1',
           'TEMP': str(temporary), 'TMP': str(temporary)}
    if argv[1:3] == ['-m', 'pytest']:
        # The full gate must not inherit a shell's -k/-m selection or other addopts.
        env.pop('PYTEST_ADDOPTS', None)
    return run_owned_command(argv, cwd=ROOT, log_path=log_path, timeout=timeout,
                             env=env)


def test_counts(junit_path: Path) -> dict:
    if not junit_path.is_file():
        return {'available': False, 'valid': False, 'successful': False,
                'reason': 'missing_junit'}
    try:
        root = ET.parse(junit_path).getroot()
        if root.tag not in {'testsuite', 'testsuites'}:
            raise ValueError('JUnit root must be testsuite or testsuites')
        parents = {child: parent for parent in root.iter() for child in parent}
        for element in root.iter():
            parent = parents.get(element)
            if element.tag == 'testcase' and (parent is None or parent.tag != 'testsuite'):
                raise ValueError('JUnit testcase must be a direct child of a testsuite')
            if (element.tag == 'testsuite' and element is not root
                    and (parent is None or parent.tag not in {'testsuite', 'testsuites'})):
                raise ValueError('JUnit testsuite is outside the suite hierarchy')
            if (element.tag in {'failure', 'error', 'skipped'}
                    and (parent is None or parent.tag != 'testcase')):
                raise ValueError(f'JUnit {element.tag} outside a testcase')
        tests = root.findall('.//testcase')
        if not tests:
            raise ValueError('JUnit contains no test cases')
        suites = list(root.iter('testsuite'))
        if not suites:
            raise ValueError('JUnit contains no test suites')
        for suite in suites:
            cases = suite.findall('.//testcase')
            observed = {'tests': len(cases),
                        'failures': sum(case.find('failure') is not None for case in cases),
                        'errors': sum(case.find('error') is not None for case in cases),
                        'skipped': sum(case.find('skipped') is not None for case in cases)}
            for key, value in observed.items():
                if key in suite.attrib and int(suite.attrib[key]) != value:
                    raise ValueError(f'Inconsistent JUnit {key} count')
        if root.tag == 'testsuites':
            observed = {'tests': len(tests),
                        'failures': sum(case.find('failure') is not None for case in tests),
                        'errors': sum(case.find('error') is not None for case in tests),
                        'skipped': sum(case.find('skipped') is not None for case in tests)}
            for key, value in observed.items():
                if key in root.attrib and int(root.attrib[key]) != value:
                    raise ValueError(f'Inconsistent aggregate JUnit {key} count')
    except (OSError, ET.ParseError, ValueError) as error:
        return {'available': True, 'valid': False, 'successful': False,
                'reason': f'{type(error).__name__}: {error}'}
    skipped = [{'name': item.attrib.get('name'), 'class': item.attrib.get('classname'),
                'reason': item.find('skipped').attrib.get('message', '')}
               for item in tests if item.find('skipped') is not None]
    failures = sum(item.find('failure') is not None for item in tests)
    errors = sum(item.find('error') is not None for item in tests)
    failed_cases = [
        {'name': item.attrib.get('name'), 'class': item.attrib.get('classname'), 'kind': kind}
        for item in tests for kind in ('failure', 'error') if item.find(kind) is not None
    ]
    return {'available': True, 'valid': True, 'successful': not failures and not errors,
            'total': len(tests), 'failures': failures, 'errors': errors,
            'passed': len(tests) - failures - errors - len(skipped), 'skipped': skipped,
            'failed_cases': failed_cases}


def startup_check(output: Path, timeout: int = 45) -> dict:
    """Own one dedicated process; do not enumerate, kill, or reuse other servers."""
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        port = listener.getsockname()[1]
    argv = [sys.executable, '-m', 'streamlit', 'run', str(ROOT / 'app.py'),
            '--server.address=127.0.0.1', f'--server.port={port}',
            '--server.headless=true', '--server.fileWatcherType=none',
            '--browser.gatherUsageStats=false']
    result = {'passed': False, 'argv': argv, 'port': port,
              'scope': 'HTTP startup health only; no prediction or browser E2E assertion'}
    process = owner = None
    started = time.monotonic()
    try:
        temporary = output / 'startup-tmp'
        temporary.mkdir()
        env = {**os.environ, 'PYTHONUTF8': '1', 'TEMP': str(temporary), 'TMP': str(temporary),
               'LIUSHAO_REVIEW_DB': str(temporary / 'reviews.sqlite')}
        with (output / 'streamlit.log').open('w', encoding='utf-8') as log:
            process, owner = start_owned_process(
                argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            try:
                # Ignore a workstation HTTP proxy for this loopback-only request.
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    owner.capture()
                    if process.poll() is not None:
                        raise RuntimeError(f'Streamlit exited: {process.returncode}')
                    if not owner.owns_tcp_listener('127.0.0.1', port):
                        time.sleep(0.2)
                        continue
                    try:
                        with opener.open(f'http://127.0.0.1:{port}/_stcore/health',
                                         timeout=2) as response:
                            ready = response.status == 200 and response.read().strip() == b'ok'
                        if ready:
                            time.sleep(0.3)
                            if process.poll() is not None:
                                raise RuntimeError('Owned server exited after health response')
                            if not owner.owns_tcp_listener('127.0.0.1', port):
                                raise RuntimeError('Health listener no longer belongs to owned server')
                            result['passed'] = True
                            break
                    except (urllib.error.URLError, TimeoutError, ConnectionError):
                        pass
                    time.sleep(0.2)
                if not result['passed']:
                    raise TimeoutError(f'No healthy response after {timeout} seconds')
            finally:
                result['cleanup'] = owner.cleanup()
                result['passed'] = result['passed'] and result['cleanup']['passed']
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        result['passed'] = False
        result['error'] = str(error)
    result['owned_process_stopped'] = process is None or process.poll() is not None
    result['seconds'] = round(time.monotonic() - started, 3)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='test-results/engineering-ci')
    args = parser.parse_args()
    if (ROOT / args.output).absolute().exists():
        raise FileExistsError('验收输出目录已存在，不能覆盖：' + args.output)
    if not os.environ.get('LIUSHAO_SOURCE_MANIFEST_SHA256'):
        print(json.dumps({'status': 'not_run', 'passed': False,
              'missing': ['LIUSHAO_SOURCE_MANIFEST_SHA256 (独立保存的发布源码清单摘要)']}, ensure_ascii=False))
        return 2
    output = (ROOT / args.output).absolute()
    run = AcceptanceRun(output, ROOT)
    report = run.report
    report.update({'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'git_head': report['source_before'].get('head'),
              'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'checks': {},
              'limits': ['Startup health is not browser E2E or successful inference.',
                         'Skipped real-model tests are not verified.',
                         'No training, independent dataset evaluation, or timing benefit is claimed.']})
    checks = report['checks']
    required = {'locked_environment', 'protected_before', 'pip_check', 'pytest',
                'junit', 'startup', 'protected_after'}
    run.checkpoint()
    try:
        checks['locked_environment'] = locked_environment()
        run.checkpoint()
        checks['protected_before'] = protected_files()
        run.checkpoint()
        checks['pip_check'] = run_command([sys.executable, '-m', 'pip', 'check'],
                                          output / 'pip-check.log', timeout=120)
        junit = output / 'pytest.xml'
        run.checkpoint()
        checks['pytest'] = run_command([sys.executable, '-m', 'pytest', 'tests', '-q', '-ra',
                                        f'--basetemp={output / "pytest-tmp"}',
                                        f'--junitxml={junit}'], output / 'pytest.log')
        report['test_counts'] = test_counts(junit)
        checks['junit'] = {'passed': report['test_counts']['successful']}
        run.checkpoint()
        checks['startup'] = startup_check(output)
        run.checkpoint()
        checks['protected_after'] = protected_files()
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        report['runner_error'] = f'{type(error).__name__}: {error}'
    report['missing_stages'] = sorted(required - checks.keys())
    command_exits_valid = all(checks.get(stage, {}).get('returncode') == 0
                              for stage in ('pip_check', 'pytest'))
    passed = (not report['missing_stages'] and all(item.get('passed') is True
              for item in checks.values()) and command_exits_valid
              and 'runner_error' not in report)
    report = run.finish(passed)
    print(json.dumps({'passed': report['passed'], 'checks': {
        name: value['passed'] for name, value in checks.items()},
        'test_counts': report.get('test_counts'),
        'runner_error': report.get('runner_error'),
        'report': str(output / 'summary.json')}, ensure_ascii=False, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
