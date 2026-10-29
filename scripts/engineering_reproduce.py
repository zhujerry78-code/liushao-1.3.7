"""Replay the frozen original CLIs in an owned workspace, never in the source tree.

python -m scripts.engineering_reproduce --stage prepare
python -m scripts.engineering_reproduce --stage train --reuse
python -m scripts.engineering_reproduce --stage tests --reuse
"""
from __future__ import annotations

import argparse
import io
import json
import os
import platform
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from scripts.engineering_ci import BASELINE, ROOT, git, locked_environment, protected_files
from scripts.engineering_ci import sha256, test_counts


WORK = ROOT / 'test-results/engineering/replay'
OUTPUT = ROOT / 'test-results/engineering/reproduction'
MARKER = '.engineering-reproduction.json'
GENERATED_REPORTS = {'docs/results/data_summary.json', 'docs/results/metrics.json'}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def baseline_json(name: str) -> dict:
    return json.loads(git('show', f'{BASELINE}:{name}'))


def source_archive() -> dict[str, bytes]:
    """Read only regular files from the named Git commit; never extract symlinks."""
    content = git('archive', '--format=tar', BASELINE)
    files = {}
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:') as archive:
        for member in archive:
            if member.isdir():
                continue
            name = PurePosixPath(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts:
                raise ValueError(f'Unsupported archive member: {member.name}')
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f'No archive file content: {member.name}')
            files[name.as_posix()] = stream.read()
    return files


def prepare_workspace(reuse: bool) -> dict:
    files = source_archive()
    expected = {'baseline_commit': BASELINE,
                'source_sha256': {name: sha256(content) for name, content in files.items()},
                'generated_report_exceptions': sorted(GENERATED_REPORTS)}
    if WORK.exists():
        if not reuse:
            raise FileExistsError('Replay directory already exists; verified continuation requires --reuse')
        marker = WORK / MARKER
        if not marker.is_file() or json.loads(marker.read_text(encoding='utf-8')) != expected:
            raise ValueError('Replay directory ownership marker or source manifest does not match')
        changed = []
        for name, content in files.items():
            if name in GENERATED_REPORTS:
                continue
            path = WORK / name
            if not path.is_file():
                changed.append(name)
                continue
            actual = path.read_bytes()
            # The original write_json uses platform newlines when rewriting its identical lock.
            same_content = actual == content
            if name == 'configs/data.lock.json':
                same_content = same_content or (
                    actual.replace(b'\r\n', b'\n') == content.replace(b'\r\n', b'\n'))
            if not same_content:
                changed.append(name)
        unexpected_python = [str(path.relative_to(WORK)) for path in WORK.rglob('*.py')
                             if path.relative_to(WORK).as_posix() not in files]
        if changed or unexpected_python:
            raise ValueError(f'Replay source changed: {changed}; extra Python: {unexpected_python}')
        return {'reused': True, 'source_file_count': len(files), 'source_hashes_verified': True}
    WORK.parent.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=False)
    for name, content in files.items():
        path = WORK / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    write_json(WORK / MARKER, expected)
    return {'reused': False, 'source_file_count': len(files), 'source_hashes_verified': True}


def run_command(stage: str, argv: list[str], run_id: str, timeout: int = 7200) -> dict:
    log_path = OUTPUT / f'{run_id}-{stage}.log'
    result = {'stage': stage, 'argv': argv, 'cwd': str(WORK),
              'log': str(log_path.relative_to(ROOT))}
    started = time.monotonic()
    print(json.dumps({'running': stage, 'log': result['log']}, ensure_ascii=False), flush=True)
    # Force child imports to the frozen copy even if the caller has a PYTHONPATH set.
    temporary = WORK / 'test-results/tmp'
    temporary.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, 'PYTHONUTF8': '1', 'PYTHONPATH': str(WORK),
           'TEMP': str(temporary), 'TMP': str(temporary),
           'LIUSHAO_REVIEW_DB': str(WORK / 'test-results/reproduction-reviews.sqlite')}
    with log_path.open('w', encoding='utf-8') as log:
        try:
            completed = subprocess.run(argv, cwd=WORK, env=env, stdout=log,
                                       stderr=subprocess.STDOUT, timeout=timeout)
            result.update(returncode=completed.returncode, passed=completed.returncode == 0)
        except (OSError, subprocess.TimeoutExpired) as error:
            result.update(passed=False, error=str(error))
    result.update(seconds=round(time.monotonic() - started, 3),
                  log_sha256=sha256(log_path.read_bytes()))
    return result


def compare_data() -> dict:
    expected = baseline_json('docs/results/data_summary.json')
    path = WORK / 'data/processed/summary.json'
    if not path.is_file():
        return {'passed': False, 'reason': 'prepared data summary missing'}
    actual = json.loads(path.read_text(encoding='utf-8'))
    checks, hashes = {}, {}
    for name in ['features.parquet', 'manifest.parquet', 'rejected.csv']:
        file_path = path.parent / name
        digest = sha256(file_path.read_bytes()) if file_path.is_file() else None
        expected_hash = expected[name + '_sha256']
        hashes[name] = {'actual': digest, 'expected': expected_hash}
        checks[name] = digest == expected_hash == actual.get(name + '_sha256')
    for name in ['data_version', 'config', 'sources', 'features', 'split_counts',
                 'split_labels', 'split_blocks', 'cross_split_overlap_after_purge']:
        checks[name] = actual.get(name) == expected[name]
    return {'passed': all(checks.values()), 'checks': checks, 'hashes': hashes,
            'data_version': actual.get('data_version'),
            'expected_data_version': expected['data_version'],
            'split_counts': actual.get('split_counts'),
            'elapsed_seconds': actual.get('elapsed_seconds')}


def compare_model() -> dict:
    expected = baseline_json('docs/results/metrics.json')
    directory = WORK / 'artifacts'
    if not all((directory / name).is_file()
               for name in ['metrics.json', 'model.json', 'model.joblib']):
        return {'passed': False, 'reason': 'trained model or metrics missing'}
    actual = json.loads((directory / 'metrics.json').read_text(encoding='utf-8'))
    descriptor = json.loads((directory / 'model.json').read_text(encoding='utf-8'))
    keys = ['model_version', 'data_version', 'rf', 'lr', 'test_per_class',
            'unjudgeable_test_rows', 'ranking_original', 'ranking_score', 'review']
    checks = {name: actual.get(name) == expected[name] for name in keys}
    digest = sha256((directory / 'model.joblib').read_bytes())
    checks['model_sha256_matches_descriptor'] = digest == descriptor.get('model_sha256')
    checks['descriptor_model_version'] = descriptor.get('model_version') == expected['model_version']
    config = descriptor.get('config', {})
    checks['descriptor_data_version'] = config.get('data_version') == expected['data_version']
    for model in ['rf', 'lr']:
        checks[f'{model}_frozen_threshold'] = config.get(f'{model}_threshold') == expected[model]['threshold']
    historical = baseline_json('docs/phase2/audit.json')['original_file_hashes']
    historical_digest = historical['artifacts\\model.joblib']
    return {'passed': all(checks.values()), 'checks': checks,
            'model_version': actual.get('model_version'), 'data_version': actual.get('data_version'),
            'model_sha256': digest, 'historical_model_sha256': historical_digest,
            'serialized_bytes_equal': digest == historical_digest,
            'metrics': {name: actual.get(name) for name in ['rf', 'lr', 'test_per_class']},
            'timing': actual.get('timing'), 'machine': actual.get('machine'),
            'allowed_differences': ['timing', 'machine', 'serialized model bytes'],
            'comparison': 'Exact data/model versions, thresholds, classification metrics and counts'}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'train', 'tests', 'all'], default='all')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    report = {'run_id': run_id, 'baseline_commit': BASELINE, 'requested_stage': args.stage,
              'workspace': str(WORK), 'python': sys.version, 'executable': sys.executable,
              'platform': platform.platform(), 'commands': [], 'passed': False,
              'limits': ['Frozen original experiment replay, not an independent new scenario.',
                         'No changed sample, split, threshold, model logic or tuning.',
                         'Tests do not replace a real-browser E2E regression.']}
    try:
        report['protected_before'] = protected_files()
        if not report['protected_before']['passed']:
            raise ValueError('Original protected files already differ from baseline')
        report['locked_environment'] = locked_environment()
        if not report['locked_environment']['passed']:
            raise ValueError('Python or installed packages do not match the existing lock')
        report['workspace_source'] = prepare_workspace(args.reuse)
        stages = ['prepare', 'train', 'tests'] if args.stage == 'all' else [args.stage]
        for stage in stages:
            if stage in ['train', 'tests']:
                report['data_comparison'] = compare_data()
                if not report['data_comparison']['passed']:
                    raise ValueError('Prepared data does not reproduce the frozen hashes/version')
            if stage == 'prepare':
                argv = [sys.executable, '-m', 'scripts.prepare_data',
                        '--config', 'configs/data.lock.json']
            elif stage == 'train':
                argv = [sys.executable, '-m', 'scripts.train']
            else:
                report['model_comparison'] = compare_model()
                if not report['model_comparison']['passed']:
                    raise ValueError('Model results do not reproduce frozen versions/metrics')
                junit = OUTPUT / f'{run_id}-tests.xml'
                argv = [sys.executable, '-m', 'pytest', 'tests', '-q', '-ra',
                        f'--basetemp={WORK / "test-results" / (run_id + "-pytest")}',
                        f'--junitxml={junit}']
            command = run_command(stage, argv, run_id, timeout=600 if stage == 'tests' else 7200)
            report['commands'].append(command)
            if not command['passed']:
                raise RuntimeError(f'{stage} failed; inspect {command["log"]}')
            if stage == 'prepare':
                report['data_comparison'] = compare_data()
                if not report['data_comparison']['passed']:
                    raise ValueError('Original prepare CLI ran but data/version comparison failed')
            elif stage == 'train':
                report['model_comparison'] = compare_model()
                if not report['model_comparison']['passed']:
                    raise ValueError('Original train CLI ran but model comparison failed')
            else:
                report['test_counts'] = test_counts(junit)
                counts = report['test_counts']
                if not counts.get('available') or counts.get('passed') != 21 or counts.get('skipped'):
                    raise ValueError('Original 21 tests must all pass with no skipped real-model test')
        report['workspace_source_after'] = prepare_workspace(reuse=True)
        report['passed'] = True
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        try:
            report['protected_after'] = protected_files()
            if not report['protected_after']['passed']:
                report['passed'] = False
        except Exception as error:
            report['passed'] = False
            report['protection_error'] = f'{type(error).__name__}: {error}'
        report['full_reproduction_verified'] = bool(
            report['passed'] and report.get('data_comparison', {}).get('passed')
            and report.get('model_comparison', {}).get('passed')
            and report.get('test_counts', {}).get('passed') == 21)
        write_json(OUTPUT / f'{run_id}.json', report)
        write_json(OUTPUT / 'summary.json', report)
    print(json.dumps({'passed': report['passed'], 'stage': args.stage,
                      'full_reproduction_verified': report['full_reproduction_verified'],
                      'error': report.get('error'), 'report': str(OUTPUT / f'{run_id}.json')},
                     ensure_ascii=False, indent=2), flush=True)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
