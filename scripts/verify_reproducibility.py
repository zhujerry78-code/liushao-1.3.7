"""Compare independently prepared data and independently trained predictions."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from liushao.data import sha256, write_json


def main():
    from scripts.evaluation_preflight import unavailable
    needed = ['data/' + d + '/' + n for d in ('processed', 'reproduced')
              for n in ('summary.json', 'features.parquet', 'manifest.parquet', 'rejected.csv')]
    needed += ['artifacts/' + d + n for d in ('', 'reproduced/') for n in ('metrics.json', 'test_predictions.csv')]
    if unavailable(Path('.'), needed):
        return 2
    base, repeated = Path('data/processed'), Path('data/reproduced')
    a = json.loads((base / 'summary.json').read_text(encoding='utf-8'))
    b = json.loads((repeated / 'summary.json').read_text(encoding='utf-8'))
    checks = {name: sha256(base / name) == sha256(repeated / name)
              for name in ['features.parquet', 'manifest.parquet', 'rejected.csv']}
    checks['data_version'] = a['data_version'] == b['data_version']
    first, second = Path('artifacts'), Path('artifacts/reproduced')
    metrics1 = json.loads((first / 'metrics.json').read_text(encoding='utf-8'))
    metrics2 = json.loads((second / 'metrics.json').read_text(encoding='utf-8'))
    model_checks = {key: metrics1[key] == metrics2[key] for key in
                    ['model_version', 'data_version', 'rf', 'lr', 'ranking_original', 'ranking_score', 'review']}
    p1, p2 = pd.read_csv(first / 'test_predictions.csv'), pd.read_csv(second / 'test_predictions.csv')
    model_checks['identical_record_order'] = p1.record_id.equals(p2.record_id)
    model_checks['predicted_classes'] = p1.rf_anomaly.equals(p2.rf_anomaly) and p1.lr_anomaly.equals(p2.lr_anomaly)
    differences = {key: float(np.max(np.abs(p1[key] - p2[key]))) for key in ['rf_score', 'lr_score']}
    model_checks['scores_within_1e-12'] = all(value <= 1e-12 for value in differences.values())
    result = {'checks': checks, 'model_checks': model_checks,
              'score_max_absolute_differences': differences, 'data_version': a['data_version'],
              'method': 'independent full prepare_data and train runs; timings and serialized model bytes may differ'}
    write_json('docs/results/reproducibility.json', result)
    assert all(checks.values()) and all(model_checks.values()), result
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    raise SystemExit(main())
