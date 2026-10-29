"""Capture installed dependency metadata and bundled legal notices."""
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from liushao.data import sha256, write_json


def main():
    rows = []
    root = Path('third_party/licenses')
    root.mkdir(parents=True, exist_ok=True)
    for dist in sorted(md.distributions(), key=lambda d: d.metadata['Name'].lower()):
        metadata = dist.metadata
        name = metadata['Name']
        row = {'name': name, 'version': dist.version,
               'license_expression': metadata.get('License-Expression'),
               'license_metadata': metadata.get('License'),
               'license_classifiers': [c for c in metadata.get_all('Classifier', []) if c.startswith('License ::')],
               'project_urls': metadata.get_all('Project-URL', []),
               'homepage': metadata.get('Home-page'), 'bundled_notices': []}
        for relative in dist.files or []:
            # Copy notices, not executable packages or code. Keep paths within each distribution.
            if relative.name.lower().startswith(('license', 'licence', 'copying', 'notice', 'copyright')):
                source = Path(dist.locate_file(relative))
                if source.is_file():
                    dest = root / name / str(relative).replace('..', '_').replace('\\', '/')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(source, dest)
                    row['bundled_notices'].append({'path': dest.as_posix(), 'sha256': sha256(dest)})
        rows.append(row)
    write_json('docs/results/dependencies.json', {'python': platform.python_version(), 'packages': rows})
    lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    Path('requirements-lock.txt').write_text(lock, encoding='utf-8')
    print(json.dumps({'packages': len(rows), 'notices': sum(len(r['bundled_notices']) for r in rows),
                      'missing_license_metadata_and_files': [r['name'] for r in rows if not any([r['license_expression'], r['license_metadata'], r['license_classifiers'], r['bundled_notices']])]}, indent=2))


if __name__ == '__main__':
    main()
