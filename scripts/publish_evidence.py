"""Create explicit public derivatives; preserve verified private originals unchanged."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

from delivery.safe_files import read_json
from app_evaluation import validate_summary

ABSOLUTE = re.compile(r'^[A-Za-z]:[\\/]')


def digest(content):
    return hashlib.sha256(content).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def annotate_summary(summary, changes, original_sha256):
    for source in summary['sources']:
        publication = changes.get(source['path'])
        if publication:
            source['original_sha256'] = publication['original_sha256']
            source['publication_kind'] = publication['kind']
    summary['publication'] = {
        'kind':'public-evidence-summary', 'original_sha256':original_sha256,
        'changed_sources':sorted(changes), 'historical_evaluation_rerun':False,
        'meaning':'来源SHA对应明确标注的发布副本；历史指标、模型和数据身份不变。'}
    return summary


def redact_paths(value):
    pointers = []
    def walk(item, pointer):
        if isinstance(item, dict):
            return {key: walk(child, pointer+'/'+key.replace('~','~0').replace('/','~1')) for key, child in item.items()}
        if isinstance(item, list):
            return [walk(child, pointer+'/'+str(index)) for index, child in enumerate(item)]
        if isinstance(item, str) and (ABSOLUTE.match(item) or item.startswith('\\\\')):
            pointers.append(pointer)
            basename = item.replace('\\','/').rstrip('/').rsplit('/',1)[-1]
            return '<历史本机目录>/' + basename if Path(basename).suffix else '<历史工作目录>'
        return item
    return walk(deepcopy(value), ''), pointers


def public_sources(raw, summary_bytes):
    summary = read_json(summary_bytes)
    if summary.get('publication'):
        raise ValueError('输入已是公开派生摘要，请使用独立保存的原始证据。')
    for source in summary['sources']:
        if digest(raw[source['path']]) != source['sha256']:
            raise ValueError('原始来源SHA不匹配：'+source['path'])
    output, changes = dict(raw), {}
    for name, content in raw.items():
        original = read_json(content)
        redacted, pointers = redact_paths(original)
        if pointers:
            redacted['publication'] = {
                'kind':'path-redacted-public-derivative', 'original_sha256':digest(content),
                'changed_json_pointers':pointers, 'historical_evaluation_rerun':False,
                'meaning':'仅替换本机绝对路径；历史指标及执行时间不改写，此文件不是原始字节。'}
            output[name] = encoded(redacted)
            changes[name] = redacted['publication']
    audit_name = 'docs/phase2/audit.json'
    study_name = 'docs/innovation/study.json'
    if output[audit_name] != raw[audit_name]:
        study = read_json(output[study_name])
        key = audit_name.replace('/','\\')
        prior = study['protected_hashes'][key]
        if prior != digest(raw[audit_name]):
            raise ValueError('研究记录与原始审计SHA不匹配')
        study['protected_hashes'][key] = digest(output[audit_name])
        prior_publication = study.get('publication', {})
        study['publication'] = {
            'kind':'public-evidence-reference-update', 'original_sha256':digest(raw[study_name]),
            **({'changed_json_pointers':prior_publication['changed_json_pointers']} if prior_publication else {}),
            'original_protected_hashes':{key:prior}, 'historical_evaluation_rerun':False,
            'meaning':'protected_hashes中的审计项指向公开脱敏副本；其余模型、数据与指标保护值未变。'}
        output[study_name] = encoded(study)
        changes[study_name] = study['publication']
    for source in summary['sources']:
        name = source['path']
        new = digest(output[name])
        if new != source['sha256']:
            source['sha256'] = new
    annotate_summary(summary, changes, digest(summary_bytes))
    validate_summary(summary)
    return output, encoded(summary), changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-root',required=True,type=Path)
    parser.add_argument('--summary',required=True,type=Path)
    parser.add_argument('--summary-sha256',required=True)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    content = args.summary.read_bytes()
    if digest(content) != args.summary_sha256:
        raise ValueError('原始摘要与独立保存的SHA不匹配')
    summary = read_json(content)
    raw = {item['path']:(args.original_root/item['path']).read_bytes() for item in summary['sources']}
    public, summary_public, changes = public_sources(raw, content)
    args.output.mkdir(parents=True,exist_ok=False)
    for name, value in {**public, 'resources/evaluation/summary.json':summary_public}.items():
        target = args.output/name
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(value)
    print(json.dumps({'summary_sha256':digest(summary_public),'changed_sources':list(changes)},ensure_ascii=False))


if __name__=='__main__':
    main()
