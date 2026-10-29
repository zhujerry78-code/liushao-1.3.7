"""Verify exact score accounting on the frozen demo, with reproducible identities."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from app_model import verified_model,load_model_bundle
from app_service import analyze_input
from app_decomposition import decompose_scores,TOLERANCE
from app_evidence import build_record_evidence
from liushao.reviews import _snapshot_json


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifacts',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.parent.mkdir(parents=True,exist_ok=True)
    art=args.artifacts;v=verified_model(art/'model.joblib',art/'model.json');b=load_model_bundle(v.content,v.descriptor)
    a,result,physical,quality,error=analyze_input((art/'demo.csv').read_bytes(),'demo.csv','内置演示数据',b,v.identity,art)
    assert error is None
    tick=time.perf_counter();rows=[]
    for index,record in result.predictions.iterrows():
        d=decompose_scores(result.features.loc[[index]],b)
        assert d['status']=='available',d
        assert abs(d['rf']['score']-record.rf_score)<=TOLERANCE
        assert abs(d['lr']['score']-record.lr_score)<=TOLERANCE
        rows.append({'record_id':record.record_id,'input_csv_row':int(physical.at[index]),'source_row':int(record.source_row),'errors':d['errors']})
    cases=[]
    for source in (217809,609344):
        record=result.predictions[result.predictions.source_row.eq(source)].iloc[0]
        payload=build_record_evidence(record,result.features,b,meta={'input_name':a.name,'input_csv_row':int(physical.at[record.name]),'input_sha256':a.digest,'model_sha256':v.model_sha256,'descriptor_sha256':v.descriptor_sha256})
        raw=_snapshot_json(payload,record.record_id,b['model_version'])
        assert len(payload['cards'])<=6 and len(payload['decomposition']['default_fields'])<=12
        cases.append({'source_row':source,'input_csv_row':payload['input_csv_row'],'canonical_bytes':len(raw.encode('utf-8')),'snapshot_sha256':hashlib.sha256(raw.encode('utf-8')).hexdigest(),'missing_fields':payload['selection']['missing_fields']})
    report={'passed':True,'scope':'frozen demo numerical accounting; not independent evaluation or human evidence','rows':rows,'count':len(rows),'cases':cases,'seconds':time.perf_counter()-tick,'tolerance':TOLERANCE,
        'maximum_errors':{key:max(row['errors'][key] for row in rows) for key in rows[0]['errors']},
        'input_sha256':a.digest,'model_sha256':v.model_sha256,'descriptor_sha256':v.descriptor_sha256,
        'source_hashes':{name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('app_decomposition.py','app_evidence.py','app_evidence_rules.py','liushao/model.py','liushao/data.py')}}
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','source_hashes')},ensure_ascii=False))

if __name__=='__main__':main()
