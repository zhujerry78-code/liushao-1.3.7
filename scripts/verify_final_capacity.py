"""Synthetic 50k-input/100k-unrelated-history capacity check, never human evidence."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import pandas as pd
import psutil
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app_model import verified_model,load_model_bundle
from app_service import analyze_input,read_review_snapshot
from app_evidence import build_record_evidence
from liushao.data import read_csv_checked
from liushao import reviews


def main():
 p=argparse.ArgumentParser();p.add_argument('--artifacts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 v=verified_model(a.artifacts/'model.joblib',a.artifacts/'model.json');b=load_model_bundle(v.content,v.descriptor)
 with (a.artifacts/'demo.csv').open('rb') as f:raw=read_csv_checked(f)
 raw=pd.concat([raw]*50,ignore_index=True);assert len(raw)==50000
 content=raw.to_csv(index=False).encode('utf-8');assert len(content)<=50*1024*1024
 t=time.perf_counter();active,result,physical,quality,error=analyze_input(content,'synthetic-50000.csv','上传 CSV',b,v.identity,a.artifacts);analysis=time.perf_counter()-t;assert error is None
 row=result.predictions.iloc[0]
 meta=dict(input_name=active.name,input_sha256=active.digest,model_sha256=v.model_sha256,descriptor_sha256=v.descriptor_sha256,input_csv_row=int(physical.at[row.name]))
 t=time.perf_counter();snapshot=build_record_evidence(row,result.features,b,context=raw,meta=meta);first=time.perf_counter()-t
 db=a.output/'synthetic-history.sqlite';conn=reviews.connect(db)
 t=time.perf_counter()
 conn.executemany('INSERT INTO reviews(record_id,model_version,status,note,created_at) VALUES(?,?,?,?,?)',(('unrelated-engineering-'+str(i),b['model_version'],'待补证','合成工程容量记录','2026-10-23T00:00:00+00:00') for i in range(100000)));conn.commit();conn.close()
 for i in range(10):reviews.save_review(db,row.record_id,b['model_version'],'待补证','合成工程容量记录 '+str(i),evidence_snapshot=snapshot)
 create=time.perf_counter()-t
 t=time.perf_counter();view=read_review_snapshot(db,b['model_version'],set(result.predictions.record_id));scoped=time.perf_counter()-t
 assert len(view.history)==10 and all('evidence_snapshot' not in r for r in view.history)
 assert all(r['evidence_snapshot_status']=='deferred' for r in view.history)
 t=time.perf_counter();full=reviews.history(db,model_version=b['model_version'],record_ids=set(result.predictions.record_id));json_bytes=json.dumps(full,ensure_ascii=False,allow_nan=False).encode('utf-8');export=time.perf_counter()-t
 assert len(full)==10 and all(r['evidence_snapshot']==snapshot for r in full)
 t=time.perf_counter();again=reviews.evidence_for_review(db,full[0]['id']);lazy=time.perf_counter()-t;assert again['evidence_snapshot']==snapshot
 cache={(active.key,row.record_id):snapshot};t=time.perf_counter();assert cache[(active.key,row.record_id)] is snapshot;cache_time=time.perf_counter()-t
 memory=psutil.Process().memory_info()
 report={'passed':True,'scope':'single-machine synthetic engineering capacity; not cross-machine or human performance','input_rows':len(raw),'input_bytes':len(content),'input_sha256':active.digest,'unrelated_history':100000,'related_history':10,'snapshot_bytes':len(reviews._snapshot_json(snapshot,row.record_id,b['model_version']).encode()),'history_export_bytes':len(json_bytes),'database_bytes':db.stat().st_size,'seconds':{'analysis':analysis,'first_evidence_uncached':first,'cache_lookup':cache_time,'create_history':create,'scoped_lightweight_read':scoped,'full_scoped_history_json_export':export,'lazy_one_snapshot':lazy},'rss_bytes':memory.rss,'process_peak_working_set_bytes':getattr(memory,'peak_wset',None),'model_sha256':v.model_sha256,'descriptor_sha256':v.descriptor_sha256}
 (a.output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
