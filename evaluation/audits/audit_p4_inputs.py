"""Independently check retained IDs/text/labels against the hash-locked raw inputs."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import json
from pathlib import Path
import zipfile
import pyarrow.parquet as pq
from arkb.evaluation.external import digest,load_external,write_json


def parquet(path):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=2048):yield from batch.to_pylist()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,default=Path('evaluation/results/p4-inputs'))
    p.add_argument('--data',type=Path,default=Path('evaluation/results/p4-data'))
    p.add_argument('--lock',type=Path,default=Path('evaluation/experiments/p4-downloads.lock.json'))
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    lock=json.loads(a.lock.read_text())
    for name,expected in lock['files'].items():
        if digest(a.inputs/name)!=expected['sha256'] or (a.inputs/name).stat().st_size!=expected['bytes']:
            raise ValueError('Raw input differs from pinned lock: '+name)
    report={'download_lock_sha256':digest(a.lock),'locked_files':len(lock['files']),'datasets':{}}
    for name in ('scifact','bright-stackoverflow','bright-robotics'):
        data=load_external(a.data/name);docs={d['id']:d for d in data.corpus};ids=[]
        bright=name.startswith('bright-');domain=name.removeprefix('bright-')
        corpus=(f'bright-pro/documents/{domain}.parquet' if bright else
                'scifact/corpus/corpus-00000-of-00001.parquet')
        for row in parquet(a.inputs/corpus):
            source=str(row['id'] if bright else row['_id']);ids.append(source)
            expected={'id':source,'title':'' if bright else row.get('title') or '',
                      'text':row['content'] if bright else row['text']}
            if docs.get(source)!=expected:raise ValueError('Raw corpus identity/text changed: '+source)
        if len(ids)!=len(set(ids)) or set(ids)!=set(docs):raise ValueError('Corpus duplicate/drop.')
        query_path=(f'bright-pro/examples/{domain}.parquet' if bright else
                    'scifact/queries/queries-00000-of-00001.parquet')
        queries=list(parquet(a.inputs/query_path));query_ids=[str(r['id'] if bright else r['_id']) for r in queries]
        if len(query_ids)!=len(set(query_ids)):raise ValueError('Duplicate raw query IDs.')
        raw_queries={qid:r['query'] if bright else r['text'] for qid,r in zip(query_ids,queries)}
        if any(raw_queries[q]!=text for q,text in data.queries.items()):raise ValueError('Query text changed.')
        expected_rels={q:{} for q in data.queries}
        if bright:
            judgments=[{'query-id':str(r['id']),'corpus-id':d,'score':1} for r in queries for d in r['gold_ids']]
        else:
            with (a.inputs/'scifact-qrels/test.tsv').open() as f:judgments=list(csv.DictReader(f,delimiter='\t'))
        for r in judgments:
            qid=str(r['query-id']);source=str(r['corpus-id']);number=Decimal(str(r['score']))
            if not number.is_finite() or number<0 or number!=int(number):raise ValueError('Lossy relevance conversion.')
            if qid in expected_rels:
                if source in expected_rels[qid]:raise ValueError('Duplicate selected qrel.')
                expected_rels[qid][source]=int(number)
        if expected_rels!=data.qrels:raise ValueError('Selected relevance labels changed.')
        if bright:
            original_aspects=list(parquet(a.inputs/f'bright-pro/aspects/{domain}.parquet'))
            retained_aspects=[r for aspects in data.aspects.values() for r in aspects]
            if sorted(original_aspects,key=lambda r:r['id'])!=sorted(retained_aspects,key=lambda r:r['id']):
                raise ValueError('Aspect annotation fields changed.')
        report['datasets'][name]={'corpus_units_verified':len(ids),'raw_queries':len(queries),'selected_queries_verified':len(data.queries),
            'selected_qrels_verified':sum(map(len,data.qrels.values())),'duplicate_corpus_ids':0,'duplicate_raw_query_ids':0,
            'manifest_sha256':digest(a.data/name/'manifest.json')}
    with zipfile.ZipFile(a.inputs/'musique.zip') as archive:
        original=[json.loads(line) for line in archive.open('data/musique_full_v1.0_dev.jsonl')]
    keys=[(r['id'],r['answerable']) for r in original]
    if len(set(keys))!=len(keys) or set(Counter(r['id'] for r in original).values())!={2}:raise ValueError('Original pairs malformed.')
    by_key=dict(zip(keys,original));selected=json.loads((a.data/'musique-selected.json').read_text())
    for r in selected['rows']:
        if r!=by_key[(r['id'],r['answerable'])] or len(r['paragraphs'])!=20:raise ValueError('MuSiQue supplied context changed.')
    groups=Counter(r['id'] for r in selected['rows'])
    if set(groups.values())!={2} or set(groups)!=set(selected['metadata']['selected_group_ids']):raise ValueError('Selected pairs incomplete.')
    report['musique']={'original_variants':len(original),'selected_groups':len(groups),'selected_variants_verified':len(selected['rows']),
        'all_original_contexts_preserved':True,'selected_sha256':digest(a.data/'musique-selected.json')}
    report.update(passed=True,scope='Byte/content/identity preservation, not independent verification of annotation correctness or model training contamination.')
    write_json(a.output,report);print(json.dumps(report))


if __name__=='__main__':main()
