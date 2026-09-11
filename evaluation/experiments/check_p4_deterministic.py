"""Check CURRENT BM25 against the archived full-corpus SciFact development baseline.

No downloads or model services. Intentional ranking changes require reviewing
the retained deltas and explicitly registering a new baseline, not editing scores.
"""
import argparse
import inspect
import json
from pathlib import Path
import tarfile
import tempfile

from arkb.evaluation.external import digest, load_external, read_jsonl, score_ranking, write_json
from arkb.retrieval.bm25 import BM25Retriever


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('evaluation/experiments/artifacts/p4-scifact-20260910.tar.gz'))
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise ValueError('Retain prior regression reports; use a new output path.')
    sidecar=json.loads(args.archive.with_suffix('.manifest.json').read_text())
    if digest(args.archive)!=sidecar['sha256'] or args.archive.stat().st_size!=sidecar['bytes']:
        raise ValueError('Archived baseline hash differs.')
    with tempfile.TemporaryDirectory(prefix='arkb-p4-current-baseline-') as directory:
        root=Path(directory).resolve()
        with tarfile.open(args.archive,'r:gz') as archive:
            members=archive.getmembers()
            if len({m.name for m in members})!=len(members) or any(
                    not m.isfile() or not (root/m.name).resolve().is_relative_to(root) for m in members):
                raise ValueError('Unsafe or duplicate archive member.')
            archive.extractall(root,filter='data')
        manifest=json.loads((root/'archive-manifest.json').read_text())
        if manifest['schema']!='arkb-p4-score-archive-v1':raise ValueError('Unknown archive schema.')
        if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(manifest['files'])|{'archive-manifest.json'}:
            raise ValueError('Archive inventory differs.')
        for name,expected in manifest['files'].items():
            if digest(root/name)!=expected:raise ValueError('Archived member hash differs.')
        data=load_external(root/'dataset')
        if data.name!='scifact':raise ValueError('Expected the registered SciFact corpus.')
        baseline=read_jsonl(root/'runs/p4-scifact-lexical/rows.jsonl')
        if (len(baseline)!=len(data.queries) or {r['qid'] for r in baseline}!=set(data.queries)
                or any(r['error'] is not None or r['arm']!='bm25-native-unit' for r in baseline)):
            raise ValueError('Incomplete or failed archived baseline.')
        mapping=data.source_map();engine=BM25Retriever(data.records(),index_id=data.name);deltas=[];max_error=0.0
        for row in baseline:
            query_id=row['qid']
            ranked=[mapping[h.source] for h in engine.search(data.queries[query_id],top_k=100).results]
            metrics=score_ranking(data,query_id,ranked)
            differences={k:metrics[k]-row['metrics'][k] for k in metrics}
            max_error=max(max_error,*map(abs,differences.values()))
            if ranked!=row['ranking'] or any(abs(v)>1e-12 for v in differences.values()):
                deltas.append({'query_id':query_id,'before':row['ranking'],'after':ranked,
                    'metric_deltas':differences})
        write_json(args.output,{'status':'changed' if deltas else 'passed',
            'archive_sha256':sidecar['sha256'],'dataset':data.name,'corpus_units':len(data.corpus),
            'queries':len(baseline),'changed_queries':len(deltas),'deltas':deltas,
            'metric_absolute_tolerance':1e-12,'metric_max_absolute_error':max_error,
            'current_bm25_sha256':digest(Path(inspect.getfile(BM25Retriever))),
            'scope':'Current-code native-unit BM25 ranking/metric regression on public development data; no model calls or release-quality certification.',
            'release_eligible':False})
        if deltas:raise SystemExit('Current BM25 differs from frozen development baseline; inspect saved deltas.')
        print(f'{len(baseline)} current-code full-corpus rankings match the archived baseline.')


if __name__=='__main__':main()
