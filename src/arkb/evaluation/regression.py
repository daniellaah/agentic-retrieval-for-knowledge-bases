"""Deterministic freshness/scale probes and fail-closed holdout rotation ledger."""
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
from time import perf_counter

from arkb.evaluation.external import digest, write_json, sample_ids
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval.bm25 import BM25Retriever
from arkb.retrieval.exact import ExactRetriever


def freshness_probe(directory):
    """Exercise the captured-search/live-read contract, without model variability."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    for name,text in {'policy.md':'# Policy\nquota 17\n## Exceptions\nbackup 31',
                      'deleted.md':'# Deleted\nretired 43','renamed.md':'# Rename\nbridge 59'}.items():
        (directory/name).write_text(text)
    access=DocumentAccess(directory,vault_id='freshness-fixture')
    records=list(access.records()); snapshot=BM25Retriever(records,index_id='t0')
    old={r.chunk.source:r for r in records}
    (directory/'policy.md').write_text('# Policy\nquota 23\n## Exceptions\nbackup 37')
    (directory/'deleted.md').unlink();(directory/'renamed.md').rename(directory/'moved.md')
    stale=snapshot.search('quota',top_k=10).results[0]
    live=access.read(old['policy.md'].document_id)
    checks={'snapshot_keeps_t0_content':'17' in stale.content,
            'live_read_observes_t1_content':'23' in live.content,
            'revisions_distinguish_stale_evidence':stale.metadata['document_revision']!=live.document_revision,
            'literal_match_observes_t1':bool(ExactRetriever(access).search('23').results),
            'literal_match_excludes_t0':not ExactRetriever(access).search('17').results,
            'snapshot_keeps_deleted_until_rebuild':bool(snapshot.search('retired').results)}
    for name in ('deleted.md','renamed.md'):
        try: access.read(old[name].document_id); checks[name+'_old_id_rejected']=False
        except LookupError: checks[name+'_old_id_rejected']=True
    moved=access.read(source='moved.md')
    checks['rename_has_new_identity']=moved.document_id!=old['renamed.md'].document_id
    rebuilt=BM25Retriever(list(access.records()),index_id='t1')
    checks['rebuilt_removes_deleted']=not rebuilt.search('retired').results
    checks['rebuilt_returns_new_content']='23' in rebuilt.search('quota').results[0].content
    checks['rebuilt_uses_new_filename']=rebuilt.search('bridge').results[0].source=='moved.md'
    if not all(checks.values()): raise AssertionError(checks)
    return {'kind':'deterministic captured BM25/live filesystem contract','checks':checks,
            'passed':len(checks),'failed':0,'scope':'No claim about asynchronous freshness or vector publication latency.'}


def scale_probe(dataset, *, sizes=(1000,10000), query_count=25):
    """Nested real-document subsets for operational cost only; no relevance scores."""
    import numpy as np
    records={r.chunk.source:r for r in dataset.records()}
    order=sample_ids(list(records),max(sizes)); queries=sample_ids(list(dataset.queries),min(query_count,len(dataset.queries)))
    rows=[]
    for size in sizes:
        start=perf_counter(); retriever=BM25Retriever([records[s] for s in order[:size]],index_id=f'scale-{size}')
        setup=(perf_counter()-start)*1000; elapsed=[]; returned=[]
        for q in queries:
            start=perf_counter(); response=retriever.search(dataset.queries[q],top_k=100)
            elapsed.append((perf_counter()-start)*1000);returned.append(len(response.results))
        rows.append({'documents':size,'index_build_ms':setup,'query_count':len(queries),
                     'query_p50_ms':float(np.median(elapsed)),'query_p95_ms':float(np.quantile(elapsed,.95)),
                     'query_ms':elapsed,'returned_counts':returned,
                     'sources_sha256':hashlib.sha256(json.dumps(order[:size]).encode()).hexdigest()})
    return {'dataset':dataset.name,'purpose':'operational scale only, intentionally no quality metric',
            'seed':20260910,'query_ids':queries,'nested_sources':order,'rows':rows,
            'limits':'BM25 in-memory native-unit scale; not embedding throughput or concurrent service load.'}


def reserve_rotation(ledger, *, release_id, dataset_id, groups, queries, reviewed, exposure):
    """Consume an unseen holdout before evaluation, including failed attempts.

    Review/exposure are curator attestations, not authenticated by this function.
    A public-development set is never accepted as a release holdout. Preserve
    this ledger outside the tuning checkout in a real release workflow.
    """
    if reviewed is not True or exposure!='private-unseen':
        raise ValueError('Only independently reviewed, private unseen data can be reserved.')
    if not all(isinstance(x,str) and x.strip() for x in (release_id,dataset_id)):
        raise ValueError('Missing release/dataset identity.')
    groups=list(groups); queries=list(queries)
    if not groups or len(groups)!=len(set(groups)) or not queries or len(queries)!=len(set(queries)):
        raise ValueError('Holdout must have distinct nonempty groups and queries.')
    if any(not isinstance(x,str) or not x.strip() for x in groups+queries):
        raise ValueError('Invalid holdout identities.')
    qhashes=[hashlib.sha256(' '.join(q.casefold().split()).encode()).hexdigest() for q in queries]
    if len(qhashes)!=len(set(qhashes)): raise ValueError('Normalized query duplicates.')
    ledger=Path(ledger); ledger.parent.mkdir(parents=True,exist_ok=True)
    with ledger.open('a+') as f:
        fcntl.flock(f,fcntl.LOCK_EX);f.seek(0)
        rows=[json.loads(line) for line in f if line.strip()]
        previous='0'*64
        for row in rows:
            payload={k:v for k,v in row.items() if k!='sha256'}
            if row['previous']!=previous or _event_hash(payload)!=row['sha256']:
                raise ValueError('Rotation ledger hash chain changed.')
            previous=row['sha256']
        if any(r['release_id']==release_id or r['dataset_id']==dataset_id
               or set(groups).intersection(r['groups']) or set(qhashes).intersection(r['query_hashes']) for r in rows):
            raise ValueError('Holdout already exposed/reserved; use independently new groups.')
        event={'release_id':release_id,'dataset_id':dataset_id,'groups':groups,'query_hashes':qhashes,
               'status':'reserved_and_exposed','reserved_at':datetime.now(timezone.utc).isoformat(),'previous':previous}
        event['sha256']=_event_hash(event)
        f.write(json.dumps(event,sort_keys=True)+'\n');f.flush()
        import os
        os.fsync(f.fileno())
    return event


def _event_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def reserve_core_release(directory, ledger, release_id):
    """Validate the existing reviewed core freeze before reserving its test set."""
    from arkb.evaluation.v2 import load_dataset
    from arkb.evaluation.core import fingerprint
    directory=Path(directory)
    data=load_dataset(directory,notes_dir=directory/'corpus')
    freeze=json.loads((directory/'core-freeze.json').read_text())
    if freeze['frozen_dataset_fingerprint']!=fingerprint(data.manifest) or not freeze['readiness']['freeze_ready']:
        raise ValueError('Unvalidated core freeze.')
    test=[c for c in data.cases if c['split']=='test']
    registry={r['family_id']:r for r in freeze['registry']['families']}
    families={c['intent_family_id'] for c in test}
    known=freeze['policy'].get('known_development_queries',{})
    if (families.intersection(freeze['policy']['known_development_families'])
            or any(registry[f]['seen_in_development'] for f in families)
            or any(fingerprint(' '.join(c['query'].casefold().split())) in known for c in test)):
        raise ValueError('Development exposure in test.')
    return reserve_rotation(ledger,release_id=release_id,dataset_id=freeze['frozen_dataset_fingerprint'],
        groups=sorted({registry[f]['leakage_group'] for f in families}),queries=[c['query'] for c in test],
        reviewed=not data.provisional,exposure='private-unseen')
