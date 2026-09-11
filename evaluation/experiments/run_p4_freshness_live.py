"""Verify actual SQLite/Qdrant publication versus live Markdown reads."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from arkb.evaluation.external import digest, write_json


def execute(output, url):
    from arkb.config import RuntimeConfig
    from arkb.runtime import Runtime
    from arkb.knowledge.sqlite import SQLiteStorage
    from arkb.knowledge.documents import DocumentAccess
    from arkb.knowledge.qdrant import QdrantIndex
    notes=output/'corpus';notes.mkdir()
    original={'policy.md':'# Policy\nquota 17', 'deleted.md':'# Deleted\nretired 43', 'renamed.md':'# Rename\nbridge 59'}
    for name,body in original.items():(notes/name).write_text(body)
    write_json(output/'t0-files.json',original)
    checks={};artifacts={}
    with Runtime(RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers').resolve(),qdrant_url=url)) as runtime:
        build=runtime.index(db=output/'index.sqlite',vault_id='p4-live-freshness',notes_dir=notes,chunking='none')
        artifacts['t0']=asdict(build)
        with SQLiteStorage(output/'index.sqlite',read_only=True) as old_storage:
            engine=runtime.retrieval_engine(old_storage,build.manifest,modes=('bm25','semantic'),exact=True)
            old_hits=engine.search('quota bridge retired',mode='semantic',top_k=3).results
            old={h.source:h for h in old_hits}
            (notes/'policy.md').write_text('# Policy\nquota 23')
            (notes/'deleted.md').unlink();(notes/'renamed.md').rename(notes/'moved.md')
            access=DocumentAccess(notes,vault_id='p4-live-freshness')
            checks['live_read_before_rebuild']='23' in access.read(old['policy.md'].source_id).content
            checks['revision_changed_before_rebuild']=access.read(old['policy.md'].source_id).document_revision!=old['policy.md'].metadata['document_revision']
            checks['live_match_before_rebuild']=bool(runtime.match('23',db=output/'index.sqlite',vault_id='p4-live-freshness').results)
            stale=engine.search('quota bridge retired',mode='semantic',top_k=3).results
            checks['old_semantic_snapshot_stays_frozen']={h.source:h.content for h in stale}=={h.source:h.content for h in old_hits}
            for name in ('deleted.md','renamed.md'):
                try:access.read(old[name].source_id);checks[name+'_old_read_rejected']=False
                except LookupError:checks[name+'_old_read_rejected']=True
            after=runtime.index(db=output/'index.sqlite',vault_id='p4-live-freshness',notes_dir=notes,chunking='none')
            artifacts['t1']=asdict(after)
            checks['new_publication']=after.manifest.index_version!=build.manifest.index_version
            current=runtime.search('quota bridge retired',db=output/'index.sqlite',vault_id='p4-live-freshness',mode='semantic',top_k=3,exact=True)
            checks['new_semantic_sources']={h.source for h in current.results}=={'policy.md','moved.md'}
            checks['new_semantic_content']=any(h.source=='policy.md' and '23' in h.content for h in current.results)
            checks['new_bm25_removes_deleted']=not runtime.search('retired',db=output/'index.sqlite',vault_id='p4-live-freshness',mode='bm25').results
            checks['old_captured_engine_survives_publication']={h.source for h in engine.search('retired',mode='semantic',top_k=3).results}==set(old)
        with SQLiteStorage(output/'index.sqlite',read_only=True) as storage:
            for manifest in (build.manifest,after.manifest):
                _,records,vectors=storage.load_snapshot(manifest.index_version)
                backend=storage.build_metadata(manifest.index_version)['backend']
                remote=QdrantIndex(runtime.qdrant_client(url),backend['collection'],manifest.embedding_spec,vault_id=manifest.vault_id)
                remote.verify_snapshot(records,vectors)
        checks['both_remote_snapshots_match_sqlite']=True
    write_json(output/'results.json',{'checks':checks,'passed':sum(checks.values()),'failed':sum(not v for v in checks.values()),
        'builds':artifacts,'scope':'controlled fixture with real local model, SQLite and Qdrant; not an asynchronous latency SLA'})
    if not all(checks.values()):raise AssertionError(checks)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qdrant-url',default='http://127.0.0.1:32776');p.add_argument('--execute',action='store_true')
    a=p.parse_args();a.output=a.output.resolve()
    if a.execute:execute(a.output,a.qdrant_url);return
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[2]
    shutil.copytree(root/'src',a.output/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copyfile(__file__,a.output/'run_p4_freshness_live.py')
    env={**os.environ,'PYTHONPATH':str(a.output/'measured-source/src'),'PYTHONDONTWRITEBYTECODE':'1','TOKENIZERS_PARALLELISM':'false'}
    with (a.output/'execution.log').open('x') as log:
        result=subprocess.run([sys.executable,str(a.output/'run_p4_freshness_live.py'),'--execute','--output',str(a.output),
                               '--qdrant-url',a.qdrant_url],env=env,stdout=log,stderr=subprocess.STDOUT)
    write_json(a.output/'checksums.json',{p.relative_to(a.output).as_posix():digest(p) for p in a.output.rglob('*')
        if p.is_file() and p.name not in ('checksums.json',) and not p.name.endswith(('-wal','-shm','.pyc'))})
    print('Live freshness exit:',result.returncode);raise SystemExit(result.returncode)


if __name__=='__main__':main()
