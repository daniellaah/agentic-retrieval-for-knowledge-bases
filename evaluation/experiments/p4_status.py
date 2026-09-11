"""Read this P4 batch's progress and append a small diagnostic checkpoint."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sqlite3


def main():
    root=Path('evaluation/results').resolve()
    queue=json.loads((root/'p4-queue.json').read_text())
    result={'observed_at':datetime.now(timezone.utc).isoformat(),'queue':queue['status'],
        'finalizer':json.loads((root/'p4-finalization.json').read_text())['status']}
    command=queue.get('active')
    if command:
        out=Path(command[command.index('--output')+1]).resolve()
        if not out.is_relative_to(root):raise ValueError('Unexpected experiment path.')
        result['run']=out.name
        metadata=out/'experiment.json'
        if metadata.exists():result['experiment']=json.loads(metadata.read_text())['status']
        rows=out/'rows.jsonl'
        if rows.exists():
            complete=rows.read_text().splitlines(keepends=True)
            complete=[line for line in complete if line.endswith('\n')]
            result['completed_rows']=len(complete)
            if 'agent' in out.name:
                result['stops']=dict(Counter(json.loads(line)['stop_reason'] for line in complete))
        if 'indexed' in out.name and (out/'index.sqlite').exists():
            try:
                with sqlite3.connect('file:'+str(out/'index.sqlite')+'?mode=ro',uri=True,timeout=1) as db:
                    result['cached_embedding_inputs']=db.execute('select count(*) from embeddings').fetchone()[0]
                    row=db.execute('select manifest from builds order by rowid desc limit 1').fetchone()
                    manifest=json.loads(row[0]) if row else None
                    result['index_status']=manifest['status'] if manifest else 'preflight'
                    if manifest:result['chunks']=manifest['chunk_count']
            except sqlite3.OperationalError as error:result['index_read_status']=str(error)
    # Once model work stops, the finalizer may seal the regression artifacts.
    # Subsequent status reads must not mutate a sealed diagnostic file.
    if queue['status']=='running':
        with (root/'p4-regression/progress-checkpoints.jsonl').open('a') as stream:
            stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
