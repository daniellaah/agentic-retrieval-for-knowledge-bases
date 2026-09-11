"""Finish audits/archives as this P4 batch completes; no model calls or scheduler."""
import json
from pathlib import Path
import subprocess
import sys
import time
from arkb.evaluation.external import digest,write_json

root=Path('evaluation/results');audits=Path('evaluation/audits');experiments=Path('evaluation/experiments')
state=root/'p4-finalization.json';done={};waiting={}
if state.exists():done=json.loads(state.read_text()).get('completed',{})


def execute(key,command,*,capture=None):
    if key in done:return
    print(key,flush=True)
    result=subprocess.run([sys.executable,*map(str,command)],text=True,capture_output=True)
    if result.returncode:raise RuntimeError(f'{key}: {result.stderr[-5000:]} {result.stdout[-2000:]}')
    if capture:write_json(capture,json.loads(result.stdout))
    done[key]={'command':list(map(str,command)),'stdout':result.stdout[-2000:]}
    write_json(state,{'status':'running','completed':done,'waiting':waiting})


def completed(path):
    metadata=path/'experiment.json'
    if not metadata.exists():return False
    status=json.loads(metadata.read_text())['status']
    if status=='failed':raise ValueError('Model experiment failed: '+str(path))
    checksums=path/'checksums.json'
    if status!='completed' or not checksums.exists():return False
    if json.loads(checksums.read_text()).get('experiment.json')!=digest(metadata):
        raise ValueError('Terminal metadata is not bound to the sealed checksums: '+str(path))
    return True


try:
    while True:
        waiting={}
        for name in ('bright-stackoverflow','bright-robotics'):
            data=root/'p4-data'/name;run=root/f'p4-{name}-indexed'
            if name+'-retrieval' not in done:
                if completed(run):
                    execute(name+'-retrieval',[audits/'replay_p4.py',run,'--dataset',data],
                            capture=audits/f'20260910-p4-{name}-replay.json')
                else:waiting[name+'-retrieval']='waiting for complete matrix';continue
            execute(name+'-truncation',[audits/'audit_p4_rerank.py',run,'--dataset',data,
                    '--output',root/'p4-regression'/f'{name}-rerank-truncation'])
            agent=root/f'p4-{name}-agent-v2'
            if name+'-agent' not in done:
                if completed(agent):
                    execute(name+'-agent',[audits/'replay_p4_agents.py',agent,'--dataset',data],
                            capture=audits/f'20260910-p4-{name}-agent-replay.json')
                else:waiting[name+'-agent']='waiting for all fixed sample attempts';continue
            execute(name+'-review',[experiments/'summarize_p4_agents.py',agent,'--dataset',data,
                    '--output',Path('evaluation/reviews')/f'p4-{name}-20260910'])
            archive=experiments/'artifacts'/f'p4-{name}-20260910.tar.gz'
            if name+'-archive' not in done:
                command=[experiments/'archive_p4.py','--dataset',data,'--run',root/f'p4-{name}-lexical',
                         '--run',run,'--run',agent,'--output',archive]
                execute(name+'-archive',command)
            execute(name+'-archive-replay',[audits/'verify_p4_archive.py',archive,'--output',
                    audits/f'20260910-p4-{name}-archive-validation.json'])
        mu=root/'p4-musique-agent-v3'
        if 'musique-archive-replay' not in done:
            if completed(mu):
                execute('musique',[audits/'replay_p4_agents.py',mu],capture=audits/'20260910-p4-musique-replay.json')
                execute('musique-official',[audits/'check_p4_musique.py','--gold',mu/'gold.jsonl','--predictions',mu/'predictions.jsonl',
                        '--output',audits/'20260910-p4-musique-official.json'])
                execute('musique-review',[experiments/'summarize_p4_agents.py',mu,'--output',Path('evaluation/reviews/p4-musique-20260910')])
                execute('musique-tool-errors',[audits/'audit_p4_tool_errors.py',mu,
                    '--output',audits/'20260910-p4-musique-tool-errors.json'])
                archive=experiments/'artifacts/p4-musique-20260910.tar.gz'
                execute('musique-archive',[experiments/'archive_p4.py','--run',mu,'--output',archive])
                execute('musique-archive-replay',[audits/'verify_p4_archive.py',archive,'--output',audits/'20260910-p4-musique-archive-validation.json'])
            else:waiting['musique']='waiting for complete pairs'
        write_json(state,{'status':'running','completed':done,'waiting':waiting})
        if not waiting:break
        queue=json.loads((root/'p4-queue.json').read_text())
        if queue['status'] in ('failed','interrupted'):raise ValueError('Model queue stopped: '+queue['status'])
        time.sleep(30)
    execute('official-aspects',[audits/'verify_p4.py'])
    execute('agent-costs',[experiments/'summarize_p4_costs.py',
        '--run',root/'p4-bright-stackoverflow-agent-v2','--run',root/'p4-bright-robotics-agent-v2',
        '--run',root/'p4-musique-agent-v3','--output',audits/'20260910-p4-agent-costs.json'])
    source_command=[audits/'audit_p4_sources.py']
    for name in ('scifact','bright-stackoverflow','bright-robotics'):
        for track in ('lexical','indexed'):
            source_command += ['--run',root/f'p4-{name}-{track}']
    for name in ('bright-stackoverflow','bright-robotics','musique'):
        source_command += ['--run',root/f'p4-{name}-agent-{ "v3" if name=="musique" else "v2"}']
    source_command += ['--run',root/'p4-bright-stackoverflow-agent',
                       '--run',root/'p4-musique-agent-v2',
                       '--output',audits/'20260910-p4-production-source-consistency-v2.json']
    execute('production-source-consistency',source_command)
    write_json(root/'p4-regression/checksums.json',{p.relative_to(root/'p4-regression').as_posix():digest(p)
        for p in (root/'p4-regression').rglob('*') if p.is_file() and p.name!='checksums.json'})
    execute('regression-evidence-archive',[experiments/'archive_p4_regressions.py',
        '--output',experiments/'artifacts/p4-regression-evidence-20260910-v2.tar.gz',
        '--audit-output',audits/'20260910-p4-regression-archive-validation-v2.json'])
    summary={'status':'model_experiments_completed','release_eligible':False,'human_output_reviews':0,'datasets':{}}
    for name in ('scifact','bright-stackoverflow','bright-robotics'):
        summary['datasets'][name]={track:json.loads((root/f'p4-{name}-{track}'/'summary.json').read_text()) for track in ('lexical','indexed')}
    summary['agents']={name:json.loads((root/f'p4-{name}-agent-{ "v3" if name=="musique" else "v2"}'/'summary.json').read_text()) for name in ('bright-stackoverflow','bright-robotics','musique')}
    write_json(experiments/'p4-external-20260910-summary.json',summary)
    write_json(state,{'status':'completed','completed':done,'release_eligible':False})
    print('All P4 model batches audited, archived and replayed. Human quality/release holdout remain pending.',flush=True)
except BaseException as e:
    write_json(state,{'status':'failed','completed':done,'waiting':waiting,'error':{'type':type(e).__name__,'message':str(e)}})
    raise
