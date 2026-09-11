"""Run the remaining authorized P4 experiments sequentially; no scheduler setup."""
import json
from pathlib import Path
import subprocess
import sys
import time
from arkb.evaluation.external import verify_checksums, write_json

root=Path('evaluation/results')
state=root/'p4-queue.json'
jobs=[]
for name in ('bright-stackoverflow','bright-robotics'):
    jobs.append([sys.executable,'-u','evaluation/experiments/run_p4.py','--dataset',str(root/'p4-data'/name),
                 '--output',str(root/f'p4-{name}-indexed'),'--track','indexed'])
    jobs.append([sys.executable,'-u','evaluation/experiments/run_p4_agents.py','--dataset',str(root/'p4-data'/name),
                 '--index-run',str(root/f'p4-{name}-indexed'),'--output',str(root/f'p4-{name}-agent-v2'),'--track','bright'])
jobs.append([sys.executable,'-u','evaluation/experiments/run_p4_agents.py','--track','musique',
             '--output',str(root/'p4-musique-agent-v3')])
progress={'status':'waiting_for_scifact','jobs':jobs,'completed':[]}
write_json(state,progress)
while True:
    m=json.loads((root/'p4-scifact-indexed/experiment.json').read_text())
    if m['status']=='completed': break
    if m['status']=='failed': raise SystemExit('SciFact failed; queue halted.')
    time.sleep(10)
try:
    for command in jobs:
        output=Path(command[command.index('--output')+1])
        if output.exists():
            metadata=output/'experiment.json'
            if not metadata.exists() or json.loads(metadata.read_text())['status']!='completed':
                raise ValueError('Existing incomplete attempt must be retained and explicitly replaced: '+str(output))
            verify_checksums(output)
            progress['completed'].append(command)
            continue
        progress.update(status='running',active=command)
        write_json(state,progress)
        result=subprocess.run(command)
        if result.returncode: raise RuntimeError('Experiment failed: '+str(command))
        progress['completed'].append(command)
    progress.update(status='completed',active=None)
except BaseException as e:
    progress.update(status='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed',error=str(e));raise
finally:
    write_json(state,progress)
