"""Offline contract fixture only; scripted provider results are not benchmark data."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import sys

from tokenizers import Tokenizer,models,pre_tokenizers
from tests.agent.helpers import ScriptedModel,reply,tool_call
from arkb.agent import run_agent,AgentBudget,AgentObserver,AgentTools
from arkb.knowledge.documents import DocumentAccess
from arkb.knowledge.embeddings import count_tokens,tokenizer_fingerprint
from arkb.retrieval.bm25 import BM25Retriever
from arkb.retrieval.engine import RetrievalEngine
from arkb.retrieval.exact import ExactRetriever
from arkb.evaluation.external import ExternalDataset,digest,write_json


def test_offline_agent_replay_catches_incorrect_evidence_totals(tmp_path):
    repo=Path(__file__).resolve().parents[2];run=tmp_path/'run';run.mkdir()
    data=ExternalDataset('bright-fixture',[{'id':'a','title':'','text':'alpha'},
        {'id':'b','title':'','text':' '.join(['beta']*5000)}],{'1':'alpha','2':'beta'},
        {'1':{'a':1},'2':{'b':1}},{},{})
    data.save(tmp_path/'data',materialize=True)
    tokenizer=Tokenizer(models.WordLevel({'[UNK]':0,'alpha':1,'beta':2},unk_token='[UNK]'))
    tokenizer.pre_tokenizer=pre_tokenizers.Whitespace()
    (run/'reference-tokenizer.json').write_text(tokenizer.to_str())
    retriever=BM25Retriever(data.records(),index_id='fixture');documents=DocumentAccess(tmp_path/'data/corpus',vault_id=data.name)
    tools=AgentTools(documents=documents,exact=ExactRetriever(documents),engine=RetrievalEngine(bm25=retriever,semantic=retriever))
    source=next(s for s,d in data.source_map().items() if d=='a')
    scripted=[ScriptedModel(reply(calls=[tool_call('search',query='alpha',mode='bm25')]),
                reply(calls=[tool_call('read',source=source)]),reply('alpha')),
              ScriptedModel(reply(calls=[tool_call('search',query='beta',mode='bm25')]))]
    rows=[]
    for i,(qid,query) in enumerate(data.queries.items()):
        observer=AgentObserver(budget=AgentBudget(max_tool_calls=12,max_query_calls=10,max_read_calls=6,
            max_evidence_tokens=4000,max_elapsed_ms=120000),counter=lambda s:count_tokens(s,tokenizer=tokenizer),
            counter_identity='reference-text:'+tokenizer_fingerprint(tokenizer))
        result=run_agent(query,client=scripted[i],tools=tools,model='qwen3.5:4b',max_turns=8,think=True,observer=observer)
        rows.append({'case_index':i,'id':qid,'result':asdict(result),'error':None,'stop_reason':result.stop_reason,'elapsed_ms':0})
    assert [r['stop_reason'] for r in rows]==['final','budget']
    write_json(run/'agentic_sample_ids.json',{'tasks':{'fixture':['1','2']}})
    write_json(run/'protocol.json',{'track':'bright','model':'qwen3.5:4b','think':True,'max_turns':8,
        'budget':{'tools':12,'queries':10,'reads':6,'evidence_tokens':4000,'cooperative_deadline_ms':120000},
        'selection_sha256':digest(run/'agentic_sample_ids.json'),'dataset_manifest_sha256':digest(tmp_path/'data/manifest.json')})
    write_json(run/'experiment.json',{'status':'completed'})
    shutil.copytree(repo/'src',run/'measured-source/src',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    def save():
        (run/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write_json(run/'checksums.json',{p.relative_to(run).as_posix():digest(p) for p in run.rglob('*') if p.is_file() and p.name!='checksums.json'})
    save()
    command=[sys.executable,str(repo/'evaluation/audits/replay_p4_agents.py'),str(run),'--dataset',str(tmp_path/'data')]
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    assert json.loads(result.stdout)['source_spans_verified']==3
    rows[0]['result']['observation']['evidence']['delivered_tokens']+=1;save()
    rejected=subprocess.run(command,capture_output=True,text=True)
    assert rejected.returncode!=0 and 'Evidence totals differ' in rejected.stderr
