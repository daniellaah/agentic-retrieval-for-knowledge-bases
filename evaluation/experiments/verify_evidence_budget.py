"""One declared live budget contract check; separate from comparative P2 runs."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

from arkb.agent import AgentBudget
from arkb.config import RuntimeConfig
from arkb.runtime import Runtime
from arkb.evaluation.runs import _code_metadata


def main():
    bundle,output=map(Path,sys.argv[1:]);bundle=bundle.resolve()
    if output.exists():raise ValueError('Do not overwrite a live verification attempt.')
    query='Read the complete body of 01_workflows_and_agents.md using read. Do not search.'
    budget=AgentBudget(max_tool_calls=1,max_query_calls=0,max_read_calls=1,max_evidence_tokens=1,max_elapsed_ms=60000)
    metadata={'purpose':'Live contract check with an intentionally tiny evidence budget; not a quality comparison',
              'query':query,'budget':asdict(budget),'status':'running','source':_code_metadata()}
    output.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    try:
        with Runtime(RuntimeConfig(offline=True,tokenizer_cache=Path('.uv-cache/tokenizers'))) as runtime:
            metadata['model_digest']=next(m.digest for m in runtime.model_client().list().models if m.model=='qwen3.5:4b')
            result=runtime.ask(query,db=bundle/'index.sqlite',notes_dir=bundle/'dataset/corpus',vault_id=bundle.name,
                               max_turns=2,observer=runtime.agent_observer(budget=budget))
            metadata['result']=asdict(result)
        events=result.observation['tools']
        valid=(result.stop_reason=='budget' and result.observation['budget_stop_reason']=='max_evidence_tokens'
            and result.response is None and len(events)==1 and events[0]['name']=='read' and events[0]['executed']
            and events[0]['returned_evidence_tokens']>1 and not events[0]['delivered_to_conversation']
            and result.observation['evidence']['delivered_tokens']==0 and len(result.observation['models'])==1
            and _code_metadata()==metadata['source'])
        metadata.update(status='completed',valid=valid)
    except Exception as error:
        metadata.update(status='failed',valid=False,error={'type':type(error).__name__,'message':str(error)});raise
    finally:output.write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'valid':valid,'stop_reason':result.stop_reason,'evidence':result.observation['evidence']}))
    if not valid:raise SystemExit(1)


if __name__=='__main__':main()
