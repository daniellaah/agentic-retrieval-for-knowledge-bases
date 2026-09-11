"""Opt-in accounting and online limits; never reads evaluation labels.

The conversation remains the source of AgentTrace. This observer records
operation status, raw provider usage and withheld observations separately.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
from time import perf_counter


@dataclass(frozen=True, kw_only=True)
class AgentBudget:
    max_tool_calls: int | None = None
    max_query_calls: int | None = None
    max_read_calls: int | None = None
    max_evidence_tokens: int | None = None
    max_elapsed_ms: int | None = None

    def __post_init__(self):
        for key,value in asdict(self).items():
            if value is not None and (type(value) is not int or value<0):
                raise ValueError(f'{key} must be a nonnegative integer or null.')


class AgentObserver:
    """Single-run observer. A counter counts reference text, not model prompts.

    Evidence limits charge every delivered excerpt including duplicates. If a
    whole observation exceeds the remaining allowance, keep its raw result but
    withhold it from the conversation and stop. Deadline checks cannot cancel
    an in-flight synchronous request; late results are retained but withheld.
    """

    def __init__(self, *, budget=None, counter=None, counter_identity=None, clock=perf_counter):
        if budget is not None and not isinstance(budget,AgentBudget):
            raise ValueError('budget must be an AgentBudget.')
        if budget is not None and budget.max_evidence_tokens is not None and counter is None:
            raise ValueError('Evidence budgets require a reference token counter.')
        if counter is not None and (not callable(counter) or not isinstance(counter_identity,str) or not counter_identity.strip()):
            raise ValueError('A counter requires a stable nonblank identity.')
        self.budget,self.counter,self.counter_identity=budget,counter,counter_identity
        self.clock=clock;self.started=None;self.models=[];self.tools=[]
        self.reason=None;self.error=None;self._unique={}

    def start(self):
        if self.started is not None:raise ValueError('Create a fresh observer for every run.')
        self.started=self.clock()

    def elapsed(self):return (self.clock()-self.started)*1000

    def count(self,text):
        if self.counter is None:return None
        value=self.counter(text)
        if type(value) is not int or value<0:raise ValueError('Token counter returned an invalid count.')
        return value

    def deadline(self):
        if self.budget and self.budget.max_elapsed_ms is not None and self.elapsed()>=self.budget.max_elapsed_ms:
            self.reason='max_elapsed_ms';return True
        return False

    def start_model(self,turn,request):
        for event in self.tools:
            if event['delivered_to_conversation']:event['submitted_to_model']=True
        event={'turn':turn,'request':deepcopy(request),'response':None,'status':'running',
               'started_ms':self.elapsed(),'elapsed_ms':None,'usage':None,
               'request_json_reference_tokens':self.count(json.dumps(request,ensure_ascii=False,sort_keys=True)),
               'error':None}
        self.models.append(event)

    def end_model(self,response):
        event=self.models[-1]
        event.update(status='success',response=response.model_dump(exclude_none=True),
                     elapsed_ms=self.elapsed()-event['started_ms'])
        fields=('prompt_eval_count','eval_count','total_duration','load_duration','prompt_eval_duration','eval_duration')
        event['usage']={key:getattr(response,key,None) for key in fields}

    def request_tools(self,turn,calls):
        events=[]
        for call in calls:
            event={'turn':turn,'index':len(self.tools),'name':call.function.name,
                   'arguments':deepcopy(call.function.arguments),'status':'requested','executed':False,
                   'started_ms':None,'elapsed_ms':None,'raw_result':None,'error':None,
                   'delivered_to_conversation':False,'submitted_to_model':False,
                   'returned_evidence_tokens':None,'delivered_evidence_tokens':None,'skip_reason':None}
            self.tools.append(event);events.append(event)
        return events

    def permit_tool(self,event):
        if self.deadline():return False
        if self.budget:
            executed=[e for e in self.tools if e['executed']]
            checks=[('max_tool_calls',len(executed)),
                    ('max_query_calls',sum(e['name'] in ('match','search') for e in executed)),
                    ('max_read_calls',sum(e['name']=='read' for e in executed))]
            for key,count in checks:
                applies=key=='max_tool_calls' or (key=='max_query_calls' and event['name'] in ('match','search')) or (key=='max_read_calls' and event['name']=='read')
                limit=getattr(self.budget,key)
                if applies and limit is not None and count>=limit:
                    self.reason=key;return False
        return True

    def start_tool(self,event):
        event.update(executed=True,status='running',started_ms=self.elapsed())

    def end_tool(self,event,result):
        event.update(status='success',elapsed_ms=self.elapsed()-event['started_ms'],raw_result=deepcopy(result))
        hits=[result.get('result')] if event['name']=='read' else result.get('results',[])
        texts=[h for h in hits if isinstance(h,dict) and isinstance(h.get('content'),str)]
        count=sum(self.count(h['content']) for h in texts) if self.counter is not None else None
        event['returned_evidence_tokens']=count
        if self.deadline():return False
        delivered=sum(e['delivered_evidence_tokens'] or 0 for e in self.tools)
        if self.budget and self.budget.max_evidence_tokens is not None and delivered+count>self.budget.max_evidence_tokens:
            self.reason='max_evidence_tokens';return False
        event.update(delivered_to_conversation=True,delivered_evidence_tokens=count)
        for h in texts:
            identity=json.dumps({key:h.get(key) for key in ('source','document_revision','start_char','end_char','content')},sort_keys=True,ensure_ascii=False)
            self._unique[identity]=self.count(h['content'])
        return True

    def fail(self,error,stage,event=None):
        detail={'stage':stage,'type':type(error).__name__,'message':str(error)}
        self.error=detail
        target=event if event is not None else (self.models[-1] if stage in ('model_request','model_protocol') and self.models else None)
        if target is not None:
            target.update(status='error',error=detail)
            if target['started_ms'] is not None:target['elapsed_ms']=self.elapsed()-target['started_ms']

    def finish(self,stop_reason):
        for event in self.tools:
            if event['status']=='requested':
                event.update(status='skipped',skip_reason=self.reason or 'prior_error')
        def usage(key):
            values=[e['usage'].get(key) if e['usage'] else None for e in self.models]
            known=[v for v in values if type(v) is int and v>=0]
            return {'total':sum(known) if len(known)==len(values) else None,
                    'known_total':sum(known),'defined_requests':len(known),'requests':len(values)}
        return deepcopy({'schema_version':'arkb-agent-observation-v1','budget':asdict(self.budget) if self.budget else None,
            'counter_identity':self.counter_identity,'models':self.models,'tools':self.tools,
            'stop_reason':stop_reason,'budget_stop_reason':self.reason,'error':self.error,'elapsed_ms':self.elapsed(),
            'usage':{key:usage(key) for key in ('prompt_eval_count','eval_count')},
            'evidence':{'returned_tokens':sum(e['returned_evidence_tokens'] or 0 for e in self.tools) if self.counter else None,
                        'delivered_tokens':sum(e['delivered_evidence_tokens'] or 0 for e in self.tools) if self.counter else None,
                        'unique_exact_excerpt_tokens':sum(self._unique.values()) if self.counter else None},
            'limits':['Deadline is cooperative; in-flight work may exceed it and is not cancelled.',
                      'Reference JSON tokens are not provider-rendered context tokens.',
                      'Exact-excerpt deduplication still counts overlapping different excerpts.',
                      'Delivered observations on the last tool turn may never be submitted to a model.']})
