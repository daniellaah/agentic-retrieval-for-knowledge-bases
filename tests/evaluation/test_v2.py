"""Evidence-level contracts that source membership alone cannot establish."""
from copy import deepcopy
import hashlib
import json

import pytest

from arkb.evaluation.v2 import load_dataset, evidence_scores, ranking_scores, SCHEMA
from arkb.knowledge.documents import load_notes


@pytest.fixture
def bundle(tmp_path):
    corpus=tmp_path/'corpus';corpus.mkdir()
    (corpus/'a.md').write_text('# A\n\nRule is 30 days. Exception is 7 days.\n')
    (corpus/'b.md').write_text('# B\n\nEquivalent rule.\n')
    notes={n.source:n for n in load_notes(corpus)}
    def span(eid,source,start,end):
        note=notes[source];quote=note.content[start:end]
        return dict(id=eid,source=source,document_revision=note.document_revision,start_char=start,end_char=end,
                    quote=quote,quote_sha256=hashlib.sha256(quote.encode()).hexdigest(),annotation_status='provisional',reviewers=[])
    evidence=[span('rule','a.md',0,16),span('exception','a.md',16,35),span('alternate','b.md',0,16)]
    case=dict(id='q1',intent_family_id='policy',task_type='knowledge_qa',query='What is the policy?',
              query_language='en',split='dev',answerability='answerable',query_origin='test',
              annotation_status='provisional',reviewers=[],evidence_requirements=[
                  dict(id='policy',weight=1,critical=True,alternatives=[['rule','exception'],['alternate']])],
              required_facts=['30 days except 7 days'],expected_behavior='Answer with the exception.')
    qrels=[dict(query_id='q1',source=s,grade=3,annotation_status='provisional',reviewers=[]) for s in notes]
    def save(cases=None,spans=None,rels=None):
        for name,rows in [('queries.jsonl',cases if cases is not None else [case]),
                          ('evidence.jsonl',spans if spans is not None else evidence),
                          ('qrels.jsonl',rels if rels is not None else qrels)]:
            (tmp_path/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        manifest=dict(schema_version=SCHEMA,corpus_id='test',description='test',
                      files={name:hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()
                             for name in ['queries.jsonl','evidence.jsonl','qrels.jsonl']},
                      corpus=[dict(source=n.source,document_revision=n.document_revision,
                                   body_sha256=hashlib.sha256(n.content.encode()).hexdigest(),origin='test') for n in notes.values()])
        (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    save()
    return tmp_path,corpus,notes,case,evidence,qrels,save


def test_provisional_opt_in_and_hash_validation(bundle):
    directory,corpus,*_=bundle
    with pytest.raises(ValueError,match='Provisional'):
        load_dataset(directory,notes_dir=corpus)
    assert load_dataset(directory,notes_dir=corpus,allow_provisional=True).provisional
    with (directory/'queries.jsonl').open('a') as f:f.write('\n')
    with pytest.raises(ValueError,match='checksum'):
        load_dataset(directory,notes_dir=corpus,allow_provisional=True)


def test_or_of_and_evidence_requires_exact_revision_and_actual_text(bundle):
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    def hit(source,start,end):
        return dict(source=source,document_revision=notes[source].document_revision,
                    start_char=start,end_char=end,content=notes[source].content[start:end])
    partial=hit('a.md',0,16)
    scores=evidence_scores(case,[partial,partial],dataset)
    assert scores['evidence_coverage']==0 and not scores['all_critical_facets_covered']
    assert scores['span_coverage']['exception']==0
    whole=hit('a.md',0,35)
    assert evidence_scores(case,[whole],dataset)['evidence_coverage']==1
    assert evidence_scores(case,[hit('b.md',0,16)],dataset)['evidence_coverage']==1
    for changed in ({'document_revision':'stale'},{'content':'unsupported'},{'end_char':True}):
        score=evidence_scores(case,[{**whole,**changed}],dataset)
        assert score['evidence_coverage']==0 and score['rejected_observations']==1
    assert evidence_scores(case,[whole],dataset)['grounded_task_success'] is None


def test_overlapping_ranges_cover_once_and_can_join_adjacent_spans(bundle):
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    hits=[dict(source='a.md',document_revision=notes['a.md'].document_revision,
               start_char=a,end_char=b,content=notes['a.md'].content[a:b]) for a,b in [(0,20),(10,35)]]
    assert evidence_scores(case,hits,dataset)['evidence_coverage']==1


@pytest.mark.parametrize('mutation,match',[
    ('split','leaks'),('span','coordinates'),('quote','quote'),('unknown','unknown'),
    ('qrel','supporting'),('extra','fields'),('review','reviewers'),('weight','weight')])
def test_invalid_annotation_bundles_fail_before_execution(bundle,mutation,match):
    directory,corpus,notes,case,evidence,qrels,save=bundle
    cases=[deepcopy(case)];spans=deepcopy(evidence);rels=deepcopy(qrels)
    if mutation=='split':cases.append({**deepcopy(case),'id':'q2','query':'Other question','split':'test'})
    elif mutation=='span':spans[0]['start_char']=True
    elif mutation=='quote':spans[0]['quote']='wrong'
    elif mutation=='unknown':cases[0]['evidence_requirements'][0]['alternatives']=[['missing']]
    elif mutation=='qrel':rels[0]['grade']=0
    elif mutation=='extra':cases[0]['gold_in_prompt']=True
    elif mutation=='review':cases[0]['annotation_status']='reviewed'
    elif mutation=='weight':cases[0]['evidence_requirements'][0]['weight']=float('nan')
    save(cases,spans,rels)
    with pytest.raises(ValueError,match=match if mutation!='weight' else 'Nonfinite'):
        load_dataset(directory,notes_dir=corpus,allow_provisional=True)


def test_ranking_cutoff_gain_and_unassessed_are_explicit():
    qrels={'best':3,'background':1,'also':2}
    result=ranking_scores(qrels,['unknown','background','best'],k=2)
    assert result['recall@2']==0 and result['mrr@2']==0
    assert result['assessed@2']==.5 and result['unassessed_returned_count']==1
    # DCG=1/log2(3); IDCG=7+3/log2(3); background earns graded gain, not relevant recall.
    assert result['ndcg_exp@2']==pytest.approx(0.07094846566967604)
    short=ranking_scores({'best':3},['best'],k=10)
    assert short['precision@10']==.1 and short['assessed@10']==.1
    assert short['recall@10']==1
    assert ranking_scores({},[],k=10)['recall@10'] is None


def test_unanswerable_is_distinct_from_no_retrieval(bundle):
    directory,corpus,notes,case,evidence,qrels,save=bundle
    case.update(task_type='evidence_gap',answerability='unanswerable',evidence_requirements=[],required_facts=[])
    save([case],[],[])
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    score=evidence_scores(case,[],dataset)
    assert score['evidence_coverage'] is None
    assert score['grounded_task_success'] is None


def test_pilot_scores_actual_serialized_tool_evidence_and_separates_contracts(bundle):
    from arkb.agent.tools import _evidence
    from arkb.retrieval.models import SearchResult
    from arkb.evaluation.pilot import score_case
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    hit=SearchResult(source_id='a',source='a.md',content=notes['a.md'].content,
                     method='read',start_char=0,end_char=len(notes['a.md'].content),
                     metadata={'document_revision':notes['a.md'].document_revision})
    observations=json.loads(json.dumps([_evidence(hit)]))
    read=score_case(case,observations,dataset,method='explicit_read')
    assert read['source_metrics_from_top_k_chunks'] is None
    assert read['evidence']['evidence_coverage']==1
    match=score_case(case,observations,dataset,method='explicit_match')
    assert match['source_metrics_from_top_k_chunks'] is None
    assert match['exact_source_set_equal'] is False  # qrels also requires b.md.
    ranked=score_case(case,observations,dataset,method='semantic')
    assert ranked['source_metrics_from_top_k_chunks']['recall@10']==.5
    observations[0]['content']='The source ID alone is not evidence.'
    assert score_case(case,observations,dataset,method='semantic')['evidence']['evidence_coverage']==0


def test_pilot_protocol_mismatch_fails_before_creating_run_or_calling_services(bundle):
    from arkb.evaluation.pilot import run_pilot
    directory,corpus,*_=bundle
    protocol=directory/'protocol.json';protocol.write_text('{}')
    output=directory/'output'
    with pytest.raises(ValueError,match='protocol'):
        run_pilot(directory,output,qdrant_url='http://must-not-be-contacted.invalid',
                  protocol_path=protocol,allow_provisional=True)
    assert not output.exists()


def test_unlocated_full_body_is_evidence_but_unlocated_partial_or_stale_text_is_not(bundle):
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    hit={'source':'a.md','document_revision':notes['a.md'].document_revision,'content':notes['a.md'].content}
    assert evidence_scores(case,[hit],dataset)['evidence_coverage']==1
    for changes in ({'content':notes['a.md'].content[:16]},{'document_revision':'stale'},{'start_char':0}):
        assert evidence_scores(case,[{**hit,**changes}],dataset)['evidence_coverage']==0


def test_output_review_binds_exact_answer_and_does_not_turn_evidence_into_correctness(bundle):
    from arkb.evaluation.outputs import output_packet,score_output,pending_review
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    hit={'id':'E0','source':'a.md','document_revision':notes['a.md'].document_revision,'content':notes['a.md'].content}
    packet=output_packet(case,'Wrong: retain for 999 days.',[hit],stop_reason='final')
    score=score_output(packet,dataset)
    assert score['evidence']['evidence_coverage']==1
    assert score['grounded_task_success'] is None and score['delivery_utility'] is None
    review=pending_review(packet)
    review.update(status='human_reviewed',reviewer='test-reviewer',rationale='Fixture marks a clearly incorrect number.')
    review['criteria']={k:False for k in review['criteria']}
    assert score_output(packet,dataset,review=review)['grounded_task_success'] is False
    other=output_packet(case,'Different answer.',[hit],stop_reason='final')
    with pytest.raises(ValueError,match='different'):score_output(other,dataset,review=review)
    with pytest.raises(ValueError,match='hash'):score_output({**packet,'answer':'changed'},dataset)


def test_gold_context_uses_one_or_alternative_and_preserves_and_spans(bundle):
    from arkb.evaluation.outputs import gold_context
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    context=gold_context(case,dataset)
    assert len(context)==2 and {r['source'] for r in context}=={'a.md'}
    assert evidence_scores(case,context,dataset)['evidence_coverage']==1


def test_unique_evidence_counter_merges_partial_overlap_without_gold_requirements(bundle):
    from arkb.evaluation.v2 import unique_evidence_tokens
    directory,corpus,notes,case,*_=bundle
    dataset=load_dataset(directory,notes_dir=corpus,allow_provisional=True)
    hits=[{'source':'a.md','document_revision':notes['a.md'].document_revision,'start_char':a,'end_char':b,
           'content':notes['a.md'].content[a:b]} for a,b in [(0,20),(10,25),(0,20),(30,35)]]
    result=unique_evidence_tokens(hits,dataset,counter=len)
    assert result['union_reference_tokens']==30 and result['merged_intervals']==2
    assert result['documents']==1 and result['rejected_observations']==0
