from dataclasses import asdict
from pathlib import Path

import pytest

from arkb.evaluation.controlled import component_rows, fused_candidates, stopping_diagnostics, stopping_order
from arkb.evaluation.v2 import load_dataset
from arkb.retrieval.models import SearchResult

ROOT = Path(__file__).resolve().parents[2]


def fixture():
    path = ROOT/'evaluation/data/v2/pilot'
    dataset = load_dataset(path, notes_dir=path/'corpus', allow_provisional=True)
    case = dataset.cases[0]
    hits = [asdict(SearchResult(source_id=d['source'], source=d['source'], content=dataset.bodies[d['source']],
            method='semantic', metadata={'document_revision':d['document_revision']})) for d in dataset.manifest['corpus']]
    return dataset, case, {'bm25':hits[:40], 'semantic':list(reversed(hits[:40]))}


def test_fixed_pool_scoring_preserves_candidates_and_replays_losses():
    import json
    dataset, case, legs = fixture()
    protocol = json.loads((ROOT/'evaluation/experiments/p3-protocol.json').read_text())
    _, pool = fused_candidates(legs, 20)
    raw = {'case_id':case['id'], 'query':case['query'], 'legs':legs,
           'rerank':{'candidates':[asdict(h) for h in pool], 'scores':list(range(len(pool))),
                     'identity':'frozen-test', 'score_type':'test_relevance'}}
    rows = component_rows(raw, case, dataset, protocol)
    assert len(rows)==4
    assert [r['candidate_union_count'] for r in rows]==[20,40,40,40]
    assert all(r['fusion_pool_count']==20 and len(r['observations'])==10 for r in rows)
    assert rows[1]['stage_evidence']['fusion_pool']==rows[3]['stage_evidence']['fusion_pool']
    assert [h['source'] for h in rows[3]['observations']]==[h.source for h in reversed(pool[-10:])]
    raw['rerank']['candidates'].reverse()
    with pytest.raises(ValueError, match='frozen pool'): component_rows(raw,case,dataset,protocol)


def test_crossover_balances_first_arm_within_each_case():
    cfg={'cases':['a','b','c','d'],'arms':{'on':True,'off':False},'trials':2}
    order=stopping_order(cfg)
    assert len(order)==len(set(order))==16
    for cid in cfg['cases']:
        first=[next(a for c,a,t in order if c==cid and t==trial) for trial in range(2)]
        assert set(first)=={'on','off'}
    with pytest.raises(ValueError):stopping_order({**cfg,'cases':['a','a']})


def test_threshold_only_counts_evidence_submitted_before_a_model_request():
    dataset,case,_=fixture()
    from arkb.evaluation.outputs import gold_context
    evidence=gold_context(case,dataset)
    event={'submitted_to_model':False,'turn':1,'name':'search','raw_result':{'results':evidence},'executed':True}
    model={'turn':1,'request':{'messages':[]}}
    report={'models':[model],'tools':[event]}
    assert stopping_diagnostics(case,report,dataset)['first_full_evidence_model_turn'] is None
    event['submitted_to_model']=True
    report['models'].append({'turn':2,'request':{'messages':[]}})
    report['tools'].append({**event,'turn':2})
    score=stopping_diagnostics(case,report,dataset)
    assert score['first_full_evidence_model_turn']==2
    assert score['executed_calls_after_label_threshold']==1
