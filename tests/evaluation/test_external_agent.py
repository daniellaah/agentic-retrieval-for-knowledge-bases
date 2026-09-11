from copy import deepcopy
import pytest
from arkb.evaluation.external_agent import evidence_packet,coverage


def result():
    def event(i,delivered,submitted,source):
        return {'index':i,'turn':i+1,'name':'search','delivered_to_conversation':delivered,
                'submitted_to_model':submitted,'raw_result':{'results':[{'source':source,'content':'body'}]}}
    return {'observation':{'models':[{'turn':1},{'turn':2}],
        'tools':[event(0,True,True,'a.md'),event(1,True,False,'b.md'),event(2,False,False,'c.md')]}}


def test_withheld_and_last_turn_evidence_cannot_earn_submitted_coverage():
    packet=evidence_packet(result(),{'a.md':'a','b.md':'b','c.md':'c'})
    assert packet['returned']==['a','b','c']
    assert packet['delivered']==['a','b'] and packet['submitted']==['a']
    scores=coverage({'a':1,'b':1,'c':1},[{'weight':2,'supporting_docs':['a']},
        {'weight':1,'supporting_docs':['b','c']}],packet)
    assert scores['submitted']['positive_document_recall']==pytest.approx(1/3)
    assert scores['submitted']['weighted_aspect_coverage']==pytest.approx(2/3)
    assert scores['delivered']['weighted_aspect_coverage']==1


def test_corrupt_flags_unknown_sources_and_duplicate_observations():
    value=result();value['observation']['tools'][2]['submitted_to_model']=True
    with pytest.raises(ValueError,match='flags'):evidence_packet(value,{'a.md':'a','b.md':'b','c.md':'c'})
    with pytest.raises(ValueError,match='corpus'):evidence_packet(result(),{})
    value=result();value['observation']['tools'].append(deepcopy(value['observation']['tools'][0]))
    assert evidence_packet(value,{'a.md':'a','b.md':'b','c.md':'c'})['submitted']==['a']
