import pytest
from arkb.evaluation.multihop import parse_prediction,musique_metrics,answer_f1


def test_unknown_sources_duplicate_json_and_incomplete_execution_fail_closed():
    for text,reason in [('{"answer":"x","answerable":true,"support_sources":["bad.md"]}','final'),
                        ('{"answer":"x","answer":"y"}','final'),('','max_turns')]:
        p,error=parse_prediction(text,{'known.md':3},stopped=reason)
        assert error and p['predicted_answerable'] is None


def test_parse_support_mapping_and_no_answer_pair():
    p,e=parse_prediction('```json\n{"answer":"The Nobel Prize","answerable":true,"support_sources":["known.md"]}\n```',{'known.md':3})
    assert not e and p['predicted_support_idxs']==[3]
    assert answer_f1(p['predicted_answer'],['Nobel Prize'])==1
    gold=[{'id':'q','answerable':b,'answer':'Nobel Prize','answer_aliases':[],
           'paragraphs':[{'idx':3,'is_supporting':True}]} for b in (True,False)]
    predictions=[{'id':'q',**p},{'id':'q','predicted_answer':'','predicted_answerable':False,'predicted_support_idxs':[]}]
    assert musique_metrics(gold,predictions)['group_answer_sufficiency_f1']==1
    predictions[1]['predicted_answerable']=None
    assert musique_metrics(gold,predictions)['group_answer_sufficiency_f1']==0
    assert musique_metrics(gold,predictions)['answerability_accuracy']==.5
    with pytest.raises(ValueError):musique_metrics(gold[:1],predictions[:1])
