import pytest

from arkb.evaluation.reviews import review_agreement,review_progress


def judgment(query,grade,reviewer):
    return {'query_id':query,'source':'a.md','grade':grade,'reviewer':reviewer,'rationale':'Read full document.'}


def test_blank_review_is_unknown_and_has_no_invented_time():
    expected={('q1','a.md'),('q2','a.md')}
    result,_=review_progress([judgment('q1',None,None)],expected)
    assert result['judged_pairs']==0 and result['pending_pairs']==2
    assert result['mean_recorded_ms'] is None and not result['release_eligible']


def test_agreement_denominator_uses_only_independent_completed_overlap():
    expected={('q1','a.md'),('q2','a.md'),('q3','a.md')}
    a=[judgment('q1',0,'alice'),judgment('q2',3,'alice'),judgment('q3',2,'alice')]
    b=[judgment('q1',0,'bob'),judgment('q2',2,'bob'),judgment('q3',None,None)]
    result=review_agreement(a,b,expected)
    assert result['double_judged_pairs']==2 and result['exact_agreement']==.5
    # Observed weighted disagreement 1/18; expected from marginals 7/18.
    assert result['quadratic_weighted_kappa']==pytest.approx(6/7)
    with pytest.raises(ValueError,match='different'):
        review_agreement(a,a,expected)


def test_constant_perfect_agreement_has_undefined_kappa():
    result=review_agreement([judgment('q',3,'a')],[judgment('q',3,'b')],{('q','a.md')})
    assert result['exact_agreement']==1 and result['quadratic_weighted_kappa'] is None


def test_invalid_judgment_and_duplicate_are_not_silently_dropped():
    row=judgment('q',3,'alice');expected={('q','a.md')}
    for changed in ({'grade':True},{'reviewer':None},{'rationale':''},{'duration_ms':-1}):
        with pytest.raises(ValueError):review_progress([{**row,**changed}],expected)
    with pytest.raises(ValueError,match='duplicate'):review_progress([row,row],expected)
