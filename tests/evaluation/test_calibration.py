import pytest
from arkb.evaluation.calibration import calibration_report,wilson_interval


def row(i,truth,predicted,split='calibration'):
    return {'id':str(i),'family_id':str(i),'split':split,'judge_identity':'j1',
            'human_pass':truth,'judge_pass':predicted,'reviewer':'fixture-reviewer'}


def test_no_human_labels_produces_pending_not_perfect_calibration():
    r=calibration_report([],judge_identity='j1')
    assert r['conditional_false_accept_rate'] is None and r['conditional_false_accept_wilson95'] is None
    assert not r['protocol_checks_passed'] and r['judge_coverage'] is None


def test_false_accept_rate_uses_bad_answers_and_reports_missing_coverage():
    r=calibration_report([row(1,False,True),row(2,False,False),row(3,True,True),row(4,True,None)],judge_identity='j1')
    assert r['confusion_matrix']=={'TP':1,'FP':1,'TN':1,'FN':0}
    assert r['conditional_false_accept_rate']==.5 and r['judge_coverage']==.75
    assert not r['protocol_checks_passed']


def test_zero_false_accepts_still_has_uncertainty_and_small_sample_cannot_pass():
    assert wilson_interval(0,10)[1]==pytest.approx(.2775327998628892)
    r=calibration_report([row(i,False,False) for i in range(10)]+[row(11,True,True)],judge_identity='j1',min_cases=10)
    assert 'false_accept_bound_not_met' in r['blockers']


def test_family_leakage_and_repeated_family_pseudoreplication_are_rejected():
    a=row(1,False,False);b={**row(2,False,False),'family_id':'1'}
    with pytest.raises(ValueError,match='one preregistered'):calibration_report([a,b],judge_identity='j1')
    b['split']='dev'
    with pytest.raises(ValueError,match='leaks'):calibration_report([a,b],judge_identity='j1')
