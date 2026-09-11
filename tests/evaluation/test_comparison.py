import pytest
from arkb.evaluation.comparison import paired_family_bootstrap


def test_repeated_trials_and_repeated_family_cases_do_not_inflate_n():
    left={'a':[0.,0.,0.],'b':[0.],'c':[1.]}
    right={'a':[1.,1.,1.],'b':[1.],'c':[1.]}
    families={'a':'one','b':'one','c':'two'}
    result=paired_family_bootstrap(left,right,families,seed=42,resamples=1000)
    assert result['family_count']==2 and result['case_count']==3
    assert result['delta_right_minus_left']==.5
    assert result['delta_ci95']==[0.,1.]
    assert result==paired_family_bootstrap(left,right,families,seed=42,resamples=1000)


def test_missing_arms_or_undefined_outcomes_are_never_dropped():
    with pytest.raises(ValueError,match='complete'):
        paired_family_bootstrap({'a':[1.]},{'b':[1.]},{'a':'a'})
    for invalid in (None,[],[None],[float('nan')]):
        with pytest.raises(ValueError,match='finite'):
            paired_family_bootstrap({'a':invalid,'b':[1.]},{'a':[1.],'b':[1.]},{'a':'a','b':'b'})


def test_family_weighting_and_constant_paired_improvement():
    r=paired_family_bootstrap({'a':[.1],'b':[.2]},{'a':[.3],'b':[.4]},
                              {'a':'a','b':'b'},resamples=1000)
    assert r['delta_right_minus_left']==pytest.approx(.2)
    assert r['delta_ci95']==pytest.approx([.2,.2])


def test_single_family_cannot_claim_population_confidence():
    with pytest.raises(ValueError,match='two independent'):
        paired_family_bootstrap({'a':[1.]},{'a':[1.]},{'a':'one'})
