from math import comb
import pytest
from arkb.evaluation.power import critical_minority,paired_binary_power


def test_exact_critical_counts_match_small_enumerated_binomial():
    for n in range(31):
        expected=max((k for k in range(n//2+1) if 2*sum(comb(n,i) for i in range(k+1))/2**n<=.05),default=-1)
        assert critical_minority(n)==expected


def test_prospective_power_is_seeded_and_not_defined_from_missing_labels():
    options=dict(families=120,improvement=.05,discordance=.1,seed=1)
    assert paired_binary_power(**options)==paired_binary_power(**options)
    assert paired_binary_power(families=120,improvement=0,discordance=0)['estimated_power']==0
    assert paired_binary_power(families=20,improvement=1,discordance=1)['estimated_power']==1
    with pytest.raises(ValueError):paired_binary_power(families=120,improvement=.3,discordance=.1)
