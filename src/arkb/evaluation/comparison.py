"""Paired, family-weighted comparisons; repeated trials are not new tasks."""
from collections import defaultdict
from collections.abc import Mapping
import math
from statistics import mean

import numpy as np


def paired_family_bootstrap(left: Mapping[str,list[float]], right: Mapping[str,list[float]],
                            families: Mapping[str,str], *, seed=0, resamples=10000):
    """Average trials within case, cases within family; bootstrap whole families.

    Both arms must contain exactly the same cases. Missing/undefined outcomes
    require an explicit upstream policy; this function never silently drops them.
    Percentile intervals describe sampled-family uncertainty, not judge validity,
    representativeness, or a product reliability guarantee.
    """
    if not left or set(left)!=set(right) or set(families)!=set(left):
        raise ValueError('Comparison requires the same nonempty complete case set and family map.')
    if type(resamples) is not int or resamples<1000 or type(seed) is not int:
        raise ValueError('Use an integer seed and at least 1000 bootstrap resamples.')
    by_family=defaultdict(list)
    trial_counts={'left':0,'right':0}
    for case in sorted(left):
        family=families[case]
        if not isinstance(family,str) or not family.strip():
            raise ValueError('Each case requires a nonempty family.')
        for name,arm in [('left',left),('right',right)]:
            values=arm[case]
            if (not isinstance(values,(list,tuple)) or not values
                    or any(type(v) not in (int,float) or not math.isfinite(v) for v in values)):
                raise ValueError('Outcomes must be nonempty finite observed numeric trials.')
            trial_counts[name]+=len(values)
        by_family[family].append((mean(left[case]),mean(right[case])))
    if len(by_family)<2:
        raise ValueError('At least two independent families are required for a bootstrap interval.')
    family_values={family:(mean(a for a,b in values),mean(b for a,b in values))
                   for family,values in sorted(by_family.items())}
    values=np.asarray(list(family_values.values()),dtype=float)
    delta=values[:,1]-values[:,0]
    rng=np.random.default_rng(seed)
    # Bounded batches avoid allocating resamples * family_count in one matrix.
    samples=[]
    for start in range(0,resamples,256):
        indices=rng.integers(0,len(delta),size=(min(256,resamples-start),len(delta)))
        samples.extend(delta[indices].mean(axis=1).tolist())
    low,high=np.quantile(samples,[.025,.975])
    return {'aggregation':'equal family weight; equal case weight within family; mean trials within case',
            'case_count':len(left),'family_count':len(by_family),'trial_counts':trial_counts,
            'left_mean':float(values[:,0].mean()),'right_mean':float(values[:,1].mean()),
            'delta_right_minus_left':float(delta.mean()),'delta_ci95':[float(low),float(high)],
            'wins':int((delta>0).sum()),'ties':int((delta==0).sum()),'losses':int((delta<0).sum()),
            'seed':seed,'resamples':resamples,'family_scores':{k:list(v) for k,v in family_values.items()},
            'interpretation':'exploratory' if low<=0<=high else 'interval excludes zero; requires preregistered effect/cost gates'}
