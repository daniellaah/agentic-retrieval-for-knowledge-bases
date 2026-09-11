"""Prospective paired-binary power sensitivity; never estimate missing quality.

Simulate independent family pairs under explicit win/loss probabilities, then
apply the exact conditional two-sided McNemar (binomial sign) test. Assumptions
are caller-supplied: dev evidence coverage cannot supply human quality rates.
"""
from functools import lru_cache
import math

import numpy as np


@lru_cache(maxsize=4096)
def critical_minority(discordant, alpha=.05):
    if type(discordant) is not int or discordant<0: raise ValueError('Invalid discordant count.')
    cumulative=0.; threshold=-1
    for minority in range(discordant//2+1):
        probability=math.exp(math.lgamma(discordant+1)-math.lgamma(minority+1)
                             -math.lgamma(discordant-minority+1)-discordant*math.log(2))
        cumulative+=probability
        if min(1.,2*cumulative)<=alpha: threshold=minority
        else: break
    return threshold


def paired_binary_power(*, families, improvement, discordance, simulations=10000, seed=0, alpha=.05):
    if (type(families) is not int or families<2 or type(simulations) is not int or simulations<1000
            or type(seed) is not int or not 0<alpha<1 or not 0<=improvement<=discordance<=1):
        raise ValueError('Require n>=2, simulations>=1000 and 0<=improvement<=discordance<=1.')
    wins=(discordance+improvement)/2; losses=(discordance-improvement)/2
    samples=np.random.default_rng(seed).multinomial(families,[wins,losses,1-discordance],size=simulations)
    m=samples[:,0]+samples[:,1]; minority=np.minimum(samples[:,0],samples[:,1])
    thresholds=np.asarray([critical_minority(int(value),alpha) for value in m])
    # Reject only in the intended positive direction, not an adverse difference.
    power=float(np.mean((minority<=thresholds) & (samples[:,0]>samples[:,1])))
    return {'families':families,'improvement':improvement,'discordance':discordance,'simulations':simulations,
            'seed':seed,'alpha':alpha,'estimated_power':power,'monte_carlo_se':math.sqrt(power*(1-power)/simulations),
            'assumption':'One binary outcome per independent family; fixed scenario, not measured output quality.'}
