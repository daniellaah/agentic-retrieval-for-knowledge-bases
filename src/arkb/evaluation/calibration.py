"""Held-out binary judge calibration from explicitly supplied human outcomes.

One calibration outcome per family is required for the reported binomial
interval. Larger correlated judgment sets need a separately declared design.
"""
from collections import Counter
import math


def wilson_interval(successes,total):
    if type(successes) is not int or type(total) is not int or not 0<=successes<=total:
        raise ValueError('Invalid binomial counts.')
    if total==0:return None
    z=1.959963984540054;p=successes/total;denom=1+z*z/total
    centre=(p+z*z/(2*total))/denom
    half=z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/denom
    return [max(0.,centre-half),min(1.,centre+half)]


def calibration_report(rows,*,judge_identity,min_cases=100,max_false_accept_upper=.10,min_coverage=.95):
    if not isinstance(judge_identity,str) or not judge_identity.strip():raise ValueError('Judge identity is required.')
    if type(min_cases) is not int or min_cases<1:raise ValueError('min_cases must be positive.')
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0<=v<=1 for v in (max_false_accept_upper,min_coverage)):
        raise ValueError('Calibration thresholds must be probabilities.')
    ids=set();families={};calibration=[]
    required={'id','family_id','split','judge_identity','human_pass','judge_pass','reviewer'}
    for row in rows:
        if not isinstance(row,dict) or set(row)!=required:raise ValueError('Invalid calibration fields.')
        for key in ('id','family_id','reviewer'):
            if not isinstance(row[key],str) or not row[key].strip():raise ValueError('Calibration IDs and reviewer are required.')
        if row['id'] in ids:raise ValueError('Duplicate calibration item.')
        ids.add(row['id'])
        if row['split'] not in ('dev','calibration') or row['judge_identity']!=judge_identity:
            raise ValueError('Calibration split or judge version mismatch.')
        if type(row['human_pass']) is not bool or (row['judge_pass'] is not None and type(row['judge_pass']) is not bool):
            raise ValueError('Human truth must be boolean; missing judge outcomes remain null.')
        previous=families.setdefault(row['family_id'],row['split'])
        if previous!=row['split']:raise ValueError('Family leaks from judge tuning into calibration.')
        if row['split']=='calibration':calibration.append(row)
    if len({r['family_id'] for r in calibration})!=len(calibration):
        raise ValueError('Select one preregistered outcome per independent calibration family.')
    known=[r for r in calibration if r['judge_pass'] is not None]
    matrix=Counter(('TP' if r['human_pass'] else 'FP') if r['judge_pass'] else ('FN' if r['human_pass'] else 'TN') for r in known)
    counts={key:matrix[key] for key in ('TP','FP','TN','FN')}
    negatives=counts['FP']+counts['TN'];positives=counts['TP']+counts['FN']
    interval=wilson_interval(counts['FP'],negatives)
    coverage=len(known)/len(calibration) if calibration else None
    blockers=[]
    if len(calibration)<min_cases:blockers.append('insufficient_independent_calibration_cases')
    if coverage is None or coverage<min_coverage:blockers.append('insufficient_judge_coverage')
    if not positives:blockers.append('no_human_positive_cases')
    if interval is None or interval[1]>max_false_accept_upper:blockers.append('false_accept_bound_not_met')
    return {'judge_identity':judge_identity,'calibration_cases':len(calibration),'calibration_families':len(calibration),
            'tuning_cases':sum(r['split']=='dev' for r in rows),'judge_coverage':coverage,
            'confusion_matrix':counts,'conditional_false_accept_rate':counts['FP']/negatives if negatives else None,
            'conditional_false_accept_wilson95':interval,'positive_recall':counts['TP']/positives if positives else None,
            'missing_judge_cases':len(calibration)-len(known),'protocol_checks_passed':not blockers,'blockers':blockers,
            'thresholds':{'min_cases':min_cases,'max_false_accept_upper':max_false_accept_upper,'min_coverage':min_coverage},
            'human_independence_authenticated':False,'release_eligible':False,
            'limits':'Human outcomes are supplied, not generated here. Wilson assumes independent sampled families; representativeness and human correctness require external review.'}


def main():
    import argparse
    import json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('judgments',type=Path);p.add_argument('--judge-identity',required=True)
    a=p.parse_args();rows=[json.loads(line) for line in a.judgments.read_text().splitlines() if line.strip()]
    print(json.dumps(calibration_report(rows,judge_identity=a.judge_identity),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
