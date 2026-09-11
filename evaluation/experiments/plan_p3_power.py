"""Record paired quality-test power under explicit, unmeasured scenarios."""
import argparse
import hashlib
import json
from pathlib import Path
from arkb.evaluation.power import paired_binary_power


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    results=[paired_binary_power(families=n,improvement=delta,discordance=q,seed=20260910)
             for q in (.1,.3) for delta in (.03,.05,.1) for n in (60,120,300,600)]
    summary=[]
    for q in (.1,.3):
        for delta in (.03,.05,.1):
            passing=[r['families'] for r in results if r['discordance']==q and r['improvement']==delta and r['estimated_power']>=.8]
            summary.append({'discordance':q,'improvement':delta,'first_grid_size_at_power_80':min(passing) if passing else None})
    payload={'status':'prospective_sensitivity_only','independently_reviewed_quality_pairs':0,
             'results':results,'grid_decisions':summary,'target_power':.8,
             'quality_sample_size_selected':None,
             'limitations':['Discordance scenarios are assumed, not measured from provisional evidence scores.',
                 'Binary one-outcome-per-family assumption; correlated variants/trials do not increase n.',
                 'Exact two-sided binomial test at alpha .05; intended-direction detection only.',
                 'This superiority power exercise does not establish noninferiority or validate the dataset.'],
             'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'power_source_sha256':hashlib.sha256(Path('src/arkb/evaluation/power.py').read_bytes()).hexdigest()}
    with a.output.open('x') as stream:json.dump(payload,stream,ensure_ascii=False,indent=2);stream.write('\n')
    print(json.dumps(summary))


if __name__=='__main__':main()
