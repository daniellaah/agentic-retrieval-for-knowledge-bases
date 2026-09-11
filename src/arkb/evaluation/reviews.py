"""Audit incomplete human document judgments without filling missing labels."""
from collections import Counter
import math


def review_progress(rows, expected_pairs):
    expected=set(expected_pairs);seen=set();complete={};durations=[]
    for row in rows:
        pair=(row.get('query_id'),row.get('source'))
        if pair not in expected or pair in seen:
            raise ValueError('Unknown or duplicate judgment pair.')
        seen.add(pair)
        grade=row.get('grade');reviewer=row.get('reviewer');duration=row.get('duration_ms')
        if grade is None:
            continue
        if type(grade) is not int or not 0<=grade<=3 or not isinstance(reviewer,str) or not reviewer.strip():
            raise ValueError('A judgment requires integer grade 0..3 and reviewer identity.')
        rationale=row.get('rationale')
        if not isinstance(rationale,str) or not rationale.strip():
            raise ValueError('A judgment requires a recorded rationale.')
        if duration is not None:
            if type(duration) not in (int,float) or not math.isfinite(duration) or duration<=0:
                raise ValueError('Review duration must be positive finite milliseconds.')
            durations.append(duration)
        complete[pair]=row
    return {'expected_pairs':len(expected),'exported_pairs':len(seen),'judged_pairs':len(complete),
            'pending_pairs':len(expected)-len(complete),
            'coverage':len(complete)/len(expected) if expected else None,
            'by_reviewer':dict(Counter(row['reviewer'] for row in complete.values())),
            'timed_judgments':len(durations),'total_recorded_ms':sum(durations),
            'mean_recorded_ms':sum(durations)/len(durations) if durations else None,
            'human_independence_verified':False,'release_eligible':False},complete


def review_agreement(first, second, expected_pairs):
    a,judged_a=review_progress(first,expected_pairs);b,judged_b=review_progress(second,expected_pairs)
    matrix=[[0]*4 for _ in range(4)]
    for pair in judged_a.keys() & judged_b.keys():
        left,right=judged_a[pair],judged_b[pair]
        if left['reviewer']==right['reviewer']:
            raise ValueError('Two reviews of a pair must identify different reviewers.')
        matrix[left['grade']][right['grade']]+=1
    n=sum(map(sum,matrix));row_counts=list(map(sum,matrix));column_counts=[sum(row[j] for row in matrix) for j in range(4)]
    observed=sum(matrix[i][j]*((i-j)/3)**2 for i in range(4) for j in range(4))/n if n else None
    expected=sum(row_counts[i]*column_counts[j]*((i-j)/3)**2 for i in range(4) for j in range(4))/n**2 if n else None
    return {'first':a,'second':b,'double_judged_pairs':n,'confusion_matrix':matrix,
            'exact_agreement':sum(matrix[i][i] for i in range(4))/n if n else None,
            'quadratic_weighted_kappa':1-observed/expected if expected else None,
            'kappa_undefined_reason':'no overlap or no expected disagreement' if not expected else None,
            'limits':'Agreement checks consistency, not correctness or verified human independence.'}


def main():
    import argparse
    import hashlib
    import json
    from pathlib import Path
    from arkb.evaluation.v2 import load_dataset
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--review',type=Path,required=True)
    p.add_argument('--second',type=Path)
    args=p.parse_args()
    dataset=load_dataset(args.dataset,notes_dir=args.dataset/'corpus',allow_provisional=True)
    def read_queue(directory):
        metadata=json.loads((directory/'manifest.json').read_text())
        if metadata['dataset_manifest_sha256']!=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('Review belongs to a different dataset version.')
        return [json.loads(line) for line in (directory/'document-judgments.jsonl').read_text().splitlines()]
    expected={(c['id'],source) for c in dataset.cases if c['task_type']!='no_retrieval' for source in dataset.bodies}
    first=read_queue(args.review)
    report=review_agreement(first,read_queue(args.second),expected) if args.second else review_progress(first,expected)[0]
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
