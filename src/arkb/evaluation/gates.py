"""Separate run completion, measurement validity and reviewed output quality."""
from collections import Counter


def evaluation_gate(rows,expected_keys,*,inputs_stable,replay_valid,dataset_provisional):
    keys=[(row['case_id'],row['arm'],row['trial']) for row in rows]
    expected=set(expected_keys)
    valid=len(keys)==len(set(keys)) and set(keys)==expected and inputs_stable and replay_valid
    reviewed=[r for r in rows if r['metrics']['review_status']=='human_reviewed']
    utilities=[r['metrics']['delivery_utility'] for r in rows]
    missing=sum(v is None for v in utilities)
    reasons=[]
    if not valid:reasons.append('invalid_or_incomplete_run')
    if dataset_provisional:reasons.append('provisional_dataset')
    if len(reviewed)!=len(rows):reasons.append('independent_output_review_incomplete')
    # This pilot is dev-only; a test protocol and quality/cost decision are separate.
    reasons.append('development_diagnostic_not_release_test')
    return {'experiment_valid':valid,'planned_rows':len(expected),'recorded_rows':len(rows),
            'missing_rows':len(expected-set(keys)),'duplicate_rows':len(keys)-len(set(keys)),
            'stop_reasons':dict(Counter(r['stop_reason'] for r in rows)),
            'human_reviewed_rows':len(reviewed),'undefined_delivery_utilities':missing,
            'mean_delivery_utility':sum(utilities)/len(utilities) if utilities and not missing else None,
            'release_eligible':False,'release_blockers':reasons,
            'note':'Runtime/budget failures retain zero utility; unreviewed final answers remain unknown, not passing.'}
