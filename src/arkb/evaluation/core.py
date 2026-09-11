"""Core intake accountability, leakage-group split planning and dataset freezing.

Human attestations are explicit inputs; software cannot authenticate reviewers,
independence, permissions or whether a held-out label was actually kept private.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import shutil

from arkb.evaluation.outputs import fingerprint
from arkb.evaluation.v2 import load_dataset, TASKS


def plan_splits(entries, targets, *, seed):
    """Keep connected leakage groups intact; exposed families stay in dev.

    Deterministic greedy packing does not prove infeasibility: leftovers require
    review of the plan or more data, never breaking a group to meet a quota.
    """
    if set(targets) != {'dev','validation','test'} or any(type(v) is not int or v<0 for v in targets.values()):
        raise ValueError('Split targets must be nonnegative integer family counts.')
    if type(seed) is not int: raise ValueError('Integer split seed required.')
    groups=defaultdict(list); seen=set()
    for row in entries:
        if row['family_id'] in seen: raise ValueError('Duplicate intake family.')
        seen.add(row['family_id'])
        if not isinstance(row['leakage_group'],str) or not row['leakage_group'].strip():
            raise ValueError('Reviewed leakage groups required before split planning.')
        if type(row['seen_in_development']) is not bool: raise ValueError('Explicit exposure required.')
        groups[row['leakage_group']].append(row)
    assignments={}; counts=dict.fromkeys(targets,0); unassigned=[]
    forced={g for g,rows in groups.items() if any(r['seen_in_development'] for r in rows)}
    def assign(group, split):
        for row in groups[group]: assignments[row['family_id']]=split
        counts[split]+=len(groups[group])
    for group in sorted(forced): assign(group,'dev')
    order=sorted(set(groups)-forced,key=lambda g:(-len(groups[g]),fingerprint([seed,g])))
    for group in order:
        fitting=[s for s in targets if counts[s]+len(groups[group])<=targets[s]]
        if not fitting:
            unassigned.extend(r['family_id'] for r in groups[group]); continue
        split=max(fitting,key=lambda s:((targets[s]-counts[s])/max(targets[s],1),targets[s]-counts[s],s))
        assign(group,split)
    return {'assignments':assignments,'family_counts':counts,
            'remaining':{s:max(0,targets[s]-counts[s]) for s in targets},
            'over_target':{s:max(0,counts[s]-targets[s]) for s in targets},
            'unassigned_families':sorted(unassigned),'leakage_groups':len(groups),
            'complete':counts==targets and not unassigned, 'seed':seed}


def validate_pool_audit(dataset, row, directory):
    sha=row['pool_review']['artifact_sha256']
    path=Path(directory)/'pool-audits'/f'{sha}.json'
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=sha:
        raise ValueError('Missing or altered pool audit artifact.')
    audit=json.loads(path.read_text())
    cases={c['id']:c for c in dataset.cases if c['intent_family_id']==row['family_id']}
    if (set(audit)!={'schema_version','family_id','dataset_fingerprint','reviewer','queries','rationale'}
            or audit['schema_version']!='arkb-pool-audit-v1' or audit['family_id']!=row['family_id']
            or audit['dataset_fingerprint']!=fingerprint(dataset.manifest)
            or audit['reviewer']!=row['pool_review']['reviewer']
            or not isinstance(audit['rationale'],str) or not audit['rationale'].strip()
            or not isinstance(audit['queries'],dict) or set(audit['queries'])!=set(cases)):
        raise ValueError('Pool audit identity, rubric, reviewer or case set mismatch.')
    judged={(r['query_id'],r['source']) for r in dataset.qrels if r['annotation_status']=='reviewed'}
    for cid,query in audit['queries'].items():
        if set(query)!={'pool_methods','pooled_sources','outside_pool_sources'}:
            raise ValueError('Pool audit must retain pooling and outside-pool checks.')
        for key in query:
            values=query[key]
            if (not isinstance(values,list) or any(not isinstance(v,str) or not v.strip() for v in values)
                    or len(values)!=len(set(values))): raise ValueError('Invalid pool audit lists.')
        sources=set(query['pooled_sources']); outside=set(query['outside_pool_sources'])
        if sources & outside or (sources|outside)-set(dataset.bodies):
            raise ValueError('Overlapping or unknown audit sources.')
        if cases[cid]['task_type']!='no_retrieval':
            if len(query['pool_methods'])<2 or not sources:
                raise ValueError('Retrieval pool needs multiple methods and assessed candidates.')
            if sources!=set(dataset.bodies) and not outside:
                raise ValueError('Incomplete pool needs an outside-pool check.')
        if any((cid,source) not in judged for source in sources|outside):
            raise ValueError('Pool contains unreviewed query-document judgments.')


def core_readiness(dataset, registry, policy, *, directory=None):
    if (not isinstance(policy['corpus_id'],str) or not policy['corpus_id'].strip()
            or type(policy['minimum_documents']) is not int or policy['minimum_documents']<1):
        raise ValueError('Core policy requires an identity and positive document target.')
    families={c['intent_family_id'] for c in dataset.cases}
    entries=registry['families']
    if registry['schema_version']!='arkb-core-intake-v1': raise ValueError('Unsupported intake schema.')
    ids=[r['family_id'] for r in entries]
    if len(ids)!=len(set(ids)) or set(ids)!=families: raise ValueError('Registry must cover each family exactly once.')
    if registry['dataset_fingerprint']!=fingerprint(dataset.manifest): raise ValueError('Intake bound to a different dataset.')
    # A permanent exposure ledger prevents relabeling known dev families as unseen.
    exposed=set(policy['known_development_families'])
    known_queries=policy.get('known_development_queries',{})
    for case in dataset.cases:
        query_key=fingerprint(' '.join(case['query'].casefold().split()))
        if query_key in known_queries and known_queries[query_key]!=case['intent_family_id']:
            raise ValueError('Known development query cannot acquire a fresh family identity.')
    problems=[]; reviewed=[]
    for row in entries:
        if set(row)!={'family_id','leakage_group','seen_in_development','primary_task','origin','author',
                      'source_reference','use_permission','independence_review','pool_review'}:
            raise ValueError('Invalid family intake fields.')
        if row['primary_task'] not in TASKS or type(row['seen_in_development']) is not bool:
            raise ValueError('Invalid primary task/exposure.')
        if row['family_id'] in exposed and not row['seen_in_development']:
            raise ValueError('Known development family cannot become unseen.')
        if any(not isinstance(row[k],str) or not row[k].strip() for k in ('origin','author','source_reference')):
            raise ValueError('Each family needs provenance and authorship.')
        review=row['independence_review']
        if set(review)!={'status','reviewers','rationale'} or review['status'] not in ('pending','reviewed'):
            raise ValueError('Invalid independence review.')
        people=review['reviewers']
        if not isinstance(people,list) or any(not isinstance(x,str) or not x.strip() for x in people) or len(people)!=len(set(people)):
            raise ValueError('Invalid reviewers.')
        accepted=(review['status']=='reviewed' and bool(people) and row['author'] not in people
                  and isinstance(review['rationale'],str) and bool(review['rationale'].strip()))
        if accepted: reviewed.append(row['family_id'])
        else: problems.append({'family':row['family_id'],'reason':'independence_review_pending_or_not_independent'})
        if not isinstance(row['leakage_group'],str) or not row['leakage_group'].strip():
            problems.append({'family':row['family_id'],'reason':'leakage_group_pending'})
        if row['use_permission']!='confirmed': problems.append({'family':row['family_id'],'reason':'source_permission_pending'})
        pool=row['pool_review']
        if set(pool)!={'status','reviewer','artifact_sha256'} or pool['status'] not in ('pending','reviewed'):
            raise ValueError('Invalid pool review.')
        if (pool['status']!='reviewed' or not isinstance(pool['reviewer'],str) or not pool['reviewer'].strip()
                or not isinstance(pool['artifact_sha256'],str) or len(pool['artifact_sha256'])!=64
                or any(c not in '0123456789abcdef' for c in pool['artifact_sha256'])):
            problems.append({'family':row['family_id'],'reason':'judgment_pool_audit_pending'})
        else:
            try:
                if directory is None: raise ValueError('Pool audit directory required.')
                validate_pool_audit(dataset,row,directory)
            except (ValueError,OSError,TypeError,KeyError) as error:
                problems.append({'family':row['family_id'],'reason':'pool_audit_invalid','detail':str(error)})
    split=None
    if all(isinstance(r['leakage_group'],str) and r['leakage_group'].strip() for r in entries):
        split=plan_splits(entries,policy['split_targets'],seed=policy['split_seed'])
    total=sum(policy['split_targets'].values())
    blockers=[]
    if len(families)<total: blockers.append('independent_family_intake_shortfall')
    if dataset.provisional: blockers.append('provisional_annotations')
    if problems: blockers.append('family_review_or_provenance_incomplete')
    if split is None or not split['complete']: blockers.append('split_plan_incomplete')
    if len(dataset.bodies)<policy['minimum_documents']: blockers.append('corpus_document_shortfall')
    return {'queries':len(dataset.cases),'proposed_families':len(families),'reviewed_independent_families':len(reviewed),
        'target_families':total,'additional_proposed_families_needed':max(0,total-len(families)),
        'additional_verified_families_needed':max(0,total-len(reviewed)),
        'corpus_documents':len(dataset.bodies),'additional_documents_needed':max(0,policy['minimum_documents']-len(dataset.bodies)),
        'by_primary_task':dict(Counter(r['primary_task'] for r in entries)),
        'origin_counts':dict(Counter(r['origin'] for r in entries)),
        'problems':problems,'split_plan':split,'freeze_ready':not blockers,'blockers':blockers,
        'release_eligible':False,'limits':'Intake review attestations are not authenticated by software; a dataset freeze does not certify an experiment or output quality.'}


def freeze_core(dataset_dir, registry, policy, output):
    """Copy validated reviewed data, changing only grouped split assignments."""
    dataset_dir,output=Path(dataset_dir).resolve(),Path(output).resolve()
    if output.is_relative_to(dataset_dir):
        raise ValueError('Freeze destination must be outside the intake directory.')
    dataset=load_dataset(dataset_dir,notes_dir=dataset_dir/'corpus',allow_provisional=True)
    report=core_readiness(dataset,registry,policy,directory=dataset_dir)
    if not report['freeze_ready']: raise ValueError('Core freeze refused: '+', '.join(report['blockers']))
    if output.exists(): raise ValueError('Core freeze destination already exists.')
    shutil.copytree(dataset_dir,output)
    with (output/'queries.jsonl').open('w') as stream:
        for case in dataset.cases:
            stream.write(json.dumps({**case,'split':report['split_plan']['assignments'][case['intent_family_id']]},ensure_ascii=False)+'\n')
    manifest={**dataset.manifest,'corpus_id':policy['corpus_id'],'description':'Reviewed core freeze; exposure and grouped assignments retained in core-freeze.json.'}
    manifest['files']={name:hashlib.sha256((output/name).read_bytes()).hexdigest() for name in manifest['files']}
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    load_dataset(output,notes_dir=output/'corpus')
    (output/'core-freeze.json').write_text(json.dumps({'registry':registry,'policy':policy,'readiness':report,
        'source_dataset_fingerprint':fingerprint(dataset.manifest),'frozen_dataset_fingerprint':fingerprint(manifest)},ensure_ascii=False,indent=2)+'\n')
    return report
