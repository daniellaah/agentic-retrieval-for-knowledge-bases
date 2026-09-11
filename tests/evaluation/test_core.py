from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from arkb.evaluation.core import core_readiness, freeze_core, plan_splits
from arkb.evaluation.outputs import fingerprint
from arkb.evaluation.v2 import load_dataset

ROOT=Path(__file__).resolve().parents[2]


def test_known_exposure_propagates_to_whole_leakage_group():
    entries=[{'family_id':'a','leakage_group':'related','seen_in_development':True},
             {'family_id':'b','leakage_group':'related','seen_in_development':False},
             {'family_id':'c','leakage_group':'fresh','seen_in_development':False}]
    plan=plan_splits(entries,{'dev':2,'validation':0,'test':1},seed=7)
    assert plan['assignments']=={'a':'dev','b':'dev','c':'test'} and plan['complete']
    assert plan==plan_splits(list(reversed(entries)),{'dev':2,'validation':0,'test':1},seed=7)
    impossible=plan_splits(entries,{'dev':1,'validation':1,'test':1},seed=7)
    assert not impossible['complete'] and impossible['over_target']['dev']==1


def test_pilot_readiness_never_promotes_provisional_or_counts_variants_as_families(tmp_path):
    p=ROOT/'evaluation/data/v2/pilot';intake=ROOT/'evaluation/data/v2/core-intake'
    dataset=load_dataset(p,notes_dir=p/'corpus',allow_provisional=True)
    registry=json.loads((intake/'pilot-registry.json').read_text());policy=json.loads((intake/'core-policy.json').read_text())
    report=core_readiness(dataset,registry,policy,directory=p)
    assert (report['queries'],report['proposed_families'],report['reviewed_independent_families'])==(60,31,0)
    assert report['additional_proposed_families_needed']==269
    assert report['split_plan']['family_counts']=={'dev':31,'validation':0,'test':0}
    with pytest.raises(ValueError,match='refused'):freeze_core(p,registry,policy,tmp_path/'no-freeze')
    assert not (tmp_path/'no-freeze').exists()
    changed=deepcopy(registry);changed['families'][0]['seen_in_development']=False
    with pytest.raises(ValueError,match='cannot become unseen'):core_readiness(dataset,changed,policy)
    changed=deepcopy(registry);changed['dataset_fingerprint']='bad'
    with pytest.raises(ValueError,match='different dataset'):core_readiness(dataset,changed,policy)
    renamed=registry['families'][0]['family_id']
    changed=deepcopy(registry);changed['families'][0]['family_id']='fresh-id'
    modified=replace(dataset,cases=tuple({**c,'intent_family_id':'fresh-id'} if c['intent_family_id']==renamed else c for c in dataset.cases))
    with pytest.raises(ValueError,match='fresh family identity'):core_readiness(modified,changed,policy)


def make_reviewed_fixture(tmp_path):
    source=ROOT/'evaluation/data/v2/pilot';output=tmp_path/'intake';output.mkdir();(output/'corpus').mkdir()
    data=load_dataset(source,notes_dir=source/'corpus',allow_provisional=True)
    case=next(c for c in data.cases if c['task_type']=='no_retrieval')
    case={**case,'annotation_status':'reviewed','reviewers':['fixture-reviewer']}
    for name,rows in [('queries.jsonl',[case]),('qrels.jsonl',[]),('evidence.jsonl',[])]:
        (output/name).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    doc=data.manifest['corpus'][0];shutil.copyfile(source/'corpus'/doc['source'],output/'corpus'/doc['source'])
    manifest={**data.manifest,'corpus':[doc],'files':{n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in data.manifest['files']}}
    (output/'manifest.json').write_text(json.dumps(manifest))
    audit={'schema_version':'arkb-pool-audit-v1','family_id':case['intent_family_id'],'dataset_fingerprint':fingerprint(manifest),
           'reviewer':'fixture-reviewer','queries':{case['id']:{'pool_methods':[],'pooled_sources':[],'outside_pool_sources':[]}},
           'rationale':'Test fixture: no retrieval is required.'}
    payload=json.dumps(audit).encode();sha=hashlib.sha256(payload).hexdigest();(output/'pool-audits').mkdir()
    (output/'pool-audits'/f'{sha}.json').write_bytes(payload)
    entry={'family_id':case['intent_family_id'],'leakage_group':'fixture','seen_in_development':True,'primary_task':'no_retrieval',
           'origin':'test_fixture','author':'fixture-author','source_reference':'unit-test','use_permission':'confirmed',
           'independence_review':{'status':'reviewed','reviewers':['fixture-reviewer'],'rationale':'Independent test fixture.'},
           'pool_review':{'status':'reviewed','reviewer':'fixture-reviewer','artifact_sha256':sha}}
    registry={'schema_version':'arkb-core-intake-v1','dataset_fingerprint':fingerprint(manifest),'families':[entry]}
    policy={'corpus_id':'test-core','split_targets':{'dev':1,'validation':0,'test':0},'split_seed':7,'minimum_documents':1,
            'known_development_families':[case['intent_family_id']]}
    return output,registry,policy


def test_freeze_requires_bound_review_artifact_and_never_overwrites(tmp_path):
    intake,registry,policy=make_reviewed_fixture(tmp_path)
    with pytest.raises(ValueError,match='outside'):freeze_core(intake,registry,policy,intake/'nested')
    original=(intake/'manifest.json').read_bytes()
    out=tmp_path/'frozen'
    report=freeze_core(intake,registry,policy,out)
    assert report['freeze_ready'] and not report['release_eligible']
    assert not load_dataset(out,notes_dir=out/'corpus').provisional
    assert (intake/'manifest.json').read_bytes()==original
    with pytest.raises(ValueError,match='already exists'):freeze_core(intake,registry,policy,out)
    next((intake/'pool-audits').iterdir()).write_text('{}')
    with pytest.raises(ValueError,match='refused'):freeze_core(intake,registry,policy,tmp_path/'bad')
    assert not (tmp_path/'bad').exists()


def test_self_review_cannot_certify_independence(tmp_path):
    intake,registry,policy=make_reviewed_fixture(tmp_path)
    registry['families'][0]['independence_review']['reviewers']=['fixture-author']
    with pytest.raises(ValueError,match='refused'):freeze_core(intake,registry,policy,tmp_path/'self-reviewed')
