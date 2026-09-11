import json
import pytest
from arkb.evaluation.regression import freshness_probe, reserve_rotation, reserve_core_release


def test_freshness_regression(tmp_path):
    result=freshness_probe(tmp_path/'corpus')
    assert result['passed']==12
    assert result['failed']==0


def reserve(path,**kwargs):
    args=dict(release_id='r1',dataset_id='d1',groups=['family1'],queries=['A question'],reviewed=True,exposure='private-unseen')
    return reserve_rotation(path,**{**args,**kwargs})


@pytest.mark.parametrize('change',[{'dataset_id':'d1'},{'groups':['family1']},{'queries':[' A  QUESTION ']},{'release_id':'r1'}])
def test_holdout_identity_cannot_be_recycled(tmp_path,change):
    path=tmp_path/'rotation.jsonl';reserve(path)
    with pytest.raises(ValueError,match='exposed'):
        reserve(path,**(dict(release_id='r2',dataset_id='d2',groups=['family2'],queries=['Another'])|change))
    assert len(path.read_text().splitlines())==1


def test_new_holdout_and_tamper_detection(tmp_path):
    path=tmp_path/'rotation.jsonl';first=reserve(path)
    second=reserve(path,release_id='r2',dataset_id='d2',groups=['family2'],queries=['Another'])
    assert second['previous']==first['sha256']
    path.write_text(path.read_text().replace('family1','changed'))
    with pytest.raises(ValueError,match='hash chain'):
        reserve(path,release_id='r3',dataset_id='d3',groups=['family3'],queries=['Third'])


@pytest.mark.parametrize('change',[{'reviewed':False},{'exposure':'public-development'},{'queries':[]},{'groups':[]}])
def test_unreviewed_public_or_empty_holdout_rejected(tmp_path,change):
    with pytest.raises(ValueError):reserve(tmp_path/'rotation.jsonl',**change)


def test_current_core_is_not_a_reviewed_release_holdout(tmp_path):
    from pathlib import Path
    with pytest.raises(ValueError):
        reserve_core_release(Path('evaluation/data/v2/pilot'),tmp_path/'ledger.jsonl','release')
    assert not (tmp_path/'ledger.jsonl').exists()
