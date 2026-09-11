import math
import json

import pytest

from arkb.evaluation.external import (
    ExternalDataset, aspect_metrics, load_external, opaque_source, rank_metrics,
    reference_metrics, sample_ids, score_ranking, verify_checksums, digest, write_json, import_public,
)
from arkb.knowledge.documents import load_notes


def dataset():
    return ExternalDataset('fixture',[{'id':'../../q1/gold.txt','title':'Heading','text':'  alpha\n# nested\nβ  '},
        {'id':'b','title':'','text':'negative'}],{'q':'alpha'},{'q':{'../../q1/gold.txt':1,'b':0}},{},{'scope':'fixture'})


def test_external_roundtrip_keeps_labels_outside_notes(tmp_path):
    d=dataset(); d.save(tmp_path/'d',materialize=True)
    restored=load_external(tmp_path/'d')
    assert restored==d
    notes={n.source:n for n in load_notes(tmp_path/'d/corpus')}
    assert notes[opaque_source(d.name,d.corpus[0]['id'])].content=='alpha\n# nested\nβ'
    assert notes[opaque_source(d.name,'b')].title==''
    assert all('/' not in name and len(name)==67 for name in notes)
    assert len(notes)==2
    assert all('gold.txt' not in p.read_text() for p in (tmp_path/'d/corpus').iterdir())
    d.verify_materialized(tmp_path/'d/corpus')
    (tmp_path/'d/corpus'/opaque_source(d.name,'b')).write_text('changed')
    with pytest.raises(ValueError,match='text changed'):d.verify_materialized(tmp_path/'d/corpus')


def test_manifest_tampering_and_escape_rejected(tmp_path):
    dataset().save(tmp_path/'d')
    p=tmp_path/'d/queries.json';p.write_text('{}')
    with pytest.raises(ValueError,match='checksum'):load_external(tmp_path/'d')
    m=tmp_path/'d/manifest.json'; data=json.loads(m.read_text())
    data['files']['../queries.json']='0'*64;m.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='manifest'):load_external(tmp_path/'d')


def test_binary_one_is_positive_and_gain_is_linear():
    m=rank_metrics({'a':1,'b':3},['x','a','b'],ks=(1,2,3))
    assert m['recall@2']==.5
    assert m['mrr@1']==0
    assert m['mrr@2']==.5
    assert m['ndcg@3']==pytest.approx((1/math.log2(3)+3/2)/(3+1/math.log2(3)))
    assert rank_metrics({},[],ks=(10,))['recall@10'] is None


def test_reference_covers_empty_and_missing_runs():
    pytest.importorskip('pytrec_eval')
    qrels={'q':{'a':1,'b':2},'empty':{'x':1}}
    rankings={'q':['n','a','b'],'empty':[]}
    refs=reference_metrics(qrels,rankings)
    for q in qrels:
        expected=rank_metrics(qrels[q],rankings[q])
        assert refs[q]==pytest.approx({k:v for k,v in expected.items() if not k.startswith('mrr')})
    with pytest.raises(ValueError):reference_metrics(qrels,{'q':[]})


def test_duplicates_unknown_documents_and_missing_gold_rejected():
    with pytest.raises(ValueError):rank_metrics({'a':1},['a','a'])
    with pytest.raises(ValueError):score_ranking(dataset(),'q',['missing'])
    d=dataset();d.qrels['q']['missing']=1
    with pytest.raises(ValueError):d.validate()
    d=dataset();d.corpus.append(d.corpus[0])
    with pytest.raises(ValueError):d.validate()


def test_aspect_metric_rewards_complementary_evidence():
    aspects=[{'id':'a','weight':3,'supporting_docs':['a1','a2']},
             {'id':'b','weight':1,'supporting_docs':['b1']}]
    repeat=aspect_metrics(aspects,['a1','a2'],k=2)
    diverse=aspect_metrics(aspects,['a1','b1'],k=2)
    assert repeat['aspect_recall@2']==.75
    assert diverse['aspect_recall@2']==1
    # With weights 3:1 and alpha=.5, a second A still has higher marginal
    # gain than B. Do not implement an incorrect "one of each first" ideal.
    assert repeat['alpha_ndcg@2']==pytest.approx(1)
    assert diverse['alpha_ndcg@2']<1
    assert aspect_metrics(aspects,[],k=2)['alpha_ndcg@2']==0


def test_nonexclusive_or_incomplete_aspects_rejected():
    d=dataset();d.aspects={'q':[{'id':'a','weight':1,'supporting_docs':['b']}]}
    with pytest.raises(ValueError):d.validate()
    d.aspects={'q':[{'id':'a','weight':float('nan'),'supporting_docs':['../../q1/gold.txt']}]}
    with pytest.raises(ValueError):d.validate()


def test_sampling_order_independent_and_does_not_expand_size():
    assert sample_ids(['1','2','3'],2)==sample_ids(['3','2','1'],2)
    with pytest.raises(ValueError):sample_ids(['1'],2)
    with pytest.raises(ValueError):sample_ids(['1','1'],1)


def test_run_checksum_rejects_changed_rows_and_external_paths(tmp_path):
    p=tmp_path/'rows.jsonl';p.write_text('[]')
    write_json(tmp_path/'checksums.json',{'rows.jsonl':digest(p)})
    assert verify_checksums(tmp_path)==1
    p.write_text('{}')
    with pytest.raises(ValueError):verify_checksums(tmp_path)
    write_json(tmp_path/'checksums.json',{'../outside':'0'*64})
    with pytest.raises(ValueError):verify_checksums(tmp_path)


def test_unverified_downloads_are_never_imported(tmp_path):
    write_json(tmp_path/'downloads.json',{'files':{}})
    with pytest.raises(ValueError,match='download hashes'):import_public(tmp_path,'scifact')


def test_invalid_json_does_not_replace_an_existing_manifest(tmp_path):
    path=tmp_path/'manifest.json';write_json(path,{'valid':True})
    before=path.read_bytes()
    with pytest.raises(ValueError):write_json(path,{'invalid':float('nan')})
    assert path.read_bytes()==before
