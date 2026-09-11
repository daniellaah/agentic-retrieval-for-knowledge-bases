import hashlib
import importlib.util
import io
from pathlib import Path
import pytest


def test_download_integrity_failure_preserves_prior_file(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parents[2]/'evaluation/experiments/fetch_p4.py'
    spec=importlib.util.spec_from_file_location('p4_fetch_under_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:io.BytesIO(b'wrong'))
    monkeypatch.setattr(module.time,'sleep',lambda _:None)
    target=tmp_path/'input';target.write_bytes(b'prior-valid-data')
    expected={'url':'https://example.invalid/fixture','sha256':hashlib.sha256(b'right').hexdigest(),'bytes':5}
    with pytest.raises(ValueError,match='registered'):module.fetch(expected['url'],target,expected)
    assert target.read_bytes()==b'prior-valid-data'
    monkeypatch.setattr(module,'urlopen',lambda *a,**k:io.BytesIO(b'right'))
    assert module.fetch(expected['url'],target,expected)==expected
    assert target.read_bytes()==b'right'
