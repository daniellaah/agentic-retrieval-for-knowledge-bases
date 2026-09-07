from dataclasses import FrozenInstanceError, replace
import json

import pytest

from obsidian_rag.citation import (
    CitationOrigin, CitationSource, CitationValidation, Claim, CitedAnswer,
)


def source(source_id='S1', *, content='A fact.', start=0):
    return CitationSource(source_id, 'a.md', 'Title', content, start, start + len(content),
                          (CitationOrigin(start, start + len(content), .8),))


def test_contract_preserves_unicode_coordinates_and_unknown_legacy_identity():
    s = source(content='条件 e\u0301 🧠', start=7)
    assert s.end_char == 7 + len(s.content)
    assert s.origins[0].document_revision is None
    with pytest.raises(FrozenInstanceError):
        s.content = 'changed'
    with pytest.raises(ValueError, match='length'):
        replace(s, end_char=s.end_char + 1)


@pytest.mark.parametrize('changes', [
    {'source_id': '1'}, {'source_id': 'S0'}, {'source_id': True},
    {'origins': []}, {'origins': ()}, {'content': ' '}, {'start_char': True},
    {'origins': (CitationOrigin(30, 40, .2),)},
])
def test_source_rejects_unusable_identity_or_span(changes):
    with pytest.raises(ValueError):
        replace(source(), **changes)


def test_merged_source_requires_known_matching_versions():
    a = CitationOrigin(0, 4, .8, 'chunk-a', 'doc', 'rev', 'vault', 'v1')
    b = replace(a, start_char=2, end_char=7, chunk_id='chunk-b')
    merged = replace(source(content='abcdefg'), origins=(a, b))
    assert len(merged.origins) == 2
    for bad in (replace(b, document_revision='other'), CitationOrigin(2, 7, .7)):
        with pytest.raises(ValueError, match='one known'):
            replace(merged, origins=(a, bad))


@pytest.mark.parametrize('status,claims,missing', [
    ('answered', (), ()), ('answered', (Claim('A', ('S1',)),), ('B',)),
    ('partial', (Claim('A', ('S1',)),), ()),
    ('insufficient_evidence', (Claim('A', ('S1',)),), ('B',)),
    ('insufficient_evidence', (), ()), ('unknown', (), ('B',)),
])
def test_contradictory_answer_states_are_rejected(status, claims, missing):
    with pytest.raises(ValueError):
        CitedAnswer(status, claims, missing)


def test_serialization_separates_facts_missing_information_and_validation():
    answer = CitedAnswer('partial', (Claim('条件。', ('S2', 'S1')),), ('缺少延迟数据。',))
    assert json.loads(json.dumps(answer.to_dict())) == {
        'status': 'partial', 'claims': [{'text': '条件。', 'source_ids': ['S2', 'S1']}],
        'missing_information': ['缺少延迟数据。'],
    }
    assert CitationValidation().to_dict()['support_status'] == 'not_checked'
    assert Claim('Uncited fact', ()).source_ids == ()
    with pytest.raises(ValueError, match='Duplicate'):
        Claim('A', ('S1', 'S1'))
    with pytest.raises(ValueError, match='immutable'):
        Claim('A', ['S1'])
