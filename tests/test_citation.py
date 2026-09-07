from dataclasses import FrozenInstanceError, replace
import json

import pytest

from obsidian_rag.citation import (
    CitationOrigin, CitationSource, CitationValidation, Claim, CitedAnswer,
    CitationParseError, citation_json_schema, parse_cited_answer, validate_citations,
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


def response(**changes):
    return json.dumps({'status': 'answered', 'claims': [{'text': 'A fact.', 'source_ids': ['S1']}],
                       'missing_information': [], **changes}, ensure_ascii=False)


@pytest.mark.parametrize('raw', [
    '', '```json\n{}\n```', '{} trailing', '{"status":', '[]',
    response(extra='field'), response(claims={}), response(missing_information=''),
    response(claims=[{'text': 'A', 'source_ids': 'S1'}]),
    response(claims=[{'text': True, 'source_ids': ['S1']}]),
    response(claims=[{'text': 'A', 'source_ids': [1]}]),
    response(claims=[{'text': 'A', 'source_ids': ['S1', 'S1']}]),
    response(claims=[{'text': 'A', 'source_ids': ['S1'], 'url': 'file://a.md'}]),
    response().replace('"status": "answered"', '"status":"partial","status":"answered"'),
    response().replace('"A fact."', 'NaN'),
])
def test_parser_rejects_malformed_or_ambiguous_output_without_repair(raw):
    with pytest.raises(CitationParseError) as error:
        parse_cited_answer(raw)
    assert error.value.raw_response == raw


def test_validation_distinguishes_structure_references_and_support():
    raw = response(claims=[{'text': 'A', 'source_ids': []},
                           {'text': 'B', 'source_ids': ['S1', 'S9']}])
    answer = parse_cited_answer(raw)
    result = validate_citations(answer, [source()])
    assert [(i.code, i.claim_index, i.source_id) for i in result.issues] == [
        ('missing_reference', 0, None), ('unknown_source', 1, 'S9')]
    assert result.to_dict()['structure_valid'] is True
    assert result.references_valid is False
    supported_format = CitedAnswer('answered', (Claim('Joint claim', ('S2', 'S1')),))
    checked = validate_citations(supported_format, [source(), source('S2')])
    assert checked.references_valid
    assert checked.to_dict()['support_status'] == 'not_checked'


@pytest.mark.parametrize('text', ['A [1]', 'A [S1]', 'A [a.md]', '[A](file://invented)', 'file://a.md'])
def test_model_cannot_smuggle_source_markers_or_links_in_prose(text):
    answer = CitedAnswer('answered', (Claim(text, ('S1',)),))
    assert validate_citations(answer, [source()]).issues[0].code == 'inline_reference'


def test_registry_and_schema_reject_duplicate_identity_and_schema_is_fresh():
    with pytest.raises(ValueError, match='Duplicate'):
        validate_citations(parse_cited_answer(response()), [source(), source()])
    with pytest.raises(ValueError, match='Duplicate'):
        citation_json_schema(['S1', 'S1'])
    schema = citation_json_schema(['S1', 'S2'])
    assert schema['properties']['claims']['items']['properties']['source_ids']['items']['enum'] == ['S1', 'S2']
    schema['required'].clear()
    assert citation_json_schema(['S1'])['required']
