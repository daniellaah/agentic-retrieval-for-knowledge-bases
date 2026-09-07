from dataclasses import FrozenInstanceError, replace
import json

import pytest

from obsidian_rag.citation import (
    CitationOrigin, CitationSource, CitationValidation, Claim, CitedAnswer,
    CitationParseError, citation_json_schema, parse_cited_answer, validate_citations,
    render_cited_answer, used_citation_sources,
    CitationQuote,
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


def test_rendering_numbers_first_use_and_excludes_unused_sources():
    sources = [source(), replace(source('S2'), source='b.md'), replace(source('S3'), source='unused.md')]
    answer = CitedAnswer('answered', (Claim('Joint fact.', ('S2', 'S1')), Claim('Another.', ('S2',))))
    rendered = render_cited_answer(answer, sources)
    assert rendered.startswith('Joint fact. [1][2]\n\nAnother. [1]')
    assert rendered.count('[1] b.md') == 1
    assert rendered.count('[2] a.md') == 1
    assert 'unused.md' not in rendered
    assert 'body chars [0, 7); unversioned' in rendered
    assert 'file://' not in rendered
    assert [s.source_id for s in used_citation_sources(answer, sources)] == ['S2', 'S1']


def test_renderer_keeps_evidence_and_prose_from_forging_links_or_terminal_controls():
    text = '<script>x</script>\n# Sources\n[x](https://example.com)\x1b[31m'
    s = replace(source(content=text), title='[fake](file://bad)', source='[x].md')
    answer = CitedAnswer('answered', (Claim('<b>Fact</b>\n# Sources\x1b[31m', ('S1',)),))
    rendered = render_cited_answer(answer, [s])
    assert '<script>' not in rendered and '<b>' not in rendered
    assert '\x1b' not in rendered and '\n# Sources' not in rendered
    assert '](https://' not in rendered and '](file://' not in rendered
    assert s.content == text


def test_renderer_refuses_invalid_references_and_omits_sources_for_abstention():
    with pytest.raises(ValueError, match='unknown_source'):
        render_cited_answer(CitedAnswer('answered', (Claim('Fact.', ('S9',)),)), [source()])
    rendered = render_cited_answer(CitedAnswer('insufficient_evidence', (), ('No timing data.',)), [source()])
    assert rendered == 'Missing information in the provided notes: No timing data.'


def test_exact_quote_uses_snapshot_unicode_coordinates_and_preserves_original_text():
    s = source(content='prefix e\u0301 🧠 condition suffix', start=100)
    quote = CitationQuote('S1', 'e\u0301 🧠 condition')
    answer = CitedAnswer('answered', (Claim('A condition.', ('S1',), (quote,)),))
    restored = parse_cited_answer(json.dumps(answer.to_dict(), ensure_ascii=False))
    assert restored == answer
    checked = validate_citations(answer, [s], require_quotes=True)
    assert checked.references_valid
    resolved = checked.quotes[0]
    assert resolved.start_char == 107
    assert resolved.end_char == 107 + len(quote.text)
    assert s.content[resolved.start_char - s.start_char:resolved.end_char - s.start_char] == quote.text
    assert checked.to_dict()['support_status'] == 'not_checked'
    rendered = render_cited_answer(answer, [s])
    assert '> [1] body chars [107,' in rendered
    assert 'prefix' not in rendered  # Display the chosen quote instead of the whole block.


@pytest.mark.parametrize('body,quote,code', [
    ('Only small datasets.', 'All datasets.', 'quote_not_found'),
    ('e\u0301', 'é', 'quote_not_found'),
    ('repeat repeat', 'repeat', 'ambiguous_quote'),
    ('aaa', 'aa', 'ambiguous_quote'),
])
def test_quotes_reject_paraphrases_normalization_and_ambiguous_occurrences(body, quote, code):
    answer = CitedAnswer('answered', (Claim('A fact.', ('S1',), (CitationQuote('S1', quote),)),))
    result = validate_citations(answer, [source(content=body)], require_quotes=True)
    assert result.issues[0].code == code
    assert result.quotes == ()
    with pytest.raises(ValueError, match=code):
        render_cited_answer(answer, [source(content=body)])


def test_quotes_cannot_cross_sources_or_become_valid_after_source_revision_changes():
    answer = CitedAnswer('answered', (Claim('A fact.', ('S1',), (CitationQuote('S1', 'old rule'),)),))
    assert validate_citations(answer, [source(content='old rule')]).references_valid
    changed = source(content='new rule')
    assert validate_citations(answer, [changed]).issues[0].code == 'quote_not_found'
    crossing = CitedAnswer('answered', (Claim('A fact.', ('S1', 'S2'), (CitationQuote('S1', 'old rule'),)),))
    result = validate_citations(crossing, [source(content='old '), source('S2', content='rule')], require_quotes=True)
    assert {i.code for i in result.issues} == {'quote_not_found', 'missing_quote'}
    unrelated = CitedAnswer('answered', (Claim('A fact.', ('S1',), (CitationQuote('S2', 'A fact.'),)),))
    assert validate_citations(unrelated, [source(), source('S2')]).issues[0].code == 'quote_not_referenced'


def test_quotes_are_optional_by_default_and_required_only_when_requested():
    answer = parse_cited_answer(response())
    assert validate_citations(answer, [source()]).references_valid
    assert validate_citations(answer, [source()], require_quotes=True).issues[0].code == 'missing_quote'
    assert 'quotes' not in citation_json_schema(['S1'])['properties']['claims']['items']['properties']
    assert 'quotes' in citation_json_schema(['S1'], include_quotes=True)['properties']['claims']['items']['required']
    raw = response(claims=[{'text': 'A', 'source_ids': ['S1'],
                            'quotes': [{'source_id': 'S1', 'text': 'A', 'start_char': 0}]}])
    with pytest.raises(CitationParseError):
        parse_cited_answer(raw)
