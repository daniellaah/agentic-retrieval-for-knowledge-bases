from dataclasses import asdict

import pytest

from obsidian_rag.browsecomp import parse_case


def case_row():
    return {'query_id': '007', 'query': 'Which code?', 'answer': 'SECRET-LABEL',
            'evidence_docs': [{'docid': '001', 'text': 'Evidence.', 'url': 'https://example.org/a'}],
            'gold_docs': [{'docid': '002', 'text': 'The code.', 'url': 'https://example.org/b'}],
            'negative_docs': []}


def test_case_separates_runtime_question_from_answer_and_relevance_labels():
    question, labels = parse_case(case_row())
    assert asdict(question) == {'query_id': '007', 'question': 'Which code?'}
    assert labels.answer == 'SECRET-LABEL'
    assert labels.evidence_docids == ('001',)
    assert labels.gold_docids == ('002',)


def test_duplicate_questions_cannot_silently_overwrite_reference_answers():
    from obsidian_rag.browsecomp import parse_cases
    with pytest.raises(ValueError, match='duplicate'):
        parse_cases([case_row(), {**case_row(), 'answer': 'DIFFERENT'}])


@pytest.mark.parametrize('change', [{'query': ' '}, {'answer': None}, {'query_id': True},
                                  {'evidence_docs': [case_row()['evidence_docs'][0]] * 2}])
def test_invalid_case_contract_is_rejected(change):
    with pytest.raises(ValueError):
        parse_case({**case_row(), **change})


def test_document_ids_and_text_are_not_normalized():
    from obsidian_rag.browsecomp import parse_document
    doc = parse_document({'docid': '0001', 'text': '  e\u0301\r\n# Title\n', 'url': 'https://example.org'})
    assert doc.docid == '0001'
    assert doc.text == '  e\u0301\r\n# Title\n'
    with pytest.raises(ValueError):
        parse_document({'docid': 1, 'text': 'Text', 'url': 'https://example.org'})
