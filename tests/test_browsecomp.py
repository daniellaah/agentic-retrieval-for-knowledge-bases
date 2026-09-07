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


def test_local_corpus_preserves_text_and_maps_opaque_ids_without_path_traversal(tmp_path):
    import json
    from obsidian_rag.browsecomp import read_corpus, document_note, source_docid
    path = tmp_path / 'corpus.jsonl'
    path.write_text(json.dumps({'docid': '../001/雪', 'text': ' \r\n# H\ne\u0301\n ',
                                'url': 'https://example.org'}) + '\n')
    doc, = read_corpus(path)
    note = document_note(doc)
    assert note.content == ' \r\n# H\ne\u0301\n '
    assert '..' not in note.source
    assert source_docid(note.source) == '../001/雪'
    with path.open('a') as stream:
        stream.write(path.read_text())
    with pytest.raises(ValueError, match='duplicate'):
        list(read_corpus(path))


def test_official_xor_fixture_and_pinned_download_can_be_read_offline(tmp_path):
    from obsidian_rag.browsecomp import download_browsecomp, read_cases, read_corpus
    encrypted = 'SE0Qn+E='
    def dataset_loader(name, *, split, revision, streaming):
        assert revision == 'a' * 40 and streaming is True
        if name.endswith('-corpus'):
            assert split == 'train'
            return [{'docid': 'hello', 'text': 'Body\r\n', 'url': 'https://example.org'}]
        assert split == 'test'
        doc = {'docid': encrypted, 'text': encrypted, 'url': encrypted}
        return [{'query_id': '007', 'query': encrypted, 'answer': encrypted,
                 'evidence_docs': [doc], 'gold_docs': [doc], 'negative_docs': []}]
    out = tmp_path / 'download'
    download_browsecomp(out, query_revision='a' * 40, corpus_revision='a' * 40,
                       dataset_loader=dataset_loader)
    questions, labels = read_cases(out / 'cases.jsonl')
    assert asdict(questions[0]) == {'query_id': '007', 'question': 'hello'}
    assert labels[0].evidence_docids == ('hello',)
    assert next(read_corpus(out / 'corpus.jsonl')).text == 'Body\r\n'
    with pytest.raises(FileExistsError):
        download_browsecomp(out, query_revision='a' * 40, corpus_revision='a' * 40,
                           dataset_loader=dataset_loader)
    with pytest.raises(ValueError, match='revision'):
        download_browsecomp(tmp_path / 'mutable', query_revision='main', corpus_revision='a' * 40)
