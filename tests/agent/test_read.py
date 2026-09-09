from unittest.mock import Mock

import pytest

from arkb.agent import AgentTools
from arkb.knowledge.documents import DocumentAccess
from arkb.retrieval import ExactRetriever


def test_read_full_empty_and_open_ranges(tools, documents, engine):
    record = next(documents.records(source='a.md'))
    hit = tools.read(record.document_id)['result']
    assert hit['content'] == record.chunk.content and hit['title'] == 'Alpha'
    assert hit['document_id'] == record.document_id
    assert hit['chunk_id'] is None and hit['section_id'] is None
    assert (hit['start_char'], hit['end_char']) == (0, len(hit['content']))
    assert tools.read(record.document_id, end_char=2)['result']['content'] == '中文'
    assert tools.read(record.document_id, start_char=3)['result']['content'] == hit['content'][3:]
    assert tools.read(record.document_id, start_char=2, end_char=2)['result']['content'] == ''
    empty = next(documents.records(source='empty.md'))
    assert tools.read(empty.document_id)['result']['content'] == ''
    engine.search.assert_not_called()


@pytest.mark.parametrize('options', [
    {'start_char': -1}, {'end_char': True}, {'start_char': 1.2}, {'start_char': '1'},
    {'end_char': 1000}, {'start_char': 4, 'end_char': 2}, {'section_id': ''},
    {'section_id': 'a' * 64, 'start_char': 0}, {'section_id': 'a' * 64, 'end_char': 1},
])
def test_read_invalid_inputs(tools, documents, options):
    with pytest.raises(ValueError):
        tools.read(next(documents.records()).document_id, **options)


@pytest.mark.parametrize('document_id', ['', ' ', None, True, 'not-an-id', '../a.md'])
def test_read_invalid_identifiers(tools, document_id):
    with pytest.raises(ValueError):
        tools.read(document_id)


def test_read_missing_document_section_and_deletion(tools, documents):
    with pytest.raises(LookupError):
        tools.read('0' * 64)
    record = next(documents.records())
    with pytest.raises(LookupError):
        tools.read(record.document_id, section_id='0' * 64)
    (documents.directory / record.chunk.source).unlink()
    with pytest.raises(LookupError):
        tools.read(record.document_id)


def test_read_delegates_once_and_propagates_errors(documents, engine):
    record = next(documents.records())
    access = Mock(spec=DocumentAccess)
    access.read.return_value = record
    exact = Mock(spec=ExactRetriever)
    tools = AgentTools(documents=access, exact=exact, engine=engine)
    tools.read(record.document_id, start_char=1, end_char=3)
    access.read.assert_called_once_with(record.document_id, source=None, section_id=None, start_char=1, end_char=3)
    error = PermissionError('cannot read')
    access.read.side_effect = error
    with pytest.raises(PermissionError) as raised:
        tools.read(record.document_id)
    assert raised.value is error
    engine.search.assert_not_called()
    exact.search.assert_not_called()


def test_read_known_source_directly_without_discovery(documents, engine):
    exact = Mock(spec=ExactRetriever)
    tools = AgentTools(documents=documents, exact=exact, engine=engine)
    record = next(documents.records(source='a.md'))
    assert tools.read(source='a.md') == tools.read(record.document_id)
    assert tools.read(record.document_id, source='a.md') == tools.read(record.document_id)
    assert tools.read(source='a.md', end_char=2)['result']['content'] == '中文'
    engine.search.assert_not_called()
    exact.search.assert_not_called()


@pytest.mark.parametrize('options', [
    {}, {'source': ''}, {'source': ' '}, {'source': 1}, {'source': True},
])
def test_read_requires_a_valid_document_selector(tools, options):
    with pytest.raises(ValueError):
        tools.read(**options)


def test_read_rejects_conflicting_document_selectors(tools, documents):
    record = next(documents.records(source='a.md'))
    with pytest.raises(LookupError):
        tools.read(record.document_id, source='b.md')
