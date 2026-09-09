from dataclasses import replace

from arkb.retrieval import SearchResult


def test_identity_joins_methods_but_distinguishes_documents_chunks_and_spans():
    hit = SearchResult(source_id='doc', source='a.md', content='text',
                       chunk_id='chunk', method='semantic')
    assert hit.identity == replace(hit, method='bm25', score=2., score_type='bm25').identity
    assert hit.identity != replace(hit, source_id='other-document').identity
    assert hit.identity != replace(hit, chunk_id='other-chunk').identity
    span = replace(hit, chunk_id=None, start_char=0, end_char=4)
    assert span.identity != replace(span, start_char=4, end_char=8).identity
    assert span.identity != replace(span, start_char=None, end_char=None).identity
