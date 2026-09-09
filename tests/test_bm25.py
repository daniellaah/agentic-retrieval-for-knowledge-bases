import pytest

from arkb.chunking import whole_note_chunks
from arkb.schema import ChunkRecord, Note


def records(*texts):
    notes = [Note('', text, f'{i}.md') for i, text in enumerate(texts)]
    return [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]


def test_bm25_scores_rare_terms_and_preserves_snapshot_evidence():
    from arkb.retrieval.bm25 import BM25Retriever
    corpus = records('common common', 'common rare', 'common word')
    result = BM25Retriever(corpus, index_id='snapshot').search('RARE', top_k=3)
    assert [r.source for r in result.results] == ['1.md']
    hit = result.results[0]
    # N=3, df=1, tf=1, length=average: log(1 + 2.5/1.5).
    assert hit.score == pytest.approx(0.9808292530117263)
    assert hit.score_type == 'bm25'
    assert hit.chunk_id == corpus[1].chunk_id
    assert hit.source_id == corpus[1].document_id
    assert hit.content == 'common rare'
    assert hit.metadata['index_version'] == result.index_id == 'snapshot'


def test_frequency_length_normalization_filtering_and_exact_identifier_tokens():
    from arkb.retrieval.bm25 import BM25Retriever
    corpus = records('ERR_RESET', 'ERR_RESET ERR_RESET', 'ERR_RESET ' + 'padding ' * 30,
                     'err reset', 'Café')
    retriever = BM25Retriever(corpus)
    assert [h.source for h in retriever.search('err_reset', top_k=10).results] == ['1.md', '0.md', '2.md']
    assert [h.source for h in retriever.search('err_reset', top_k=1).results] == ['1.md']
    assert [h.source for h in retriever.search('err_reset', top_k=1, filters={'source': '2.md'}).results] == ['2.md']
    assert retriever.search('Cafe\u0301').results[0].source == '4.md'
    assert retriever.search('unknown').results == retriever.search('!!!').results == ()
    assert BM25Retriever([]).search('term').results == ()
    assert retriever.search('err_reset') == retriever.search('err_reset')
    assert retriever.search('err_reset').results == retriever.search('err_reset err_reset').results


def test_ties_are_stable_across_corpus_order_and_invalid_inputs_fail():
    from arkb.retrieval.bm25 import BM25Retriever
    corpus = records('same', 'same', 'same')
    expected = sorted((r.document_id, r.chunk_id) for r in corpus)[:2]
    for ordered in (corpus, list(reversed(corpus))):
        assert [(h.source_id, h.chunk_id) for h in BM25Retriever(ordered).search('same').results] == expected
    for config in ({'k1': 0}, {'k1': float('nan')}, {'b': 2}, {'b': True}):
        with pytest.raises(ValueError):
            BM25Retriever(corpus, **config)
    with pytest.raises(ValueError, match='Duplicate'):
        BM25Retriever([corpus[0], corpus[0]])
    for options in ({'top_k': 0}, {'filters': {'unknown': 'x'}}):
        with pytest.raises(ValueError):
            BM25Retriever(corpus).search('same', **options)


def test_sqlite_baseline_uses_published_semantic_chunks_without_models_or_vector_reads(tmp_path, qdrant, qdrant_config):
    from unittest.mock import Mock
    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models
    from arkb.indexing import build_index
    from arkb.retrieval.bm25 import BM25Retriever
    from arkb.retrieval.qdrant import snapshot_result
    from arkb.schema import EmbeddingSpec
    from arkb.storage import SQLiteStorage
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2, document_template='title-body-v1')
    path = tmp_path / 'index.sqlite'
    with SQLiteStorage(path) as storage:
        report = build_index(storage, [Note('Title', 'rare identifier', 'a.md')], spec=spec,
                             vault_id='v', client=client, tokenizer=tokenizer, max_input_tokens=100,
                             chunking='none', qdrant_client=qdrant, qdrant_config=qdrant_config)
    before = path.read_bytes()
    client.reset_mock()
    with SQLiteStorage(path, read_only=True) as storage:
        storage.get_embedding = Mock(side_effect=AssertionError('No vectors for BM25'))
        lexical = BM25Retriever.from_snapshot(storage, vault_id='v')
        record = storage.snapshot_records(lexical.index_id)[0]
        semantic = snapshot_result(record, .8, lexical.index_id)
    hit = lexical.search('identifier').results[0]
    assert (hit.identity, hit.content, hit.start_char, hit.end_char) == (
        semantic.identity, semantic.content, semantic.start_char, semantic.end_char)
    assert hit.metadata['document_revision'] == semantic.metadata['document_revision']
    client.embed.assert_not_called()
    assert path.read_bytes() == before
    import json
    from arkb.retrieval_evaluation import main
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'identifier', 'question': 'identifier', 'relevance': {'a.md': 1}}))
    output = tmp_path / 'report.json'
    argv = ['--db', str(path), '--vault-id', 'v', '--modes', 'bm25',
            '--cases', str(cases), '--output', str(output)]
    assert main(argv) == 0
    report = json.loads(output.read_text())
    assert report['summary']['bm25']['mrr'] == report['summary']['bm25']['recall_at_k'] == 1
    assert report['run']['manifest']['index_version'] == lexical.index_id
    with pytest.raises(SystemExit):
        main(argv)
