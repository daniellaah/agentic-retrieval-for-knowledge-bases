import numpy as np
import pytest
from numpy.typing import NDArray
from obsidian_rag.chunking import Chunk, chunk_notes, whole_note_chunks
from obsidian_rag.loaders import Note
from obsidian_rag.retrieval import retrieve, search_qdrant, search_numpy
from contextlib import closing
from dataclasses import replace
import warnings
from qdrant_client import QdrantClient
from obsidian_rag.schema import ChunkRecord, EmbeddingSpec
from obsidian_rag.indexing import QdrantIndex


@pytest.fixture
def chunks() -> list[Chunk]:
    return whole_note_chunks([
        Note(title="Diagonal", content="A diagonal vector.", source="diagonal.md"),
        Note(title="Opposite", content="An opposite vector.", source="opposite.md"),
        Note(title="Aligned", content="An aligned vector.", source="aligned.md"),
        Note(title="Orthogonal", content="A perpendicular vector.", source="orthogonal.md"),
    ])


@pytest.fixture
def chunk_vectors() -> NDArray[np.float64]:
    return np.array([[3.0, 4.0], [-4.0, 0.0], [1.0, 0.0], [0.0, 10.0]])


def test_retrieve_ranks_chunks_by_cosine_similarity(
    chunks: list[Chunk], chunk_vectors: NDArray[np.float64]
) -> None:
    results = retrieve(chunks, chunk_vectors, np.array([2.0, 0.0]), top_k=4)

    assert [result.chunk for result in results] == [
        chunks[2], chunks[0], chunks[3], chunks[1]
    ]
    np.testing.assert_allclose([result.score for result in results], [1.0, 0.6, 0.0, -1.0])


def test_retrieve_returns_two_chunks_by_default(
    chunks: list[Chunk], chunk_vectors: NDArray[np.float64]
) -> None:
    results = retrieve(chunks, chunk_vectors, np.array([2.0, 0.0]))

    assert [result.chunk for result in results] == [chunks[2], chunks[0]]


@pytest.mark.parametrize(
    ("top_k", "expected_sources"),
    [
        (1, ["aligned.md"]),
        (8, ["aligned.md", "diagonal.md", "orthogonal.md", "opposite.md"]),
    ],
)
def test_retrieve_limits_results_to_the_requested_and_available_count(
    chunks: list[Chunk],
    chunk_vectors: NDArray[np.float64],
    top_k: int,
    expected_sources: list[str],
) -> None:
    results = retrieve(chunks, chunk_vectors, np.array([2.0, 0.0]), top_k=top_k)

    assert [result.chunk.source for result in results] == expected_sources


def test_retrieve_preserves_input_order_when_scores_are_equal(chunks: list[Chunk]) -> None:
    vectors = np.array([[1.0, 0.0], [2.0, 0.0], [4.0, 0.0], [8.0, 0.0]])

    results = retrieve(chunks, vectors, np.array([2.0, 0.0]), top_k=3)

    assert [result.chunk for result in results] == chunks[:3]


def test_retrieve_does_not_modify_input_vectors(
    chunks: list[Chunk], chunk_vectors: NDArray[np.float64]
) -> None:
    query_vector = np.array([2.0, 0.0])
    original_chunks = chunk_vectors.copy()
    original_query = query_vector.copy()

    retrieve(chunks, chunk_vectors, query_vector)

    np.testing.assert_array_equal(chunk_vectors, original_chunks)
    np.testing.assert_array_equal(query_vector, original_query)


def test_retrieve_returns_no_results_for_an_empty_collection() -> None:
    assert retrieve([], np.empty((0, 0)), np.array([1.0, 0.0])) == []


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_retrieve_rejects_invalid_result_counts(
    chunks: list[Chunk], chunk_vectors: NDArray[np.float64], top_k: int | float
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        retrieve(chunks, chunk_vectors, np.array([2.0, 0.0]), top_k=top_k)


@pytest.mark.parametrize(
    ("vectors", "query"),
    [
        ([1.0, 0.0], [1.0, 0.0]),
        ([[1.0, 0.0]], [1.0, 0.0]),
        ([[1.0, 0.0]] * 4, [[1.0, 0.0]]),
        ([[1.0, 0.0]] * 4, [1.0, 0.0, 0.0]),
        ([[], [], [], []], []),
    ],
    ids=[
        "non-matrix-documents",
        "chunk-count-mismatch",
        "non-vector-query",
        "dimension-mismatch",
        "empty-vectors",
    ],
)
def test_retrieve_rejects_incompatible_vector_shapes(
    chunks: list[Chunk], vectors: list, query: list
) -> None:
    with pytest.raises(ValueError):
        retrieve(chunks, np.array(vectors), np.array(query))


@pytest.mark.parametrize("target", ["chunks", "query"])
def test_retrieve_rejects_zero_vectors(
    chunks: list[Chunk], chunk_vectors: NDArray[np.float64], target: str
) -> None:
    query_vector = np.array([2.0, 0.0])
    if target == "chunks":
        chunk_vectors[0] = 0.0
    else:
        query_vector[:] = 0.0

    with pytest.raises(ValueError, match="nonzero"):
        retrieve(chunks, chunk_vectors, query_vector)


@pytest.mark.parametrize(
    ("target", "value"), [("chunks", float("nan")), ("query", float("inf"))]
)
def test_retrieve_rejects_non_finite_values(
    chunks: list[Chunk],
    chunk_vectors: NDArray[np.float64],
    target: str,
    value: float,
) -> None:
    query_vector = np.array([2.0, 0.0])
    if target == "chunks":
        chunk_vectors[0, 0] = value
    else:
        query_vector[0] = value

    with pytest.raises(ValueError, match="finite"):
        retrieve(chunks, chunk_vectors, query_vector)


def test_retrieve_returns_distinct_chunks_from_the_same_note() -> None:
    chunks = chunk_notes(
        [Note(title="One note", content="abcdef", source="same.md")],
        count_tokens=len, chunk_size=3, chunk_overlap=0,
    )
    vectors = np.array([[0.0, 1.0], [1.0, 0.0]])

    results = retrieve(chunks, vectors, np.array([1.0, 0.0]), top_k=2)

    assert [(r.chunk.content, r.chunk.source, r.chunk.start_char) for r in results] == [
        ("def", "same.md", 3), ("abc", "same.md", 0),
    ]
    assert results[0].chunk is chunks[1]


@pytest.fixture
def published_index(tmp_path):
    from unittest.mock import Mock
    from ollama import Client, EmbedResponse
    from tokenizers import Tokenizer, models, pre_tokenizers
    from obsidian_rag.schema import EmbeddingSpec
    from obsidian_rag.indexing import build_index
    from obsidian_rag.loaders import Note
    from obsidian_rag.storage import SQLiteStorage
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2, document_template='title-body-v1')
    client = Mock(spec=Client)
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1, 0] for _ in kw['input']])
    with SQLiteStorage(tmp_path / 'db') as store:
        build_index(store, [Note(title='Title', content='Original source text', source='a.md')],
                    spec=spec, vault_id='vault', tokenizer=tokenizer, max_input_tokens=100,
                    client=client, chunking='none', index_version='v1', query_instruction='Find evidence.')
        client.embed.reset_mock()
        yield store, dict(vault_id='vault', spec=spec, tokenizer=tokenizer, client=client)


def test_indexed_search_only_embeds_query_and_restores_snapshot_content(published_index):
    from obsidian_rag.retrieval import search_index
    store, kwargs = published_index
    results = search_index(store, 'Question?', **kwargs)
    kwargs['client'].embed.assert_called_once_with(
        model='test', input=['Instruct: Find evidence.\nQuery:Question?'], truncate=False, options={'num_ctx': 100})
    assert results[0].chunk.content == 'Original source text'
    assert results[0].chunk.source == 'a.md'
    assert results[0].score == 1


def test_indexed_search_rejects_model_and_tokenizer_mismatches_before_embedding(published_index):
    from dataclasses import replace
    from obsidian_rag.retrieval import search_index
    store, kwargs = published_index
    with pytest.raises(ValueError, match='incompatible'):
        search_index(store, 'Question?', **{**kwargs, 'spec': replace(kwargs['spec'], model_revision='changed')})
    kwargs['tokenizer'].enable_padding(length=10)
    with pytest.raises(ValueError, match='tokenizer'):
        search_index(store, 'Question?', **kwargs)
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_missing_and_unpublished_versions(published_index):
    from dataclasses import replace
    from obsidian_rag.retrieval import search_index
    store, kwargs = published_index
    with pytest.raises(ValueError, match='No published'):
        search_index(store, 'Question?', **{**kwargs, 'vault_id': 'missing'})
    building = replace(store.get_manifest('v1'), index_version='v2', status='building')
    store.create_build(building, corpus_fingerprint='pending')
    with pytest.raises(ValueError, match='ready'):
        search_index(store, 'Question?', **kwargs, index_version='v2')
    kwargs['client'].embed.assert_not_called()


def test_indexed_search_rejects_unknown_backend_hits(published_index, monkeypatch):
    from obsidian_rag.retrieval import search_index
    from obsidian_rag.schema import VectorHit
    store, kwargs = published_index
    monkeypatch.setattr('obsidian_rag.retrieval.search_numpy', lambda *a, **kw: [VectorHit('orphan', 0.5)])
    with pytest.raises(ValueError, match='snapshot'):
        search_index(store, 'Question?', **kwargs)


@pytest.fixture
def vector_data():
    spec = EmbeddingSpec(model='test', model_revision='digest', dimensions=2,
                         document_template='title-body-v1', normalization='none')
    notes = [Note(title='Title', content='body', source=f'{n}.md') for n in range(3)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='vault') for n in notes]
    return spec, records


def create_qdrant_index(client, spec):
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        return QdrantIndex(client, 'test', spec, vault_id='vault', create=True)


@pytest.mark.filterwarnings('ignore:Local mode performs exact.*:UserWarning')
def test_qdrant_local_contract_roundtrip_filter_and_delete(tmp_path, vector_data):
    spec, records = vector_data
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = create_qdrant_index(client, spec)
        store.upsert(records, [[0, 1], [3, 4], [1, 0]])
        assert store.count() == 3
        hits = search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', top_k=3, exact=True)
        assert [h.chunk_id for h in hits] == [records[2].chunk_id, records[1].chunk_id, records[0].chunk_id]
        np.testing.assert_allclose([h.score for h in hits], [1, .6, 0], atol=1e-6)
        assert search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', source='0.md')[0].chunk_id == records[0].chunk_id
    with closing(QdrantClient(path=str(tmp_path / 'qdrant'))) as client:
        store = QdrantIndex(client, 'test', spec, vault_id='vault')
        assert store.count() == 3
        store.upsert([records[2]], [[-1, 0]])
        assert store.count() == 3
        store.delete([records[0].chunk_id, records[0].chunk_id])
        assert store.count() == 2
        assert search_qdrant(client, 'test', [1, 0], spec=spec, vault_id='vault', source='0.md') == []


def test_numpy_ranking_filters_before_limit_and_preserves_sources(vector_data):
    spec, records = vector_data
    vectors = [[0, 1], [3, 4], [1, 0]]
    hits = search_numpy(records, vectors, [1, 0], spec=spec, vault_id='vault', top_k=3)
    assert [h.chunk_id for h in hits] == [records[2].chunk_id, records[1].chunk_id, records[0].chunk_id]
    np.testing.assert_allclose([h.score for h in hits], [1, .6, 0])
    filtered = search_numpy(records, vectors, [1, 0], spec=spec, vault_id='vault', top_k=1, source='0.md')
    assert filtered[0].chunk_id == records[0].chunk_id
    assert search_numpy(records, vectors, [1, 0], spec=spec, vault_id='vault', source='missing.md') == []


def test_numpy_stable_ties_and_search_does_not_mutate_inputs(vector_data):
    spec, records = vector_data
    vectors = np.array([[1., 0], [1, 0], [0, 1]])
    query = np.array([2., 0])
    original_vectors, original_query = vectors.copy(), query.copy()
    hits = search_numpy(records, vectors, query, spec=spec, vault_id='vault')
    assert [h.chunk_id for h in hits] == [r.chunk_id for r in records[:2]]
    np.testing.assert_array_equal(vectors, original_vectors)
    np.testing.assert_array_equal(query, original_query)
    assert search_numpy([], np.empty((0, 2)), query, spec=spec, vault_id='vault') == []


def test_numpy_rejects_invalid_vectors_duplicate_ids_and_foreign_vault(vector_data):
    spec, records = vector_data
    for batch, vectors in ((records[1:], [[0, 1], [0, 0]]),
                           ([records[1], records[1]], [[0, 1], [0, 1]]),
                           ([replace(records[1], vault_id='other')], [[0, 1]])):
        with pytest.raises(ValueError):
            search_numpy(batch, vectors, [1, 0], spec=spec, vault_id='vault')


@pytest.mark.parametrize('options', [{'top_k': 0}, {'top_k': True}, {'source': ''}, {'exact': 1}])
def test_empty_numpy_snapshot_still_validates_search_options(vector_data, options):
    spec, _ = vector_data
    with pytest.raises(ValueError):
        search_numpy([], np.empty((0, 2)), [1, 0], spec=spec, vault_id='vault', **options)


def test_numpy_query_dimension_and_norm_are_checked(vector_data):
    spec, _ = vector_data
    for query in ([1, 0, 0], [0, 0], [float('nan'), 0]):
        with pytest.raises(ValueError):
            search_numpy([], np.empty((0, 2)), query, spec=spec, vault_id='vault')


def test_numpy_distinct_records_can_reference_the_same_chunk_object(vector_data):
    spec, records = vector_data
    first = records[0]
    second = replace(first, document_revision='a' * 64)
    hits = search_numpy([first, second], [[1, 0], [0, 1]], [1, 0], spec=spec, vault_id='vault')
    assert [h.chunk_id for h in hits] == [first.chunk_id, second.chunk_id]
