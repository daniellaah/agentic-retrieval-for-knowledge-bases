import numpy as np
import pytest
from numpy.typing import NDArray

from obsidian_rag.chunking import Chunk, chunk_notes, whole_note_chunks
from obsidian_rag.loaders import Note
from obsidian_rag.retrieval import retrieve


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


def test_indexed_search_rejects_unknown_backend_hits(published_index):
    from unittest.mock import Mock
    from obsidian_rag.retrieval import search_index
    from obsidian_rag.vector_store import VectorHit
    store, kwargs = published_index
    backend = Mock(vault_id='vault')
    backend.spec = kwargs['spec']
    backend.search.return_value = [VectorHit('orphan', 0.5)]
    with pytest.raises(ValueError, match='snapshot'):
        search_index(store, 'Question?', **kwargs, vector_store=backend)
