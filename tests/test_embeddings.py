from unittest.mock import Mock

import numpy as np
import pytest
from ollama import Client, EmbedResponse, ResponseError

from obsidian_rag.embeddings import embed_texts


@pytest.fixture
def client() -> Mock:
    return Mock(spec=Client)


def test_embed_texts_returns_vectors_in_input_order(client: Mock) -> None:
    texts = ["Capture a passing thought.", "Connect related ideas."]
    client.embed.return_value = EmbedResponse(
        embeddings=[[0.6, 0.8], [-0.8, 0.6]]
    )

    vectors = embed_texts(texts, client=client)

    assert isinstance(vectors, np.ndarray)
    assert np.issubdtype(vectors.dtype, np.floating)
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [-0.8, 0.6]])
    client.embed.assert_called_once_with(
        model="qwen3-embedding:0.6b", input=texts, truncate=False
    )


def test_embed_texts_supports_a_selected_model_and_its_vector_dimension(
    client: Mock,
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.6, 0.8]])

    vectors = embed_texts(
        ["Preserve the source."], client=client, model="qwen3-embedding:4b"
    )

    assert vectors.shape == (1, 3)
    assert client.embed.call_args.kwargs["model"] == "qwen3-embedding:4b"


def test_embed_texts_returns_an_empty_matrix_without_calling_ollama(
    client: Mock,
) -> None:
    vectors = embed_texts([], client=client)

    assert isinstance(vectors, np.ndarray)
    assert vectors.shape == (0, 0)
    client.embed.assert_not_called()


@pytest.mark.parametrize("blank_text", ["", " \n\t"])
def test_embed_texts_rejects_blank_text_before_calling_ollama(
    client: Mock, blank_text: str
) -> None:
    with pytest.raises(ValueError, match="blank"):
        embed_texts(["A useful idea.", blank_text], client=client)

    client.embed.assert_not_called()


@pytest.mark.parametrize(
    "embeddings",
    [
        [[0.6, 0.8]],
        [[0.6, 0.8], [1.0]],
        [[], []],
    ],
    ids=["missing-vector", "inconsistent-dimensions", "empty-vectors"],
)
def test_embed_texts_rejects_invalid_matrix_shapes(
    client: Mock, embeddings: list[list[float]]
) -> None:
    client.embed.return_value = EmbedResponse(embeddings=embeddings)

    with pytest.raises(ValueError):
        embed_texts(["First idea.", "Second idea."], client=client)


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf")])
def test_embed_texts_rejects_non_finite_values(
    client: Mock, invalid_value: float
) -> None:
    client.embed.return_value = EmbedResponse(
        embeddings=[[invalid_value, 0.8]]
    )

    with pytest.raises(ValueError, match="finite"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_rejects_zero_vectors(client: Mock) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero"):
        embed_texts(["A useful idea."], client=client)


def test_embed_texts_propagates_model_errors(client: Mock) -> None:
    client.embed.side_effect = ResponseError("Model not found.", status_code=404)

    with pytest.raises(ResponseError) as error:
        embed_texts(["A useful idea."], client=client)

    assert error.value.status_code == 404


def test_embed_texts_propagates_connection_errors(client: Mock) -> None:
    client.embed.side_effect = ConnectionError("Ollama is unavailable.")

    with pytest.raises(ConnectionError, match="Ollama is unavailable"):
        embed_texts(["A useful idea."], client=client)


def test_batches_obey_both_limits_and_preserve_global_order(client: Mock) -> None:
    client.embed.side_effect = lambda **kw: EmbedResponse(
        embeddings=[[float(text), 1.0] for text in kw["input"]])
    vectors = embed_texts(["1", "2", "3", "4", "5"], client=client,
                         batch_size=2, token_counts=[2, 3, 4, 2, 1], max_batch_tokens=5)
    assert [call.kwargs["input"] for call in client.embed.call_args_list] == [
        ["1", "2"], ["3"], ["4", "5"],
    ]
    np.testing.assert_array_equal(vectors[:, 0], [1, 2, 3, 4, 5])


@pytest.mark.parametrize("options", [
    {"batch_size": 0}, {"batch_size": True}, {"dimensions": -1},
    {"max_batch_tokens": 10}, {"token_counts": [1]},
    {"token_counts": [1, True]}, {"max_batch_tokens": 2, "token_counts": [1, 3]},
    {"max_retries": -1}, {"max_retries": 9}, {"retry_delay": float("nan")},
    {"dtype": "int8"}, {"normalization": "unknown"},
])
def test_invalid_plan_fails_before_first_batch(client: Mock, options) -> None:
    with pytest.raises(ValueError):
        embed_texts(["first", "second"], client=client, **options)
    client.embed.assert_not_called()


def test_successful_batches_can_be_checkpointed_before_a_later_failure(client: Mock) -> None:
    from obsidian_rag.embeddings import iter_embedding_batches
    client.embed.side_effect = [EmbedResponse(embeddings=[[1, 0]]), ConnectionError("failed")]
    batches = iter_embedding_batches(["a", "b"], client=client, batch_size=1)
    start, matrix = next(batches)
    assert start == 0
    np.testing.assert_array_equal(matrix, [[1, 0]])
    with pytest.raises(ConnectionError):
        next(batches)


def test_dimension_change_between_batches_is_rejected(client: Mock) -> None:
    client.embed.side_effect = [EmbedResponse(embeddings=[[1, 0]]),
                               EmbedResponse(embeddings=[[1, 0, 0]])]
    with pytest.raises(ValueError, match="dimensions"):
        embed_texts(["a", "b"], client=client, batch_size=1)


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_errors_are_retried_with_bounded_backoff(client: Mock, monkeypatch, status) -> None:
    sleep = Mock()
    monkeypatch.setattr("obsidian_rag.embeddings.time.sleep", sleep)
    client.embed.side_effect = [ResponseError("busy", status_code=status),
                               EmbedResponse(embeddings=[[1, 0]])]
    embed_texts(["a"], client=client, max_retries=2)
    assert client.embed.call_count == 2
    sleep.assert_called_once_with(0.25)


def test_transport_retries_stop_and_permanent_errors_are_not_retried(client: Mock) -> None:
    client.embed.side_effect = ConnectionError("offline")
    with pytest.raises(ConnectionError):
        embed_texts(["a"], client=client, max_retries=2, retry_delay=0)
    assert client.embed.call_count == 3
    client.reset_mock()
    client.embed.side_effect = ResponseError("bad input", status_code=400)
    with pytest.raises(ResponseError):
        embed_texts(["a"], client=client, max_retries=2, retry_delay=0)
    assert client.embed.call_count == 1


def test_representation_matches_declared_spec(client: Mock) -> None:
    client.embed.return_value = EmbedResponse(embeddings=[[0.6, 0.8]])
    assert embed_texts(["a"], client=client, dimensions=2,
                       dtype="float32", normalization="l2").dtype == np.float32
    client.embed.return_value = EmbedResponse(embeddings=[[3, 4]])
    with pytest.raises(ValueError, match="L2"):
        embed_texts(["a"], client=client, normalization="l2")
