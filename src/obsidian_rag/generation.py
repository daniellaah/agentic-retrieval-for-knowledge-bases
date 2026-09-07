"""Generate answers from retrieved note chunks with Ollama."""

from ollama import Client

from obsidian_rag.context import build_context
from obsidian_rag.retrieval import SearchResult


def generate_answer(
    question: str,
    results: list[SearchResult],
    *,
    client: Client,
    model: str = "qwen3.5:4b",
) -> str:
    """Return an answer based on the question and retrieved notes.

    Send chunk titles, verbatim content, and source filenames in retrieval order.
    Do not expand a retrieved chunk back to its full note. Request
    citations and an explicit admission when the notes lack the requested facts.
    These are model instructions; generated claims and citations are not verified
    by this function.

    The caller controls the Ollama host and timeout through the supplied client.
    Empty results return an insufficient-information message without calling
    Ollama. Raise ValueError for a blank question or an empty model response.
    Ollama and connection errors propagate to the caller.
    """
    context = build_context(question, results)
    if not context.has_evidence:
        return "The provided notes do not contain enough information to answer this question."

    response = client.chat(
        model=model,
        messages=context.messages,
        stream=False,
        think=False,
        options={"temperature": 0},
    )
    answer = response.message.content
    if answer is None or not answer.strip():
        raise ValueError("Ollama returned an empty answer.")

    return answer.strip()
