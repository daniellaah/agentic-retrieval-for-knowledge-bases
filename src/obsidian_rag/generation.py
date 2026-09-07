"""Generate grounded answers from the context builder's final messages."""

from ollama import Client

from obsidian_rag.context import (
    BuiltContext, ContextBudgetError, ContextConfig, GenerationCounter,
    build_context, load_generation_counter,
)
from obsidian_rag.retrieval import SearchResult


_NO_EVIDENCE = "The provided notes do not contain enough information to answer this question."


def generate_answer(
    question: str | BuiltContext,
    results: list[SearchResult] | None = None,
    *,
    client: Client,
    model: str | None = None,
    config: ContextConfig | None = None,
    counter: GenerationCounter | None = None,
) -> str:
    """Send a budgeted BuiltContext unchanged, with matching runtime limits.

    The legacy (question, results) entry point remains available and now builds
    a budgeted context. Its default counter supports the pinned Qwen3.5 artifact;
    supply a GenerationCounter for another model. Explicit BuiltContext inputs
    must already have a budget; model/config/counter mismatches are rejected.

    No-evidence returns without a model call. Evidence excluded entirely by the
    budget raises ContextBudgetError, distinct from insufficient source material.
    Source citations are requested but claim support is not automatically judged.
    Token count drift is reported when Ollama provides prompt_eval_count.
    """
    if isinstance(question, BuiltContext):
        if results is not None or config is not None or counter is not None:
            raise ValueError('Do not override a BuiltContext with results, config or counter.')
        context = question
    else:
        if results is None:
            raise ValueError('Retrieved results are required with a question.')
        # Validate evidence and handle missing evidence before contacting a model.
        context = build_context(question, results)
        if context.has_evidence:
            selected_model = model or (counter.model if counter else 'qwen3.5:4b')
            counter = counter or load_generation_counter(client=client, model=selected_model)
            context = build_context(question, results, config=config or ContextConfig(), counter=counter)
    if context.status == 'budget_exhausted':
        raise ContextBudgetError('No evidence fits the context budget; increase the window or reduce the output reserve.')
    if not context.has_evidence:
        return _NO_EVIDENCE
    if context.config is None or context.counter is None or context.prompt_tokens is None:
        raise ValueError('Generation requires a budgeted BuiltContext with a message counter.')
    selected_model = model or context.counter.model
    if selected_model != context.counter.model:
        raise ValueError('Generation model differs from the context token counter.')
    messages = context.messages
    if (context.counter(messages) != context.prompt_tokens
            or context.prompt_tokens > context.config.input_budget):
        raise ContextBudgetError('Final messages no longer match the measured context budget.')
    response = client.chat(
        model=selected_model, messages=messages, stream=False, think=False,
        options={'temperature': 0, 'num_ctx': context.config.context_window,
                 'num_predict': context.config.max_output_tokens},
    )
    actual = response.prompt_eval_count
    if actual is not None:
        if actual > context.config.input_budget:
            raise ContextBudgetError('Ollama input count exceeds the context budget.')
        if not context.counter.is_estimate and actual != context.prompt_tokens:
            raise ValueError('Ollama input count differs from the context counter; revalidate model/template compatibility.')
    answer = response.message.content
    if answer is None or not answer.strip():
        raise ValueError('Ollama returned an empty answer.')
    return answer.strip()
