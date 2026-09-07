"""Generate grounded answers from the context builder's final messages."""

from dataclasses import asdict, dataclass

from ollama import Client

from obsidian_rag.citation import (
    CitedAnswer, CitationSource, CitationParseError, CitationValidation,
    citation_json_schema, parse_cited_answer, validate_citations,
    render_cited_answer, used_citation_sources,
)

from obsidian_rag.context import (
    BuiltContext, ContextBudgetError, ContextConfig, GenerationCounter,
    build_context, load_generation_counter,
)
from obsidian_rag.retrieval.models import SearchResult


_NO_EVIDENCE = "The provided notes do not contain enough information to answer this question."


class CitedGenerationError(ValueError):
    """A failed response retained for diagnostics; never silently repaired."""

    def __init__(self, code, raw_response, *, validation=None, token_usage=None):
        super().__init__(f'Cited generation failed: {code}.')
        self.code = code
        self.raw_response = raw_response
        self.validation = validation
        self.token_usage = token_usage


@dataclass(frozen=True)
class CitedGeneration:
    answer: CitedAnswer
    sources: tuple[CitationSource, ...]
    context_id: str
    raw_response: str | None
    prompt_tokens: int | None
    actual_prompt_tokens: int | None
    output_tokens: int | None
    require_quotes: bool = False

    @property
    def validation(self) -> CitationValidation:
        return validate_citations(self.answer, self.sources, require_quotes=self.require_quotes)

    @property
    def text(self) -> str:
        return render_cited_answer(self.answer, self.sources)

    def to_dict(self) -> dict:
        return {'answer': self.answer.to_dict(), 'text': self.text,
                'sources': [asdict(s) for s in used_citation_sources(self.answer, self.sources)],
                'context_id': self.context_id, 'validation': self.validation.to_dict(),
                'raw_response': self.raw_response,
                'token_usage': {'prompt_tokens': self.prompt_tokens,
                                'actual_prompt_tokens': self.actual_prompt_tokens,
                                'output_tokens': self.output_tokens}}


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
    if isinstance(question, BuiltContext) and question.citation_mode in ('structured', 'quoted'):
        return generate_cited_answer(question, results, client=client, model=model,
                                     config=config, counter=counter).text
    context = _prepare_context(question, results, client=client, model=model, config=config,
                               counter=counter, citation_mode='legacy')
    if not context.has_evidence:
        return _NO_EVIDENCE
    response = _chat(context, client=client, model=model)
    answer = response.message.content
    if answer is None or not answer.strip():
        raise ValueError('Ollama returned an empty answer.')
    return answer.strip()


def generate_cited_answer(
    question: str | BuiltContext, results: list[SearchResult] | None = None, *,
    client: Client, model: str | None = None, config: ContextConfig | None = None,
    counter: GenerationCounter | None = None,
) -> CitedGeneration:
    """Generate once with a schema, validate IDs, and retain raw output and provenance.

    Explicit contexts must already use the structured protocol. The legacy
    string API remains available for comparisons; it does not acquire these
    guarantees. Schema validity and reference validity do not prove claim support.
    """
    context = _prepare_context(question, results, client=client, model=model, config=config,
                               counter=counter, citation_mode='structured')
    context.verify_citation_mapping()
    sources = context.citation_sources
    if not context.has_evidence:
        answer = CitedAnswer('insufficient_evidence', (), (_NO_EVIDENCE,))
        return CitedGeneration(answer, (), context.context_id, None, context.prompt_tokens, None, None)
    require_quotes = context.citation_mode == 'quoted'
    response = _chat(context, client=client, model=model,
                     schema=citation_json_schema([s.source_id for s in sources], include_quotes=require_quotes))
    raw = response.message.content
    usage = {'prompt_tokens': context.prompt_tokens, 'actual_prompt_tokens': response.prompt_eval_count,
             'output_tokens': response.eval_count}
    if response.done_reason == 'length':
        raise CitedGenerationError('truncated_output', raw, token_usage=usage)
    try:
        answer = parse_cited_answer(raw)
    except CitationParseError as error:
        raise CitedGenerationError('invalid_structure', raw, token_usage=usage) from error
    validation = validate_citations(answer, sources, require_quotes=require_quotes)
    if not validation.references_valid:
        raise CitedGenerationError('invalid_references', raw, validation=validation, token_usage=usage)
    return CitedGeneration(answer, sources, context.context_id, raw, context.prompt_tokens,
                           response.prompt_eval_count, response.eval_count, require_quotes)


def _prepare_context(question, results, *, client, model, config, counter, citation_mode):
    if isinstance(question, BuiltContext):
        if results is not None or config is not None or counter is not None:
            raise ValueError('Do not override a BuiltContext with results, config or counter.')
        context = question
    else:
        if results is None:
            raise ValueError('Retrieved results are required with a question.')
        # Validate evidence and handle missing evidence before contacting a model.
        context = build_context(question, results, citation_mode=citation_mode)
        if context.has_evidence:
            selected_model = model or (counter.model if counter else 'qwen3.5:4b')
            counter = counter or load_generation_counter(client=client, model=selected_model)
            context = build_context(question, results, config=config or ContextConfig(), counter=counter,
                                    citation_mode=citation_mode)
    if context.status == 'budget_exhausted':
        raise ContextBudgetError('No evidence fits the context budget; increase the window or reduce the output reserve.')
    return context


def _chat(context, *, client, model, schema=None):
    if context.config is None or context.counter is None or context.prompt_tokens is None:
        raise ValueError('Generation requires a budgeted BuiltContext with a message counter.')
    selected_model = model or context.counter.model
    if selected_model != context.counter.model:
        raise ValueError('Generation model differs from the context token counter.')
    messages = context.messages
    if (context.counter(messages) != context.prompt_tokens
            or context.prompt_tokens > context.config.input_budget):
        raise ContextBudgetError('Final messages no longer match the measured context budget.')
    extra = {'format': schema} if schema is not None else {}
    response = client.chat(
        model=selected_model, messages=messages, stream=False, think=False,
        options={'temperature': 0, 'num_ctx': context.config.context_window,
                 'num_predict': context.config.max_output_tokens}, **extra,
    )
    actual = response.prompt_eval_count
    if actual is not None:
        if actual > context.config.input_budget:
            raise ContextBudgetError('Ollama input count exceeds the context budget.')
        if not context.counter.is_estimate and actual != context.prompt_tokens:
            raise ValueError('Ollama input count differs from the context counter; revalidate model/template compatibility.')
    return response
