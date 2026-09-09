"""Generate grounded answers from the context builder's final messages."""

from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
from pathlib import Path

from ollama import Client

from arkb.context.citation import (
    CitedAnswer, CitationSource, CitationParseError, CitationValidation,
    citation_json_schema, parse_cited_answer, validate_citations,
    _render_cited_answer, _used_citation_sources,
)

from arkb.context.builder import (
    BuiltContext, ContextBudgetError, GenerationCounter,
)


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

    def __post_init__(self):
        if not isinstance(self.answer, CitedAnswer) or not isinstance(self.sources, tuple):
            raise ValueError('Generation results require an immutable answer and source tuple.')
        if type(self.require_quotes) is not bool:
            raise ValueError('require_quotes must be boolean.')

    @cached_property
    def validation(self) -> CitationValidation:
        return validate_citations(self.answer, self.sources, require_quotes=self.require_quotes)

    @cached_property
    def text(self) -> str:
        return _render_cited_answer(self.answer, self.sources, self.validation)

    def to_dict(self) -> dict:
        return {'answer': self.answer.to_dict(), 'text': self.text,
                'sources': [asdict(s) for s in _used_citation_sources(self.answer, self.sources)],
                'context_id': self.context_id, 'validation': self.validation.to_dict(),
                'raw_response': self.raw_response,
                'token_usage': {'prompt_tokens': self.prompt_tokens,
                                'actual_prompt_tokens': self.actual_prompt_tokens,
                                'output_tokens': self.output_tokens}}


def generate_cited_answer(context: BuiltContext, *, client: Client) -> CitedGeneration:
    """Send a prepared context once, preserving raw output and validated references.

    The context supplies both the generation model and its runtime budget.
    Reference validity does not prove that a claim is supported by its evidence.
    """
    if not isinstance(context, BuiltContext):
        raise TypeError('Generation requires a BuiltContext.')
    if context.status == 'budget_exhausted':
        raise ContextBudgetError('No evidence fits the context budget; increase the window or reduce the output reserve.')
    context.verify_citation_mapping()
    sources = context.citation_sources
    if not context.has_evidence:
        answer = CitedAnswer('insufficient_evidence', (), (_NO_EVIDENCE,))
        return CitedGeneration(answer, (), context.context_id, None, context.prompt_tokens, None, None)
    require_quotes = context.citation_mode == 'quoted'
    response = _chat(context, client=client,
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
    result = CitedGeneration(answer, sources, context.context_id, raw, context.prompt_tokens,
                              response.prompt_eval_count, response.eval_count, require_quotes)
    if not result.validation.references_valid:
        raise CitedGenerationError('invalid_references', raw, validation=result.validation, token_usage=usage)
    return result


def _chat(context, *, client, schema):
    if context.config is None or context.counter is None or context.prompt_tokens is None:
        raise ValueError('Generation requires a budgeted BuiltContext with a message counter.')
    messages = context.messages
    if (context.counter(messages) != context.prompt_tokens
            or context.prompt_tokens > context.config.input_budget):
        raise ContextBudgetError('Final messages no longer match the measured context budget.')
    response = client.chat(
        model=context.counter.model, messages=messages, stream=False, think=False,
        options={'temperature': 0, 'num_ctx': context.config.context_window,
                 'num_predict': context.config.max_output_tokens}, format=schema,
    )
    actual = response.prompt_eval_count
    if actual is not None:
        if actual > context.config.input_budget:
            raise ContextBudgetError('Ollama input count exceeds the context budget.')
        if not context.counter.is_estimate and actual != context.prompt_tokens:
            raise ValueError('Ollama input count differs from the context counter; revalidate model/template compatibility.')
    return response


# Text-only, system/user messages, think=False. This profile is deliberately
# pinned; an unrecognized model needs an explicit GenerationCounter adapter.
GENERATION_TOKENIZER_REPO = 'Qwen/Qwen3.5-4B'
GENERATION_TOKENIZER_REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
_GENERATION_TOKENIZER_SHA256 = '5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42'
_GENERATION_MODEL_DIGEST = '2a654d98e6fba55d452b7043684e9b57a947e393bbffa62485a7aac05ee4eefd'


def load_generation_counter(*, client, model: str = 'qwen3.5:4b',
                            cache_dir: Path | None = None, local_files_only: bool = False) -> GenerationCounter:
    """Load only the pinned generation tokenizer, never model weights.

    Validated with Ollama 0.33.2's Qwen3.5 renderer and the identified model
    artifact. Refuse custom system prompts, histories, templates or unknown
    digests rather than silently applying the embedding tokenizer or guessing.
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    matches = [m for m in client.list().models if m.model == model]
    if len(matches) != 1 or matches[0].digest != _GENERATION_MODEL_DIGEST:
        raise ValueError('No verified generation token counter for this model artifact; '
                         'use the supported qwen3.5:4b artifact or supply a GenerationCounter in Python.')
    info = client.show(model)
    if (info.template != '{{ .Prompt }}' or getattr(info, 'system', None) or getattr(info, 'messages', None) or
            info.modelinfo.get('general.architecture') != 'qwen35'):
        raise ValueError('Generation model template or defaults differ from the verified profile.')
    path = hf_hub_download(GENERATION_TOKENIZER_REPO, 'tokenizer.json',
                           revision=GENERATION_TOKENIZER_REVISION, cache_dir=cache_dir,
                           local_files_only=local_files_only, token=False)
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != _GENERATION_TOKENIZER_SHA256:
        raise ValueError('Generation tokenizer digest differs from the pinned artifact.')
    tokenizer = Tokenizer.from_str(raw.decode('utf-8'))
    tokenizer.normalizer = None  # Ollama preserves combining characters.

    def count(messages):
        if ([m.get('role') for m in messages] != ['system', 'user'] or
                any(set(m) != {'role', 'content'} for m in messages)):
            raise ValueError('Generation counter supports only text system/user messages.')
        # Ollama v0.33.2 model/renderers/qwen35.go, no tools and think=False.
        prompt = ''.join('<|im_start|>' + m['role'] + '\n' + m['content'].strip()
                         + '<|im_end|>\n' for m in messages)
        prompt += '<|im_start|>assistant\n<think>\n\n</think>\n\n'
        return len(tokenizer.encode(prompt, add_special_tokens=False).ids)

    return GenerationCounter(model, f'qwen35-text-no-think-v1:{_GENERATION_MODEL_DIGEST}:'
                             f'{GENERATION_TOKENIZER_REVISION}', count,
                             context_limit=info.modelinfo.get('qwen35.context_length'))
