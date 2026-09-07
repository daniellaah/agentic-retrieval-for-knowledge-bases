"""Score saved predictions separately from generation and source citation checks."""

from dataclasses import asdict, dataclass
import hashlib
from importlib.resources import files
import json
import math
import re
from time import perf_counter

from obsidian_rag.citation import parse_cited_answer


JUDGE_TEMPLATE_REVISION = '046949032b0328319cc9a02663a759ec601d9402'
JUDGE_TEMPLATE_SHA256 = '71fb29f51a56cea9fe647331d525648474195fa4816f7d05632459d572419d90'


@dataclass(frozen=True)
class JudgeConfig:
    """The caller resolves the actual installed model digest before scoring.

    Ollama is a recorded local adaptation, not a claim of identical execution
    to the official vLLM/Qwen3-32B evaluator. No model weights are downloaded.
    """
    model: str
    model_revision: str
    context_window: int = 16384
    max_output_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.8
    top_k: int = 20
    think: bool = True

    def __post_init__(self):
        for value in (self.model, self.model_revision):
            if not isinstance(value, str) or not value.strip():
                raise ValueError('Judge model and revision are required.')
        for value in (self.context_window, self.max_output_tokens, self.top_k):
            if type(value) is not int or value <= 0:
                raise ValueError('Judge token limits and top_k must be positive integers.')
        if self.max_output_tokens >= self.context_window:
            raise ValueError('Judge output reserve must be smaller than its window.')
        if (type(self.think) is not bool or not math.isfinite(self.temperature) or self.temperature < 0
                or not math.isfinite(self.top_p) or not 0 < self.top_p <= 1):
            raise ValueError('Invalid judge sampling settings.')


def build_judge_prompt(question: str, prediction: str, reference: str) -> str:
    for value in (question, prediction, reference):
        if not isinstance(value, str) or not value.strip():
            raise ValueError('Judge question, prediction and reference must be nonblank.')
    resource = json.loads(files('obsidian_rag').joinpath('browsecomp_judge.json').read_text(encoding='utf-8'))
    template = resource['template']
    if (resource['revision'] != JUDGE_TEMPLATE_REVISION
            or hashlib.sha256(template.encode()).hexdigest() != JUDGE_TEMPLATE_SHA256):
        raise ValueError('Bundled official judge template has changed.')
    return template.format(question=question, response=prediction, correct_answer=reference)


def parse_judgment(raw: str) -> dict:
    """Parse official field names, rejecting ambiguous or missing verdicts."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError('Empty judge response.')
    fields, current = {}, None
    for line in raw.splitlines():
        line = line.replace('**', '').strip()
        match = re.match(r'^(extracted_final_answer|reasoning|correct|confidence):\s*(.*)$', line, re.I)
        if match:
            current = match[1].lower()
            if current in fields:
                raise ValueError('Duplicate judge field.')
            fields[current] = match[2]
        elif current == 'reasoning' and line:
            fields[current] += '\n' + line
    if (fields.get('correct', '').lower() not in ('yes', 'no')
            or not fields.get('extracted_final_answer') or not fields.get('reasoning')):
        raise ValueError('Incomplete or invalid judge verdict.')
    confidence = fields.get('confidence')
    if confidence is not None:
        try:
            confidence = float(confidence.removesuffix('%').strip())
        except ValueError:
            raise ValueError('Invalid judge confidence.') from None
        if not math.isfinite(confidence) or not 0 <= confidence <= 100:
            raise ValueError('Invalid judge confidence.')
    return {'correct': fields['correct'].lower() == 'yes',
            'extracted_final_answer': fields['extracted_final_answer'],
            'reasoning': fields['reasoning'], 'confidence': confidence}


def judge_answer(question: str, prediction: str, reference: str, *, client, config: JudgeConfig) -> dict:
    """Grade once, retaining raw output and judge errors separately from wrong answers."""
    prompt = build_judge_prompt(question, prediction, reference)
    result = {'status': 'error', 'correct': None, 'raw_response': None, 'error': None,
              'judge': {'backend': 'ollama', **asdict(config)},
              'template_revision': JUDGE_TEMPLATE_REVISION, 'template_sha256': JUDGE_TEMPLATE_SHA256,
              'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'token_usage': None}
    started = perf_counter()
    # Conservative byte bound for the Qwen byte-level tokenizers plus template
    # allowance. This is not an exact token count; reject rather than trim text.
    if len(prompt.encode('utf-8')) + 256 + config.max_output_tokens > config.context_window:
        result['error'] = {'code': 'input_budget', 'message': 'Judge input exceeds conservative byte budget.'}
        result['judge_ms'] = (perf_counter() - started) * 1000
        return result
    try:
        response = client.chat(model=config.model, messages=[{'role': 'user', 'content': prompt}],
                               stream=False, think=config.think,
                               options={'temperature': config.temperature, 'top_p': config.top_p,
                                        'top_k': config.top_k, 'num_ctx': config.context_window,
                                        'num_predict': config.max_output_tokens})
        result['raw_response'] = response.message.content
        result['thinking'] = response.message.thinking
        result['done_reason'] = response.done_reason
        result['token_usage'] = {'prompt_tokens': response.prompt_eval_count, 'output_tokens': response.eval_count}
        if response.done_reason == 'length':
            result['error'] = {'code': 'truncated_output', 'message': 'Judge output was truncated.'}
        else:
            try:
                verdict = parse_judgment(response.message.content)
                result.update(status='graded', **verdict)
            except ValueError as error:
                result['error'] = {'code': 'invalid_judgment', 'message': str(error)}
    except Exception as error:
        result['error'] = {'code': type(error).__name__, 'message': str(error)}
    result['judge_ms'] = (perf_counter() - started) * 1000
    return result


def prediction_text(answer: dict) -> str:
    """Only generated assertions/limitations, never the renderer's source appendix."""
    parsed = parse_cited_answer(json.dumps(answer, ensure_ascii=False))
    parts = [claim.text for claim in parsed.claims]
    if parsed.missing_information:
        parts.append('Missing information: ' + ' '.join(parsed.missing_information))
    return '\n\n'.join(parts)
