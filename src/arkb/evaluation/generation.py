"""Evaluate context packing and generated citations on frozen retrieval evidence."""

import json
from time import perf_counter

from arkb.evaluation.metrics import evidence_statistics, citation_statistics

def evaluate_context(question, results, case, *, config, counter) -> dict:
    """Measure packed evidence against the original, snapshot-identified hits."""
    from arkb.generation.context import build_context

    if len({(h.metadata['index_version'], h.metadata['vault_id']) for h in results}) > 1:
        raise ValueError('Context evaluation requires one snapshot and vault.')
    started = perf_counter()
    context = build_context(question, results, config=config, counter=counter)
    elapsed = (perf_counter() - started) * 1000
    blocks = context.evidence_blocks
    groups = {}
    for block in blocks:
        hit = block.origins[0]
        key = (hit.metadata['index_version'], hit.source_id, hit.metadata['document_revision'])
        groups.setdefault(key, []).append((block.start_char, block.end_char))
    unique_chars = 0
    for intervals in groups.values():
        cursor = 0
        for start, end in sorted(intervals):
            unique_chars += max(0, end - max(cursor, start))
            cursor = max(cursor, end)
    chars = sum(len(b.content) for b in blocks)
    result = {
        'context': context.to_dict(),
        'metrics': {'prompt_tokens': context.prompt_tokens, 'block_count': len(blocks),
                    'body_characters': chars, 'unique_span_characters': unique_chars,
                    'duplicate_span_fraction': (chars - unique_chars) / chars if chars else 0.0,
                    'fits_budget': context.prompt_tokens <= config.input_budget,
                    'build_ms': elapsed, **evidence_statistics(blocks, case)},
    }
    reference = evidence_statistics(results, case)['section_coverage']
    coverage = result['metrics']['section_coverage']
    result['metrics']['section_coverage_retention'] = coverage / reference if reference else None
    return result


def evaluate_citation_context(context, *, client) -> dict:
    """Run one frozen context once, preserving failures as well as successes."""
    from httpx import HTTPError
    from ollama import ResponseError
    from arkb.generation.citations import citation_json_schema
    from arkb.generation.generate import CitedGenerationError, generate_cited_answer
    context.verify_citation_mapping()
    sources = context.citation_sources
    require_quotes = context.citation_mode == 'quoted'
    row = {'context': context.to_dict(), 'response_schema': citation_json_schema([s.source_id for s in sources], include_quotes=require_quotes),
           'success': False, 'result': None, 'raw_response': None, 'error': None}
    started = perf_counter()
    try:
        result = generate_cited_answer(context, client=client)
        payload = result.to_dict()
        payload.pop('raw_response')
        row.update(success=True, result=payload, raw_response=result.raw_response)
        # No-evidence short circuit has no model response; evaluate the application result.
        metric_input = result.raw_response if result.raw_response is not None else json.dumps(result.answer.to_dict())
    except (CitedGenerationError, ValueError, OSError, HTTPError, ResponseError) as error:
        row['raw_response'] = getattr(error, 'raw_response', None)
        row['error'] = {'code': getattr(error, 'code', type(error).__name__), 'message': str(error)}
        row['error']['token_usage'] = getattr(error, 'token_usage', None)
        metric_input = row['raw_response']
    row['generation_ms'] = (perf_counter() - started) * 1000
    row['metrics'] = citation_statistics(metric_input, sources, require_quotes=require_quotes)
    return row


def evaluate_citation_case(question, results, *, config, counter, client) -> dict:
    """Retain per-question budget failures instead of aborting a batch evaluation."""
    from arkb.generation.context import ContextBudgetError, build_context
    try:
        context = build_context(question, results, config=config, counter=counter, citation_mode='structured')
    except ContextBudgetError as error:
        return {'context': None, 'response_schema': None, 'success': False, 'result': None,
                'raw_response': None, 'generation_ms': 0,
                'error': {'code': 'context_budget', 'message': str(error)},
                'metrics': citation_statistics(None, [])}
    return evaluate_citation_context(context, client=client)


def main(argv=None) -> int:
    """Run the shared snapshot experiment, enabling context evaluation by default.

    --citations also generates answers. Keeping the shared runner preserves the
    exact-hit evidence, ANN diagnostics and artifact format used by past runs.
    """
    import sys
    from arkb.evaluation.retrieval import ann_main
    args = list(sys.argv[1:] if argv is None else argv)
    if '--context' not in args and '--citations' not in args:
        args.append('--context')
    return ann_main(args)


if __name__ == '__main__':
    raise SystemExit(main())
