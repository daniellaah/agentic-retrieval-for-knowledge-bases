"""Ranking, evidence coverage and citation metrics with explicit denominators."""

from collections.abc import Mapping, Sequence
import math

import numpy as np

def ranking_metrics(relevance: Mapping[str, int], ranked_ids: Sequence[str], *, k: int) -> dict:
    """Recall over all positive judgments; MRR over the supplied ranking.

    nDCG@K uses gain 2**grade-1 and log2(rank+1) discount. Unjudged IDs have
    grade zero. With no positive judgments all metrics are undefined (None).
    """
    if type(k) is not int or k <= 0:
        raise ValueError('k must be positive.')
    if not isinstance(relevance, Mapping) or any(
        not isinstance(key, str) or not key or type(grade) is not int or not 0 <= grade <= 30
        for key, grade in relevance.items()
    ):
        raise ValueError('Relevance requires nonblank IDs and integer grades in [0, 30].')
    if any(not isinstance(key, str) or not key for key in ranked_ids):
        raise ValueError('Ranked IDs must be nonblank strings.')
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError('Ranked IDs must not contain duplicates.')
    relevant = {key for key, grade in relevance.items() if grade > 0}
    if not relevant:
        return {'recall_at_k': None, 'mrr': None, 'ndcg_at_k': None}
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    dcg = sum((2**relevance.get(key, 0) - 1) / math.log2(rank + 1)
              for rank, key in enumerate(ranked_ids[:k], 1))
    return {'recall_at_k': len(relevant.intersection(ranked_ids[:k])) / len(relevant),
            'mrr': next((1 / rank for rank, key in enumerate(ranked_ids, 1) if key in relevant), 0.),
            'ndcg_at_k': dcg / idcg}


def recall_at_k(reference: list[str], candidate: list[str], k: int) -> float | None:
    """Set recall against exact neighbors; no reference neighbors means undefined."""
    if type(k) is not int or k <= 0:
        raise ValueError('k must be positive.')
    if len(set(reference)) != len(reference) or len(set(candidate)) != len(candidate):
        raise ValueError('Neighbor lists must not contain duplicate IDs.')
    expected = set(reference[:k])
    return len(expected & set(candidate[:k])) / len(expected) if expected else None


def evidence_statistics(chunks, case: dict) -> dict:
    """Measure source groups and union coverage of labeled Note.content spans.

    Labels must explicitly use body_start_char/body_end_char. Overlap is counted
    once; raw Markdown coordinates are never silently substituted. These are
    section-character metrics, not necessary-fact recall or generated-answer scores.
    """
    sources = {chunk.source for chunk in chunks}
    groups = case.get('required_source_groups', [])
    if not isinstance(groups, list) or any(not isinstance(g, list) or not g or
                                          any(not isinstance(s, str) or not s for s in g) for g in groups):
        raise ValueError('Expected nonempty source groups.')
    fractions = []
    for anchor in case.get('evidence_anchors', []):
        start, end = anchor.get('body_start_char'), anchor.get('body_end_char')
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError('Evidence requires valid body coordinates, with an exclusive end.')
        intervals = sorted((max(start, c.start_char), min(end, c.end_char)) for c in chunks
                           if c.source == anchor['source'] and c.start_char < end and c.end_char > start)
        covered, cursor = 0, start
        for left, right in intervals:
            covered += max(0, right - max(cursor, left))
            cursor = max(cursor, right)
        fractions.append(covered / (end - start))
    return {
        'source_group_recall': sum(any(s in sources for s in g) for g in groups) / len(groups) if groups else None,
        'section_coverage': float(np.mean(fractions)) if fractions else None,
        'all_sections_complete': all(f == 1 for f in fractions) if fractions else None,
    }


def citation_statistics(raw_response, sources, *, review: dict | None = None, require_quotes=False) -> dict:
    """Reference membership and coverage, separately from supplied human/judge labels.

    Every model claim is treated as requiring evidence. This does not detect
    omitted answer facts or multiple facts hidden in one claim. Support labels
    judge the cited sources jointly; the rate uses reviewed claims only and is
    always accompanied by review coverage. No labels means no semantic score.
    """
    from arkb.generation.citations import CitationParseError, parse_cited_answer, validate_citations
    metrics = {'structure_valid': False, 'references_valid': False, 'claim_count': None,
               'reference_count': None, 'valid_reference_count': None,
               'citation_id_validity': None, 'claim_reference_coverage': None,
               'support_review_coverage': None, 'supported_claim_rate': None,
               'answer_correct': None, 'answer_complete': None, 'issues': []}
    try:
        answer = parse_cited_answer(raw_response)
    except CitationParseError:
        metrics['issues'] = [{'code': 'invalid_structure'}]
        return metrics
    validation = validate_citations(answer, sources, require_quotes=require_quotes)
    known = {s.source_id for s in sources}
    references = [s for c in answer.claims for s in c.source_ids]
    valid = sum(s in known for s in references)
    count = len(answer.claims)
    metrics.update(structure_valid=True, references_valid=validation.references_valid,
                   claim_count=count, reference_count=len(references), valid_reference_count=valid,
                   citation_id_validity=valid / len(references) if references else None,
                   claim_reference_coverage=sum(any(s in known for s in c.source_ids) for c in answer.claims) / count if count else None,
                   issues=validation.to_dict()['issues'])
    if review is not None:
        labels = review.get('claim_support')
        if (not isinstance(review.get('reviewer'), str) or not review['reviewer'].strip()
                or not isinstance(labels, list) or len(labels) != count
                or any(label not in (None, 'supported', 'partial', 'contradicted', 'insufficient') for label in labels)):
            raise ValueError('Review requires an identified reviewer and one support label per claim.')
        # A supplied semantic score cannot bless missing or out-of-context references.
        for claim, label in zip(answer.claims, labels):
            if label == 'supported' and (not claim.source_ids or any(s not in known for s in claim.source_ids)):
                raise ValueError('A supported claim must have valid source references.')
        reviewed = [label for label in labels if label is not None]
        metrics['support_review_coverage'] = len(reviewed) / count if count else None
        metrics['supported_claim_rate'] = reviewed.count('supported') / len(reviewed) if reviewed else None
        for key in ('answer_correct', 'answer_complete'):
            value = review.get(key)
            if value is not None and type(value) is not bool:
                raise ValueError(f'{key} review must be boolean or null.')
            metrics[key] = value
    return metrics


def summarize_citations(rows) -> dict:
    """Macro rates on defined cases, with failures/counts reported alongside."""
    if not rows:
        raise ValueError('Citation summary requires at least one case.')
    summary = {'case_count': len(rows), 'success_count': sum(r['success'] for r in rows),
               'error_count': sum(not r['success'] for r in rows),
               'mean_generation_ms': float(np.mean([r['generation_ms'] for r in rows]))}
    for key in ('structure_valid', 'references_valid', 'citation_id_validity', 'claim_reference_coverage',
                'support_review_coverage', 'supported_claim_rate', 'answer_correct', 'answer_complete'):
        values = [r['metrics'][key] for r in rows if r['metrics'][key] is not None]
        summary[key] = float(np.mean(values)) if values else None
        summary[key + '_defined_cases'] = len(values)
    summary['claim_count'] = sum(r['metrics']['claim_count'] or 0 for r in rows)
    summary['reference_count'] = sum(r['metrics']['reference_count'] or 0 for r in rows)
    return summary
