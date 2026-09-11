"""Replayable component interventions and post-hoc stopping diagnostics.

Gold is used only in scoring. Retrieval and control never receive it.
"""
from dataclasses import asdict, dataclass

from arkb.agent.loop import _SEARCH_STALLED_INSTRUCTION
from arkb.agent.tools import _evidence
from arkb.evaluation.pilot import score_observations
from arkb.evaluation.v2 import evidence_scores
from arkb.retrieval.hybrid import rrf
from arkb.retrieval.models import SearchResult
from arkb.retrieval.rerank import Reranker


def stopping_order(config):
    cases, arms = config['cases'], list(config['arms'])
    if len(cases) != len(set(cases)) or len(arms) != 2 or config['trials'] != 2:
        raise ValueError('This crossover requires unique cases, two arms and two trials.')
    return [(cid, arm, trial) for trial in range(2)
            for cid in (cases if trial == 0 else list(reversed(cases)))
            for arm in (arms if (cases.index(cid)+trial)%2 == 0 else list(reversed(arms)))]


def fused_candidates(legs, depth, *, pool_size=20, rrf_k=60):
    """Prefix a frozen exact retrieval ranking; keep fusion/output caps fixed."""
    if set(legs) != {'bm25', 'semantic'} or type(depth) is not int or depth < 1:
        raise ValueError('Expected two ranked legs and positive depth.')
    lists = {name: tuple(SearchResult(**h) for h in hits[:depth]) for name, hits in legs.items()}
    union = rrf(lists, k=rrf_k)
    return union, union[:pool_size]


@dataclass
class FrozenScores:
    identity: str
    score_type: str
    candidates: tuple
    values: list

    def score(self, query, candidates):
        if tuple(candidates) != self.candidates:
            raise ValueError('Frozen scores belong to a different candidate pool.')
        return self.values


def component_rows(raw, case, dataset, protocol):
    if raw['query'] != case['query'] or raw['case_id'] != case['id']:
        raise ValueError('Frozen query/case mismatch.')
    cfg = protocol['component']
    scored = raw['rerank']
    _, pool = fused_candidates(raw['legs'], cfg['rerank_leg_depth'],
                              pool_size=cfg['fusion_pool_size'], rrf_k=cfg['rrf_k'])
    if scored['candidates'] != [asdict(h) for h in pool]:
        raise ValueError('Reranker input differs from registered frozen pool.')
    reranked = Reranker(FrozenScores(scored['identity'], scored['score_type'], pool,
                                    scored['scores'])).rerank(case['query'], pool, top_k=cfg['top_k'])
    rows = []
    for arm, options in cfg['arms'].items():
        union, candidates = fused_candidates(raw['legs'], options['leg_depth'],
                    pool_size=cfg['fusion_pool_size'], rrf_k=cfg['rrf_k'])
        if options['rerank'] and candidates != pool:
            raise ValueError('Only the registered fixed-pool rerank intervention is supported.')
        hits = reranked if options['rerank'] else candidates[:cfg['top_k']]
        observations = [_evidence(h) for h in hits]
        stages = {'candidate_union': [_evidence(h) for h in union],
                  'fusion_pool': [_evidence(h) for h in candidates], 'returned': observations}
        evidence = {name: evidence_scores(case, values, dataset) for name, values in stages.items()}
        lost = {}
        for before, after in [('candidate_union', 'fusion_pool'), ('fusion_pool', 'returned')]:
            lost[before+'->'+after] = [f for f, ok in evidence[before]['facet_satisfied'].items()
                                       if ok and not evidence[after]['facet_satisfied'][f]]
        rows.append({'case_id': case['id'], 'family': case['intent_family_id'], 'arm': arm,
                     'trial': 0, 'status': 'ok', 'observations': observations,
                     'candidate_union_count': len(union), 'fusion_pool_count': len(candidates),
                     'stage_evidence': evidence, 'lost_facets': lost,
                     'metrics': score_observations(case, observations, dataset, k=cfg['top_k'])})
    return rows


def stopping_diagnostics(case, report, dataset):
    """Observed threshold crossing, never an oracle termination controller.

    Count later executed calls only after evidence has entered a model request.
    These calls are not automatically waste: verification may still be useful.
    """
    first = None
    trajectory = []
    for model in report['models']:
        hits = []
        for event in report['tools']:
            if event['submitted_to_model'] and event['turn'] < model['turn']:
                raw = event['raw_result']
                hits.extend([raw['result']] if event['name'] == 'read' else raw['results'])
        score = evidence_scores(case, hits, dataset)
        trajectory.append({'model_turn': model['turn'], 'evidence_coverage': score['evidence_coverage']})
        if first is None and score['evidence_coverage'] == 1:
            first = model['turn']
    counts = [sum(m.get('role') == 'system' and m.get('content') == _SEARCH_STALLED_INSTRUCTION
                  for m in event['request']['messages']) for event in report['models']]
    return {'first_full_evidence_model_turn': first,
            'executed_calls_after_label_threshold': sum(t['executed'] and t['turn'] >= first
                for t in report['tools']) if first is not None else None,
            'reminders_submitted': max(counts, default=0), 'evidence_by_model_turn': trajectory}
