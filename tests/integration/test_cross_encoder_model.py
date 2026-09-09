"""Opt-in real local reranker; ordinary tests never download a model."""

import os

import pytest

from arkb.retrieval import SearchResult
from arkb.retrieval.cross_encoder import CrossEncoderScorer
from arkb.retrieval.reranker import Reranker
from arkb.retrieval_evaluation import evaluate_reranker


@pytest.mark.skipif(os.environ.get('ARKB_RUN_RERANKER_TESTS') != '1', reason='optional cross-encoder model')
def test_real_cross_encoder_promotes_relevant_frozen_evidence():
    scorer = CrossEncoderScorer(cache_folder=os.environ.get('ARKB_RERANKER_CACHE'),
                                local_files_only=os.environ.get('ARKB_RERANKER_OFFLINE', '1') == '1')
    candidates = tuple(SearchResult(source_id=str(i), source=f'{i}.md', content=text, method='frozen')
                       for i, text in enumerate(['Dogs bark and cats meow.', 'Paris is the capital of France.']))
    reranker = Reranker(scorer)
    report = evaluate_reranker(reranker, 'What is the capital of France?', candidates,
                               {'1.md': 1}, top_k=1)
    assert report['before']['mrr'] == 0 and report['after']['mrr'] == 1
    first = reranker.rerank('What is the capital of France?', candidates)
    second = reranker.rerank('What is the capital of France?', candidates)
    assert first == second
    assert first[0].source == '1.md' and first[0].score > first[1].score
