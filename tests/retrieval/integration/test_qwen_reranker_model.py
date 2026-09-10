"""Opt-in real Qwen scoring and production snapshot retrieval; no index writes."""

from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter

import pytest

from arkb.config import RetrievalConfig, RuntimeConfig
from arkb.evaluation.retrieval import evaluate_reranker
from arkb.retrieval import SearchResult
from arkb.retrieval.qwen_rerank import QWEN_MODEL, QWEN_REVISION, QwenRerankerScorer
from arkb.retrieval.rerank import Reranker
from arkb.runtime import Runtime


pytestmark = [pytest.mark.integration, pytest.mark.skipif(
    os.environ.get('ARKB_RUN_RERANKER_TESTS') != '1', reason='optional local Qwen reranker')]


@pytest.fixture(scope='module', autouse=True)
def inference_threads():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(4)
    yield
    torch.set_num_threads(previous)


def test_real_qwen_promotes_evidence_and_handles_padding_truncation_and_batches(tmp_path):
    scorer = QwenRerankerScorer(cache_folder=os.environ.get('ARKB_RERANKER_CACHE'),
        local_files_only=os.environ.get('ARKB_RERANKER_OFFLINE', '1') == '1')
    candidates = tuple(SearchResult(source_id=str(i), source=f'{i}.md', content=text, method='frozen')
        for i, text in enumerate(['Dogs bark and cats meow. ' * 600, 'Paris is the capital of France.']))
    query = 'What is the capital of France?'
    reranker = Reranker(scorer)
    report = evaluate_reranker(reranker, query, candidates, {'1.md': 1}, top_k=1)
    assert report['before']['mrr'] == 0 and report['after']['mrr'] == 1
    inputs = scorer._inputs(query, candidates)
    assert inputs['input_ids'].shape[1] == scorer.max_length == 512
    assert inputs['attention_mask'][1, 0].item() == 0  # left padding
    assert inputs['attention_mask'][:, -1].tolist() == [1, 1]  # yes/no scoring position
    first = reranker.rerank(query, candidates)
    assert first == reranker.rerank(query, candidates)
    assert first[0].source == '1.md' and first[0].score > first[1].score
    assert first[1].content == candidates[0].content  # evidence is never truncated
    batched = scorer.score(query, candidates)
    singles = [scorer.score(query, [hit])[0] for hit in candidates]
    assert batched == pytest.approx(singles, abs=1e-4)
    assert scorer.score(query, []) == []
    (tmp_path / 'frozen.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')


@pytest.fixture(scope='module')
def snapshot():
    if not os.environ.get('ARKB_RERANKER_TEST_DB'):
        pytest.skip('Set ARKB_RERANKER_TEST_DB to an example_notes snapshot for runtime integration.')
    db = Path(os.environ['ARKB_RERANKER_TEST_DB'])
    before = hashlib.sha256(db.read_bytes()).hexdigest()
    options = {'db': db, 'vault_id': os.environ.get('ARKB_RERANKER_TEST_VAULT', 'default')}
    with Runtime(RuntimeConfig(host=os.environ.get('ARKB_RERANKER_TEST_HOST', RuntimeConfig.host),
            qdrant_url=os.environ.get('ARKB_RERANKER_TEST_QDRANT_URL'),
            offline=os.environ.get('ARKB_RERANKER_OFFLINE', '1') == '1')) as runtime:
        status = runtime.status(**options)
        yield runtime, options
        assert runtime.status(**options) == status
    assert hashlib.sha256(db.read_bytes()).hexdigest() == before


@pytest.mark.parametrize('mode', ['bm25', 'semantic', 'hybrid'])
def test_runtime_reranks_real_snapshot_preserving_evidence_and_filters(snapshot, mode, tmp_path, monkeypatch):
    runtime, scope = snapshot
    query = 'How does Reciprocal Rank Fusion combine lexical and semantic rankings?'
    source = '13_hybrid_rank_fusion.md'
    settings = RetrievalConfig(reranker_cache=os.environ.get('ARKB_RERANKER_CACHE'))
    # Observe the real scoring boundary. A separate embedding request may have
    # slightly different float scores, so it cannot verify exact provenance.
    calls = []
    original_rerank = Reranker.rerank
    def record_candidates(self, query, candidates, *, top_k=None):
        calls.append((query, tuple(candidates), top_k))
        return original_rerank(self, query, candidates, top_k=top_k)
    monkeypatch.setattr(Reranker, 'rerank', record_candidates)
    started = perf_counter()
    response = runtime.search(query, **scope, mode=mode, top_k=5, settings=settings, rerank=True, exact=True)
    elapsed_ms = (perf_counter() - started) * 1000
    assert len(calls) == 1 and calls[0][0] == query and calls[0][2] == 5
    candidates = calls[0][1]
    assert len(candidates) == settings.rerank_candidates == 20
    assert response.index_id == runtime.status(**scope)['active_version']
    assert response.method == mode + '+rerank' and len(response.results) == 5
    assert source in {hit.source for hit in response.results}
    assert [hit.score for hit in response.results] == sorted((hit.score for hit in response.results), reverse=True)
    for hit in response.results:
        metadata = hit.metadata['rerank']
        original = candidates[metadata['input_rank'] - 1]
        assert (hit.identity, hit.content, hit.source, hit.start_char, hit.end_char) == (
            original.identity, original.content, original.source, original.start_char, original.end_char)
        assert {key: value for key, value in hit.metadata.items() if key != 'rerank'} == original.metadata
        assert metadata['scorer'].startswith(f'{QWEN_MODEL}@{QWEN_REVISION}/cpu/float32/')
        assert metadata['candidate_count'] == len(candidates)
        assert metadata['input_score'] == original.score
        assert metadata['input_score_type'] == original.score_type
        assert metadata['input_method'] == original.method
        assert hit.score_type == 'yes_no_logit_difference' and math.isfinite(hit.score)
    filtered = runtime.search(query, **scope, mode=mode, top_k=2, source=source,
                              settings=settings, rerank=True, exact=True)
    assert filtered.results and all(hit.source == source for hit in filtered.results)
    assert len(calls) == 2 and all(hit.source == source for hit in calls[1][1])
    (tmp_path / f'{mode}.json').write_text(json.dumps({
        'mode': mode, 'latency_including_model_load_ms': elapsed_ms,
        'candidates': [asdict(hit) for hit in candidates],
        'response': asdict(response), 'filtered': asdict(filtered),
    }, ensure_ascii=False, indent=2) + '\n')
