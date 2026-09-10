"""Fixed pipelines against scripted providers and real local snapshot contracts."""

from dataclasses import asdict, replace
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import Mock

from ollama import Client, EmbedResponse
import pytest
from tokenizers import Tokenizer, models

from arkb.config import RetrievalConfig, RuntimeConfig
from arkb.evaluation.agent_runner import main, run_baseline_evaluation, run_evaluation
from arkb.evaluation.baselines import execute_baseline
from arkb.evaluation.datasets import load_agent_eval_dataset
from arkb.evaluation.metrics import ranking_metrics
from arkb.evaluation.models import BASELINE_MODES, AgentEvalCase, BaselineEvalConfig
from arkb.knowledge.documents import load_notes
from arkb.knowledge.indexing import build_index
from arkb.knowledge.models import EmbeddingSpec
from arkb.knowledge.sqlite import SQLiteStorage
from arkb.retrieval.models import SearchResponse, SearchResult, chunk_result
from arkb.runtime import Runtime
from tests.evaluation.test_agent_dataset import DATASET, NOTES


@pytest.fixture(autouse=True)
def forbid_agent_and_generation(monkeypatch):
    guards = []
    for target in ('arkb.runtime.Runtime.ask', 'arkb.runtime.Runtime.run_agent',
                   'arkb.runtime.Runtime.agent_tools', 'arkb.agent.loop.run_agent',
                   'arkb.generation.generate.generate_cited_answer',
                   'arkb.knowledge.documents.DocumentAccess.read'):
        guard = Mock(side_effect=AssertionError('Baseline must not invoke Agent, read, or generation'))
        monkeypatch.setattr(target, guard)
        guards.append(guard)
    yield
    for guard in guards:
        guard.assert_not_called()


def hit(source, chunk):
    return SearchResult(source_id=source, source=source, chunk_id=chunk,
                        content='verbatim evidence', method='bm25', score=1., score_type='bm25')


@pytest.mark.parametrize('name', BASELINE_MODES)
def test_each_baseline_passes_only_original_query_and_one_fixed_request(name, monkeypatch):
    query = '  查找 | 原始 query\nwithout rewrites '
    case = AgentEvalCase(id='c', query=query, task_type='direct_read', expected_sources=('a.md',))
    response = SearchResponse(query=query, method='fixed', index_id='snapshot', results=(hit('a.md', '1'),))
    engine = SimpleNamespace(search=Mock(return_value=response))
    ticks = iter([10., 10.25])
    monkeypatch.setattr('arkb.evaluation.baselines.perf_counter', lambda: next(ticks))
    row = execute_baseline(case, name, engine=engine, top_k=10, ks=(1, 3, 5, 10),
                           index_version='snapshot', known_sources={'a.md'})
    mode, rerank = BASELINE_MODES[name]
    engine.search.assert_called_once_with(query, mode=mode, rerank=rerank, top_k=10, filters=None)
    assert row.latency_ms == 250 and row.mode == mode and row.rerank == rerank
    assert row.status == 'ok' and row.response is response
    assert row.metrics['recall_at_1'] == row.metrics['mrr'] == row.metrics['ndcg_at_10'] == 1


def test_chunk_sources_collapse_in_first_occurrence_order_and_reuse_metrics(monkeypatch):
    case = AgentEvalCase(id='c', query='original', task_type='exploratory_retrieval', expected_sources=('a.md', 'b.md'))
    hits = (hit('x.md', '1'), hit('a.md', '2'), hit('a.md', '3'), hit('b.md', '4'))
    engine = SimpleNamespace(search=Mock(return_value=SearchResponse(
        query=case.query, method='bm25', index_id='v', results=hits)))
    formula = Mock(wraps=ranking_metrics)
    monkeypatch.setattr('arkb.evaluation.baselines.ranking_metrics', formula)
    row = execute_baseline(case, 'bm25', engine=engine, top_k=10, ks=(1, 3, 5, 10),
                           index_version='v', known_sources={'a.md', 'b.md', 'x.md'})
    assert row.retrieved_sources == ('x.md', 'a.md', 'a.md', 'b.md')
    assert row.ranked_sources == ('x.md', 'a.md', 'b.md')
    assert row.metrics['recall_at_1'] == 0 and row.metrics['recall_at_3'] == 1 and row.metrics['mrr'] == .5
    expected = ranking_metrics({'a.md': 1, 'b.md': 1}, row.ranked_sources, k=3)
    assert row.metrics['ndcg_at_3'] == expected['ndcg_at_k']
    assert row.response.results == hits and engine.search.call_count == 1
    for k in (1, 3, 5, 10):
        assert any(call.args == ({'a.md': 1, 'b.md': 1}, row.ranked_sources) and call.kwargs == {'k': k}
                   for call in formula.call_args_list)


def test_not_applicable_error_and_valid_empty_are_distinct():
    engine = SimpleNamespace(search=Mock())
    options = dict(engine=engine, top_k=3, ks=(1, 3), index_version='v', known_sources={'a.md'})
    none = execute_baseline(AgentEvalCase(id='none', query='hello', task_type='no_retrieval'), 'bm25', **options)
    assert none.status == 'not_applicable' and none.latency_ms is None and none.ranked_sources is None
    assert all(value is None for value in none.metrics.values())
    engine.search.assert_not_called()
    case = AgentEvalCase(id='c', query='query', task_type='knowledge_qa', expected_sources=('a.md',))
    engine.search.side_effect = ConnectionError('retrieval failed')
    failed = execute_baseline(case, 'semantic', **options)
    assert failed.status == 'error' and failed.error['type'] == 'ConnectionError'
    assert failed.response is None and failed.ranked_sources is None and failed.latency_ms >= 0
    assert all(value is None for value in failed.metrics.values())
    engine.search.side_effect = None
    engine.search.return_value = SearchResponse(query='query', method='bm25', index_id='v')
    empty = execute_baseline(case, 'bm25', **options)
    assert empty.status == 'ok' and empty.ranked_sources == () and empty.error is None
    assert set(empty.metrics.values()) == {0.}


@pytest.mark.parametrize('response', [
    None, SearchResponse(query='other query', method='bm25', index_id='v'),
    SearchResponse(query='query', method='bm25', index_id='different'),
    SearchResponse(query='query', method='bm25', index_id='v', results=(hit('unknown.md', '1'),)),
    SearchResponse(query='query', method='bm25', index_id='v', results=(hit('a.md', '1'),) * 2),
])
def test_invalid_response_is_an_explicit_case_error(response):
    row = execute_baseline(AgentEvalCase(id='c', query='query', task_type='exact_lookup', expected_sources=('a.md',)),
        'bm25', engine=SimpleNamespace(search=Mock(return_value=response)), top_k=10, ks=(10,),
        index_version='v', known_sources={'a.md'})
    assert row.status == 'error' and row.error['type'] == 'ValueError' and row.metrics['mrr'] is None


@pytest.fixture
def snapshot_factory(tmp_path, qdrant, qdrant_config):
    def create(*, full_dataset=False):
        notes = NOTES if full_dataset else tmp_path / 'notes'
        if not full_dataset:
            notes.mkdir()
            for source, text in [('a.md', 'Alpha evidence'), ('b.md', 'Beta evidence'), ('c.md', 'Other material')]:
                (notes / source).write_text('# ' + source + '\n' + text)
        cases = (load_agent_eval_dataset(DATASET) if full_dataset else [
            AgentEvalCase(id='lookup', query='Alpha | original\nquery', task_type='exact_lookup', expected_sources=('a.md',)),
            AgentEvalCase(id='lookup2', query='Beta', task_type='exact_lookup', expected_sources=('b.md',)),
            AgentEvalCase(id='semantic', query='Unseen query', task_type='semantic_discovery', expected_sources=('b.md',)),
            AgentEvalCase(id='read', query='Read a.md', task_type='direct_read', expected_sources=('a.md',)),
            AgentEvalCase(id='none', query='Hello', task_type='no_retrieval'),
        ])
        dataset = DATASET if full_dataset else tmp_path / 'cases.jsonl'
        if not full_dataset:
            dataset.write_text(''.join(json.dumps(asdict(c), ensure_ascii=False) + '\n' for c in cases))
        spec = EmbeddingSpec(model='fixture', model_revision='fixed', dimensions=2, document_template='title-body-v1')
        client = Mock(spec=Client)
        client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
        client.chat.side_effect = AssertionError('No chat model for baselines')
        tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
        db = tmp_path / 'index.sqlite'
        with SQLiteStorage(db) as storage:
            build = build_index(storage, load_notes(notes), spec=spec, vault_id='eval', client=client,
                tokenizer=tokenizer, max_input_tokens=100, chunking='none',
                qdrant_client=qdrant, qdrant_config=qdrant_config, source_scope=str(notes.resolve()))
            records = storage.snapshot_records(build.manifest.index_version)
        client.reset_mock()
        config = BaselineEvalConfig(dataset_path=dataset, db=db, notes_dir=notes, vault_id='eval', output_dir=tmp_path / 'run')
        return SimpleNamespace(config=config, cases=cases, spec=spec, client=client, tokenizer=tokenizer,
                               records=records, manifest=build.manifest)
    return create


def fake_runtime(snapshot, outcome=None):
    def search(query, **options):
        if outcome is not None:
            result = outcome(query, options)
            if result is not None:
                return result
        return SearchResponse(query=query, method=options['mode'], index_id=snapshot.manifest.index_version,
            results=tuple(chunk_result(r, method='bm25', index_id=snapshot.manifest.index_version)
                          for r in snapshot.records[:options['top_k']]))
    engine = SimpleNamespace(search=Mock(side_effect=search), reranker=None)
    return SimpleNamespace(retrieval_engine=Mock(return_value=engine), config=RuntimeConfig()), engine


@pytest.mark.parametrize('names', [('bm25',), ('semantic',), ('hybrid',), ('hybrid_rerank',), tuple(BASELINE_MODES)])
def test_runner_selects_single_or_all_pipelines_and_preserves_v1_artifacts(snapshot_factory, names):
    snapshot = snapshot_factory()
    runtime, engine = fake_runtime(snapshot)
    config = replace(snapshot.config, baselines=names)
    run = run_evaluation(config, runtime=runtime)
    assert len(run.results) == 5 * len(names) and engine.search.call_count == 4 * len(names)
    assert run.summary['total_cases'] == 5 and run.summary['failure_count'] == 0
    assert set(run.summary['overall']) == set(names)
    request = runtime.retrieval_engine.call_args
    assert request.kwargs['modes'] == tuple(dict.fromkeys(BASELINE_MODES[n][0] for n in names))
    assert request.kwargs['rerank'] == ('hybrid_rerank' in names)
    for name in names:
        overall = run.summary['overall'][name]
        assert overall['case_count'] == 5 and overall['evaluated_cases'] == overall['applicable_cases'] == 4
        assert overall['not_applicable_cases'] == 1 and overall['latency_defined_cases'] == 4
        assert set(overall['metric_denominators'].values()) == {4}
        assert set(run.summary['by_task_type']['no_retrieval'][name]['metrics'].values()) == {None}
        assert run.summary['by_task_type']['exact_lookup'][name]['case_count'] == 2
    metadata = json.loads((run.output_dir / 'run_metadata.json').read_text())
    assert metadata['status'] == 'completed' and metadata['completed_results'] == len(run.results)
    assert metadata['index_version'] == snapshot.manifest.index_version
    assert metadata['knowledge_changed'] is False and metadata['setup_ms'] >= 0
    assert metadata['dataset_sha256'] == hashlib.sha256(config.dataset_path.read_bytes()).hexdigest()
    assert (run.output_dir / 'cases.jsonl').read_bytes() == config.dataset_path.read_bytes()
    rows = [json.loads(line) for line in (run.output_dir / 'results.jsonl').read_text().splitlines()]
    assert len(rows) == len(run.results) and rows[0]['query'] == snapshot.cases[0].query
    assert all(r['schema_version'] == 'retrieval-baselines-v1' for r in rows)
    report = (run.output_dir / 'report.md').read_text()
    assert 'R@1' in report and 'R@3' in report and 'R@5' in report and 'R@10' in report and 'nDCG@10' in report
    assert 'No retrieval errors.' in report and 'synthetic Agent ranking' in report


def test_case_error_keeps_completed_results_and_denominators(snapshot_factory):
    snapshot = snapshot_factory()
    def outcome(query, options):
        if query == snapshot.cases[0].query:
            raise ConnectionError('failure | detail\nnext')
        if query == snapshot.cases[2].query:
            return SearchResponse(query=query, method='bm25', index_id=snapshot.manifest.index_version)
    runtime, engine = fake_runtime(snapshot, outcome)
    run = run_baseline_evaluation(replace(snapshot.config, baselines=('bm25',)), runtime=runtime)
    overall = run.summary['overall']['bm25']
    assert overall['failure_count'] == 1 and overall['applicable_cases'] == 4
    assert overall['evaluated_cases'] == 3 and overall['latency_defined_cases'] == 4
    assert overall['metric_denominators']['recall_at_10'] == 3
    assert overall['metrics']['recall_at_10'] == pytest.approx(2 / 3)
    assert run.summary['by_task_type']['exact_lookup']['bm25']['metric_denominators']['mrr'] == 1
    assert run.results[0].error and run.results[0].ranked_sources is None
    assert run.results[2].error is None and run.results[2].ranked_sources == ()
    assert 'failure \\| detail<br>next' in (run.output_dir / 'report.md').read_text()


@pytest.mark.parametrize('failure,status', [(KeyboardInterrupt(), 'interrupted'), (OSError('artifact write failed'), 'failed')])
def test_interruptions_and_artifact_failures_preserve_flushed_rows(snapshot_factory, monkeypatch, failure, status):
    snapshot = snapshot_factory()
    runtime, engine = fake_runtime(snapshot)
    if isinstance(failure, KeyboardInterrupt):
        original = engine.search.side_effect
        def search(query, **options):
            if query == snapshot.cases[1].query:
                raise failure
            return original(query, **options)
        engine.search.side_effect = search
    else:
        import arkb.evaluation.agent_runner as runner
        original_json = runner._json
        def write(value, **options):
            if value.get('case_id') == snapshot.cases[1].id:
                raise failure
            return original_json(value, **options)
        monkeypatch.setattr(runner, '_json', write)
    with pytest.raises(type(failure)):
        run_baseline_evaluation(replace(snapshot.config, baselines=('bm25',)), runtime=runtime)
    rows = (snapshot.config.output_dir / 'results.jsonl').read_text().splitlines()
    assert len(rows) == 1
    metadata = json.loads((snapshot.config.output_dir / 'run_metadata.json').read_text())
    assert metadata['status'] == status and metadata['completed_results'] == 1


def test_snapshot_setup_and_output_errors_fail_before_requests(snapshot_factory):
    snapshot = snapshot_factory()
    runtime, engine = fake_runtime(snapshot)
    config = snapshot.config
    with pytest.raises(ValueError, match='Unknown index version'):
        run_baseline_evaluation(replace(config, index_version='absent'), runtime=runtime)
    assert not config.output_dir.exists()
    (config.notes_dir / 'a.md').write_text('# Changed\nNew body')
    with pytest.raises(ValueError, match='Live knowledge differs'):
        run_baseline_evaluation(config, runtime=runtime)
    assert not config.output_dir.exists()
    config.output_dir.mkdir()
    marker = config.output_dir / 'results.jsonl'
    marker.write_text('previous\n')
    with pytest.raises(FileExistsError):
        run_baseline_evaluation(config, runtime=runtime)
    assert marker.read_text() == 'previous\n'
    runtime.retrieval_engine.assert_not_called()
    engine.search.assert_not_called()


def test_ready_snapshot_is_pinned_and_later_live_drift_invalidates_comparison(snapshot_factory, monkeypatch):
    snapshot = snapshot_factory()
    runtime, engine = fake_runtime(snapshot)
    active = Mock(side_effect=AssertionError('Explicit version must not follow the active pointer'))
    monkeypatch.setattr(SQLiteStorage, 'active_manifest', active)
    original = engine.search.side_effect
    def search(query, **options):
        (snapshot.config.notes_dir / 'a.md').write_text('# Changed\nNew live body')
        return original(query, **options)
    engine.search.side_effect = search
    config = replace(snapshot.config, baselines=('bm25',), index_version=snapshot.manifest.index_version)
    run = run_baseline_evaluation(config, runtime=runtime)
    active.assert_not_called()
    assert run.summary['knowledge_changed'] is True
    assert json.loads((run.output_dir / 'run_metadata.json').read_text())['status'] == 'invalidated'


@pytest.mark.parametrize('options', [
    {'baselines': ()}, {'baselines': ('bm25', 'bm25')}, {'baselines': ('agent',)}, {'baselines': 'bm25'},
    {'top_k': 0}, {'top_k': True}, {'top_k': 21}, {'index_version': ''}, {'exact': 'yes'},
    {'retrieval_config': {}}, {'runtime_config': {}}, {'notes_dir': None},
    {'retrieval_config': RetrievalConfig(rerank_candidates=5)},
    {'retrieval_config': RetrievalConfig(rrf_k=-1)},
])
def test_invalid_baseline_configuration(options):
    with pytest.raises(ValueError):
        BaselineEvalConfig(**options)


def test_custom_top_k_only_reports_supported_cutoffs():
    assert BaselineEvalConfig(top_k=2).metric_ks == (1, 2)
    assert BaselineEvalConfig(top_k=7).metric_ks == (1, 3, 5, 7)
    assert BaselineEvalConfig(baselines=('bm25',), top_k=30).metric_ks == (1, 3, 5, 10, 30)


def test_all_versioned_cases_through_real_engine_and_shared_runner(snapshot_factory, qdrant, monkeypatch):
    snapshot = snapshot_factory(full_dataset=True)
    monkeypatch.setattr(Runtime, 'model_client', lambda self: snapshot.client)
    monkeypatch.setattr(Runtime, 'tokenizer', lambda self: snapshot.tokenizer)
    monkeypatch.setattr(Runtime, 'qdrant_client', lambda self, url: qdrant)
    monkeypatch.setattr('arkb.knowledge.embeddings.resolve_embedding_spec', lambda *a, **kw: snapshot.spec)
    scorer = SimpleNamespace(identity='scripted-relevance-v1', score_type='relevance_logit',
        score=Mock(side_effect=lambda query, hits: [float(len(hits) - i) for i in range(len(hits))]))
    factory = Mock(return_value=scorer)
    monkeypatch.setattr('arkb.retrieval.qwen_rerank.QwenRerankerScorer', factory)
    before = snapshot.config.db.read_bytes()
    with Runtime() as runtime:
        run = run_evaluation(snapshot.config, runtime=runtime)
        assert runtime.status(db=snapshot.config.db, vault_id='eval')['active_version'] == snapshot.manifest.index_version
    assert run.summary['total_cases'] == 40 and run.summary['total_results'] == 160
    assert run.summary['failure_count'] == 0 and len(run.summary['by_task_type']) == 6
    assert sum(r.status == 'not_applicable' for r in run.results) == 24
    assert all(group['evaluated_cases'] == 34 for group in run.summary['overall'].values())
    assert snapshot.client.embed.call_count == 34 * 3
    assert scorer.score.call_count == 34 and factory.call_count == 1
    snapshot.client.chat.assert_not_called()
    assert snapshot.config.db.read_bytes() == before


def test_existing_runner_cli_selects_baselines(snapshot_factory, monkeypatch, capsys):
    snapshot = snapshot_factory()
    assert main(['--baselines', 'bm25', '--dataset', str(snapshot.config.dataset_path),
                 '--output', str(snapshot.config.output_dir), '--db', str(snapshot.config.db),
                 '--notes-dir', str(snapshot.config.notes_dir), '--vault-id', 'eval', '--top-k', '3']) == 0
    printed = json.loads(capsys.readouterr().out)
    assert list(printed['summary']['overall']) == ['bm25'] and printed['summary']['settings']['metric_ks'] == [1, 3]
    assert 'recall_at_10' not in printed['summary']['overall']['bm25']['metrics']
    with pytest.raises(SystemExit) as error:
        main(['--baselines', 'bm25', '--num-trials', '2'])
    assert error.value.code == 2


@pytest.mark.parametrize('args', [
    ['--index-version', 'must-not-be-silently-ignored'],
    ['--baselines', 'bm25', '--model', 'must-not-be-silently-ignored'],
])
def test_cli_rejects_mixed_agent_and_baseline_settings_before_execution(args):
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2
