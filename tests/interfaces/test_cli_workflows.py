import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

from httpx import ReadError, ReadTimeout
from ollama import ChatResponse, Client, EmbedResponse, Message, ResponseError
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from arkb.interfaces.cli import main
from arkb.runtime import Runtime


@pytest.fixture(autouse=True)
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    directory = tmp_path / "example_notes"
    directory.mkdir()
    (directory / "habits.md").write_text(
        "# Habit Stages\n\nA cue starts a habit.\n", encoding="utf-8"
    )
    (directory / "literature.md").write_text(
        "# Literature Notes\n\nPreserve the author's meaning.\n", encoding="utf-8"
    )
    (directory / "permanent.md").write_text(
        "# Permanent Notes\n\nDevelop one idea per note.\n", encoding="utf-8"
    )
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def client() -> MagicMock:
    client = MagicMock(spec=Client)
    client.__enter__.return_value = client
    client.embed.side_effect = [
        EmbedResponse(embeddings=[[0.0, 1.0], [3.0, 4.0], [1.0, 0.0]]),
        EmbedResponse(embeddings=[[1.0, 0.0]]),
    ]
    client.chat.return_value = ChatResponse(
        message=Message(
            role="assistant", content=json.dumps({'status': 'answered', 'claims': [
                {'text': 'Develop one idea per note.', 'source_ids': ['S1']}], 'missing_information': []})
        )
    )
    return client


@pytest.fixture(autouse=True)
def client_factory(client: MagicMock, monkeypatch: pytest.MonkeyPatch) -> Mock:
    factory = Mock(return_value=client)
    monkeypatch.setattr("ollama.Client", factory)
    return factory


@pytest.fixture(autouse=True)
def tokenizer_download(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Mock:
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "<|endoftext|>": 1},
                                           unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    tokenizer.post_processor = processors.TemplateProcessing(
        single="$A <|endoftext|>", special_tokens=[("<|endoftext|>", 1)],
    )
    path = tmp_path / "tokenizer.json"
    tokenizer.save(str(path))
    download = Mock(return_value=str(path))
    monkeypatch.setattr("arkb.knowledge.embeddings.hf_hub_download", download)
    return download


@pytest.fixture
def persistent_client(client):
    from ollama import ListResponse, ShowResponse
    client.list.return_value = ListResponse(models=[{'model': 'qwen3-embedding:0.6b', 'digest': 'actual-digest'}])
    client.show.return_value = ShowResponse(model_info={'qwen3.embedding_length': 2, 'qwen3.context_length': 32768})
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    return client


def test_persistent_commands_build_reopen_search_and_show_status(
    workspace, persistent_client, capsys, client_factory, tokenizer_download,
):
    assert main(['index', '--json', '--offline', '--context-length', '512']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['embedded_inputs'] == 3
    assert report['manifest']['embedding_spec']['model_revision'] == 'actual-digest'
    assert persistent_client.embed.call_args.kwargs['options'] == {'num_ctx': 512}
    persistent_client.embed.reset_mock()
    (workspace / 'example_notes' / 'habits.md').write_text('# Edited\nThis must not appear in a snapshot query.')
    assert main(['search', 'Question?', '--offline', '--json', '--source', 'habits.md']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['index_id'] == report['manifest']['index_version']
    assert result['method'] == result['results'][0]['method'] == 'semantic'
    assert result['results'][0]['score_type'] == 'cosine_similarity'
    assert result['results'][0]['source_id'] and result['results'][0]['chunk_id']
    assert result['results'][0]['content'] == 'A cue starts a habit.'
    assert persistent_client.embed.call_count == 1
    assert persistent_client.embed.call_args.kwargs['input'] == [
        'Instruct: Given a question, retrieve relevant notes that help answer it.\nQuery:Question?']
    assert persistent_client.embed.call_args.kwargs['options'] == {'num_ctx': 512}
    persistent_client.chat.assert_not_called()
    client_factory.reset_mock()
    tokenizer_download.reset_mock()
    assert main(['status', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['active_version'] == result['index_id']
    client_factory.assert_not_called()
    tokenizer_download.assert_not_called()


def test_search_checks_digest_and_missing_database_before_embedding(persistent_client, capsys):
    from ollama import ListResponse
    assert main(['search', 'Question?', '--json']) == 1
    persistent_client.embed.assert_not_called()
    assert main(['index', '--json', '--offline']) == 0
    capsys.readouterr()
    persistent_client.embed.reset_mock()
    persistent_client.list.return_value = ListResponse(models=[{'model': 'qwen3-embedding:0.6b', 'digest': 'changed'}])
    assert main(['search', 'Question?', '--offline']) == 1
    assert 'incompatible' in capsys.readouterr().err
    persistent_client.embed.assert_not_called()


@pytest.mark.parametrize('arguments', [
    ['index', '--json', '--context-length', '0'], ['index', '--json', '--max-retries', '9'],
    ['index', '--json', '--batch-size', '0'], ['index', '--json', '--chunk-overlap', '512'],
    ['search', ' '], ['search', 'Question?', '--top-k', '0'], ['status', '--vault-id', ' '],
])
def test_persistent_cli_rejects_invalid_arguments(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


def test_index_scan_failure_does_not_replace_the_previous_version(workspace, persistent_client, capsys):
    assert main(['index', '--json', '--offline']) == 0
    first = json.loads(capsys.readouterr().out)['manifest']['index_version']
    assert main(['index', '--json', '--notes-dir', 'missing', '--offline']) == 1
    capsys.readouterr()
    assert main(['status', '--json']) == 0
    assert json.loads(capsys.readouterr().out)['active_version'] == first


def test_index_reuse_force_and_empty_directory_preserve_lifecycle(workspace, persistent_client, capsys):
    assert main(['index', '--offline', '--json']) == 0
    first = json.loads(capsys.readouterr().out)
    assert main(['index', '--offline', '--json']) == 0
    reused = json.loads(capsys.readouterr().out)
    assert reused['reused_index'] and reused['embedded_inputs'] == 0
    assert reused['manifest']['index_version'] == first['manifest']['index_version']
    assert main(['index', '--offline', '--json', '--force']) == 0
    forced = json.loads(capsys.readouterr().out)
    assert forced['manifest']['index_version'] != first['manifest']['index_version']
    assert forced['embedded_inputs'] == 0 and forced['cached_inputs'] == 3
    for path in (workspace / 'example_notes').iterdir():
        path.unlink()
    assert main(['index', '--offline', '--json']) == 0
    empty = json.loads(capsys.readouterr().out)
    assert empty['manifest']['document_count'] == empty['manifest']['chunk_count'] == 0
    assert empty['deleted_documents'] == 3


def test_match_uses_saved_directory_and_live_content_from_another_cwd(
    indexed_client, workspace, monkeypatch, client_factory, capsys,
):
    db = workspace / '.arkb/index.sqlite'
    (workspace / 'example_notes/habits.md').write_text('# Edited\nRAG live RAG', encoding='utf-8')
    elsewhere = workspace / 'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    client_factory.reset_mock()
    assert main(['match', 'RAG', '--db', str(db), '--json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert [h['start_char'] for h in result['results']] == [0, 9]
    assert {h['source'] for h in result['results']} == {'habits.md'}
    assert result['index_id'] is None
    client_factory.assert_not_called()
    assert main(['match', 'RAG', '--db', str(db), '--notes-dir', str(elsewhere)]) == 1
    assert 'differs from the indexed knowledge base' in capsys.readouterr().err


def test_ask_runs_multiple_agent_selected_modes_on_one_snapshot(indexed_client, workspace, monkeypatch, capsys):
    from arkb.retrieval import BM25Retriever
    from tests.agent.helpers import ScriptedModel, reply, tool_call

    db = workspace / '.arkb/index.sqlite'
    captured_versions = []

    def semantic(runtime, storage, manifest, **kwargs):
        captured_versions.append(manifest.index_version)
        return BM25Retriever.from_snapshot(storage, vault_id=manifest.vault_id, index_version=manifest.index_version)

    monkeypatch.setattr(Runtime, 'semantic', semantic)

    def after_first_search(messages):
        hit, = json.loads(messages[-1]['content'])['results']
        assert hit['content'] == 'A cue starts a habit.'
        # Publish a new snapshot mid-run. Later modes must retain the old one;
        # read must see the current source, independently of either index.
        (workspace / 'example_notes/habits.md').write_text('# Edited\nCurrent habit body', encoding='utf-8')
        assert main(['index', '--offline', '--force', '--json']) == 0
        capsys.readouterr()
        return reply(calls=[tool_call('read', source='habits.md')])

    def after_read(messages):
        assert json.loads(messages[-1]['content'])['result']['content'] == 'Current habit body'
        return reply(calls=[tool_call('search', query='habit', mode='hybrid', source='habits.md')])

    def finish(messages):
        assert json.loads(messages[-1]['content'])['results'][0]['content'] == 'A cue starts a habit.'
        return reply('Found the saved evidence and checked the live note.')

    model = ScriptedModel(reply(calls=[tool_call('search', query='habit', mode='bm25', source='habits.md')]),
                           after_first_search, after_read, finish)
    with Runtime() as runtime:
        old_version = runtime.status(db=db)['active_version']
        result = runtime.ask('Find material', db=db, client=model, max_turns=4)
        assert runtime.status(db=db)['active_version'] != old_version
    assert result.stop_reason == 'final' and result.state.turn == 4
    assert [c['function']['name'] for c in result.state.tool_calls] == ['search', 'read', 'search']
    assert captured_versions == [old_version]


def test_ask_cli_keeps_json_clean_with_real_loop_and_bm25_only_calls(
    indexed_client, monkeypatch, client_factory, tokenizer_download, capsys,
):
    from tests.agent.helpers import ScriptedModel, reply, tool_call

    model = ScriptedModel(reply(calls=[tool_call('search', query='habit', mode='bm25')]), reply('Found habits.md'))
    monkeypatch.setattr(Runtime, 'model_client', lambda self: model)
    client_factory.reset_mock()
    tokenizer_download.reset_mock()
    assert main(['ask', 'Find habit material', '--max-turns', '2', '--trace', '--json']) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result['response'] == 'Found habits.md'
    assert result['state']['turn'] == 2 and result['stop_reason'] == 'final'
    assert '[1] search' in output.err and '[2] final' in output.err
    assert json.loads(result['state']['messages'][3]['content'])['results'][0]['source'] == 'habits.md'
    client_factory.assert_not_called()
    tokenizer_download.assert_not_called()


@pytest.mark.parametrize('failure', ['model', 'tool'])
@pytest.mark.parametrize('trace_enabled', [False, True])
@pytest.mark.parametrize('json_output', [False, True])
def test_ask_cli_shows_partial_trace_on_errors_without_changing_execution(
    monkeypatch, capsys, failure, trace_enabled, json_output,
):
    from tests.agent.helpers import reply, tool_call

    first = reply(calls=[tool_call('read', source='habits.md')])
    second = (ReadTimeout('model unavailable') if failure == 'model' else
              reply(calls=[tool_call('read', source='missing.md'), tool_call('read', source='never.md')]))
    model = Mock(chat=Mock(side_effect=[first, second]))
    monkeypatch.setattr(Runtime, 'model_client', lambda self: model)
    args = ['ask', 'Read notes'] + (['--trace'] if trace_enabled else []) + (['--json'] if json_output else [])
    assert main(args) == 1
    output = capsys.readouterr()
    assert output.out == '' and 'Error:' in output.err
    assert ('[1] read\nsource: "habits.md"' in output.err) is trace_enabled
    assert ('error\n' in output.err) is trace_enabled
    assert model.chat.call_count == 2
    observation = json.loads(model.chat.call_args.kwargs['messages'][-1]['content'])
    assert observation['result']['source'] == 'habits.md'


@pytest.mark.parametrize('mode', ['bm25', 'semantic', 'hybrid'])
def test_fixed_rag_python_api_keeps_snapshot_evidence_after_cli_migration(indexed_client, workspace, mode):
    from arkb.generation.context import build_context
    from arkb.generation.generate import generate_cited_answer
    from arkb.generation.models import ContextConfig, GenerationCounter

    (workspace / 'example_notes/permanent.md').write_text('# Changed\nDifferent live facts.', encoding='utf-8')
    counter = GenerationCounter('fake', 'fixture', lambda messages: 12 + sum(len(m['content']) for m in messages))
    with Runtime() as runtime:
        response = runtime.search('one idea', mode=mode, source='permanent.md')
        context = build_context('Q?', response.results, config=ContextConfig(), counter=counter)
        result = generate_cited_answer(context, client=indexed_client).to_dict()
    assert result['text'] and result['sources'][0]['content'] == 'Develop one idea per note.'
    assert result['sources'][0]['origins'][0]['index_version'] == response.index_id


@pytest.fixture(autouse=True)
def qdrant_connections(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient
    import warnings
    def connect(*args):
        return QdrantClient(path=str(tmp_path / 'qdrant'))
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', connect)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        yield


@pytest.fixture
def indexed_client(persistent_client, capsys):
    assert main(['index', '--json', '--offline']) == 0
    capsys.readouterr()
    persistent_client.embed.reset_mock()
    return persistent_client


@pytest.mark.parametrize('arguments', [[], ['Question?'], ['index', '--json', '--backend', 'numpy'],
                                       ['search', 'Q?', '--citation-mode', 'legacy']])
def test_retired_interfaces_are_rejected(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


@pytest.mark.parametrize('arguments', [['--help'], ['index', '--json', '--help'], ['search', '--help'], ['status', '--help']])
def test_help_needs_no_services(arguments, client_factory, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 0
    assert 'usage:' in capsys.readouterr().out
    client_factory.assert_not_called()


@pytest.mark.parametrize('operation,error', [
    ('embed', ConnectionError('Ollama is unavailable.')),
    ('embed', ReadTimeout('Ollama request timed out.')),
    ('embed', ReadError('Ollama connection was interrupted.')),
    ('embed', ResponseError('Model not found.', status_code=404)),
])
def test_search_reports_service_errors_without_an_answer(indexed_client, operation, error, capsys):
    getattr(indexed_client, operation).side_effect = error
    assert main(['search', 'Question?', '--offline']) == 1
    output = capsys.readouterr()
    assert output.out == '' and str(error) in output.err
    assert 'Traceback' not in output.err


def test_search_preserves_unicode_and_whitespace(indexed_client):
    question = "  为什么保留 e\u0301？\r\n"
    assert main(['search', question, '--offline']) == 0
    assert indexed_client.embed.call_args.kwargs['input'][0].endswith('Query:' + question)
    indexed_client.chat.assert_not_called()


def test_search_rejects_invalid_embedding_before_generation(indexed_client, capsys):
    indexed_client.embed.side_effect = [EmbedResponse(embeddings=[[0., 0.]])]
    assert main(['search', 'Q?', '--offline']) == 1
    assert capsys.readouterr().out == ''
    indexed_client.chat.assert_not_called()


@pytest.mark.parametrize('error', [FileNotFoundError('Tokenizer is not cached.'), ReadError('Download failed.')])
def test_index_tokenizer_failure_makes_no_model_calls(tokenizer_download, client_factory, error, capsys):
    tokenizer_download.side_effect = error
    assert main(['index', '--json', '--offline']) == 1
    assert str(error) in capsys.readouterr().err
    client_factory.assert_not_called()


def test_index_offline_cache_option(persistent_client, tokenizer_download):
    assert main(['index', '--json', '--offline', '--tokenizer-cache', 'cache']) == 0
    assert tokenizer_download.call_args.kwargs['local_files_only'] is True
    assert tokenizer_download.call_args.kwargs['cache_dir'] == Path('cache')


def test_search_rejects_retired_snapshot_before_loading_models(indexed_client, client_factory, capsys):
    from arkb.knowledge.sqlite import SQLiteStorage
    with SQLiteStorage(Path('.arkb/index.sqlite')) as storage:
        manifest = storage.active_manifest('default')
        storage.connection.execute("UPDATE builds SET backend=? WHERE version=?",
                                   (json.dumps({'kind': 'numpy'}), manifest.index_version))
        storage.connection.commit()
    client_factory.reset_mock()
    assert main(['search', 'Q?', '--json']) == 1
    assert 'run arkb index' in capsys.readouterr().err
    client_factory.assert_not_called()


@pytest.mark.parametrize('options', [
    ['--hnsw-m', '1'], ['--index-timeout', 'nan'], ['--full-scan-threshold', '9'],
    ['--require-hnsw', '--indexing-threshold', '0'],
])
def test_cli_qdrant_config_errors_precede_service_calls(options, client_factory):
    with pytest.raises(SystemExit) as error:
        main(['index', '--json', *options])
    assert error.value.code == 2
    client_factory.assert_not_called()


def test_cli_bm25_runs_without_model_or_vector_connections(indexed_client, client_factory,
                                                          tokenizer_download, monkeypatch, capsys):
    capsys.readouterr()
    client_factory.reset_mock()
    tokenizer_download.reset_mock()
    connect = Mock(side_effect=AssertionError('No Qdrant for lexical retrieval'))
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', connect)
    assert main(['search', 'habit', '--mode', 'bm25', '--json', '--top-k', '1']) == 0
    response = json.loads(capsys.readouterr().out)
    assert response['method'] == 'bm25'
    assert response['results'][0]['source'] == 'habits.md'
    assert response['results'][0]['score_type'] == 'bm25'
    client_factory.assert_not_called()
    tokenizer_download.assert_not_called()
    connect.assert_not_called()


@pytest.fixture
def qwen_scorer(monkeypatch):
    from types import SimpleNamespace
    scorer = SimpleNamespace(identity='Qwen/Qwen3-Reranker-0.6B/test-scorer',
        score_type='yes_no_logit_difference',
        score=lambda query, hits: [5. if 'habit' in hit.content else -1. for hit in hits])
    factory = Mock(return_value=scorer)
    monkeypatch.setattr('arkb.retrieval.qwen_rerank.QwenRerankerScorer', factory)
    return factory


@pytest.mark.parametrize('mode', ['semantic', 'bm25', 'hybrid'])
def test_cli_modes_support_optional_reranking_without_generation(indexed_client, mode, qwen_scorer, capsys):
    capsys.readouterr()
    args = ['search', 'habit', '--mode', mode, '--json', '--top-k', '1', '--offline']
    assert main(args) == 0
    before = json.loads(capsys.readouterr().out)
    assert before['method'] == mode
    qwen_scorer.assert_not_called()
    assert main(args + ['--rerank']) == 0
    after = json.loads(capsys.readouterr().out)
    assert after['method'] == before['method'] + '+rerank'
    hit = after['results'][0]
    assert hit['source'] == 'habits.md' and hit['score_type'] == 'yes_no_logit_difference'
    qwen_scorer.assert_called_once_with(max_length=512, cache_folder=None, local_files_only=True)
    assert hit['metadata']['rerank']['input_method'] == before['method']
    if mode == 'hybrid':
        assert hit['metadata']['fusion']['contributions']
    indexed_client.chat.assert_not_called()


@pytest.mark.parametrize('options', [
    ['--mode', 'hybrid', '--candidate-k', '1'], ['--rrf-k', '-1'],
    ['--rerank', '--rerank-candidates', '1'], ['--candidate-k', '0'],
])
def test_invalid_retrieval_depths_fail_before_model_calls(options, client_factory):
    with pytest.raises(SystemExit) as error:
        main(['search', 'question'] + options)
    assert error.value.code == 2
    client_factory.assert_not_called()


@pytest.mark.filterwarnings('ignore:Local mode performs exact.*')
def test_four_way_benchmark_runner_uses_saved_adapters_and_preserves_artifacts(indexed_client, tmp_path,
                                                                            monkeypatch, qwen_scorer, capsys):
    from qdrant_client import QdrantClient
    from arkb.evaluation.retrieval import baseline_main as compare
    capsys.readouterr()
    monkeypatch.setattr('ollama.Client', lambda **kw: indexed_client)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', lambda *a: QdrantClient(path=str(tmp_path / 'qdrant')))
    cases, output = tmp_path / 'cases.jsonl', tmp_path / 'results.json'
    cases.write_text(json.dumps({'id': 'habit', 'question': 'habit', 'relevance': {'habits.md': 3}}))
    args = ['--cases', str(cases), '--output', str(output), '--offline', '--top-k', '2',
            '--modes', 'semantic', 'bm25', 'hybrid', 'hybrid_reranked']
    assert compare(args) == 0
    report = json.loads(output.read_text())
    assert set(report['summary']) == {'semantic', 'bm25', 'hybrid', 'hybrid_reranked'}
    assert report['summary']['hybrid_reranked']['mrr'] == 1
    row = report['results'][0]['modes']['hybrid_reranked']
    assert row['response']['results'][0]['metadata']['rerank']['candidate_count'] == 3
    assert report['run']['cases'][0]['id'] == 'habit' and report['run']['source_hashes']
    before = output.read_bytes()
    with pytest.raises(SystemExit):
        compare(args)
    assert output.read_bytes() == before
