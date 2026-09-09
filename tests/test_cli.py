import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

from httpx import ReadError, ReadTimeout
from ollama import ChatResponse, Client, EmbedResponse, Message, ResponseError
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from arkb.cli import main


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
    monkeypatch.setattr("arkb.cli.Client", factory)
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
    monkeypatch.setattr("arkb.tokenization.hf_hub_download", download)
    return download


@pytest.fixture
def persistent_client(client):
    from ollama import ListResponse, ShowResponse
    client.list.return_value = ListResponse(models=[{'model': 'qwen3-embedding:0.6b', 'digest': 'actual-digest'}])
    client.show.return_value = ShowResponse(model_info={'qwen3.embedding_length': 2, 'qwen3.context_length': 32768})
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    return client


def test_persistent_commands_build_reopen_query_and_show_status(
    workspace, persistent_client, capsys, client_factory, tokenizer_download,
):
    assert main(['index', '--offline', '--context-length', '512']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['embedded_inputs'] == 3
    assert report['manifest']['embedding_spec']['model_revision'] == 'actual-digest'
    assert persistent_client.embed.call_args.kwargs['options'] == {'num_ctx': 512}
    persistent_client.embed.reset_mock()
    (workspace / 'example_notes' / 'habits.md').write_text('# Edited\nThis must not appear in a snapshot query.')
    assert main(['query', 'Question?', '--offline', '--json', '--source', 'habits.md']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['index_version'] == report['manifest']['index_version']
    assert result['results'][0]['chunk']['content'] == 'A cue starts a habit.'
    assert persistent_client.embed.call_count == 1
    assert persistent_client.embed.call_args.kwargs['input'] == [
        'Instruct: Given a question, retrieve relevant notes that help answer it.\nQuery:Question?']
    assert persistent_client.embed.call_args.kwargs['options'] == {'num_ctx': 512}
    persistent_client.chat.assert_not_called()
    client_factory.reset_mock()
    tokenizer_download.reset_mock()
    assert main(['status']) == 0
    assert json.loads(capsys.readouterr().out)['active_version'] == result['index_version']
    client_factory.assert_not_called()
    tokenizer_download.assert_not_called()


def test_query_checks_digest_and_missing_database_before_embedding(persistent_client, capsys):
    from ollama import ListResponse
    assert main(['query', 'Question?', '--json']) == 1
    persistent_client.embed.assert_not_called()
    assert main(['index', '--offline']) == 0
    capsys.readouterr()
    persistent_client.embed.reset_mock()
    persistent_client.list.return_value = ListResponse(models=[{'model': 'qwen3-embedding:0.6b', 'digest': 'changed'}])
    assert main(['query', 'Question?', '--offline']) == 1
    assert 'incompatible' in capsys.readouterr().err
    persistent_client.embed.assert_not_called()


@pytest.mark.parametrize('arguments', [
    ['index', '--context-length', '0'], ['index', '--max-retries', '9'],
    ['index', '--batch-size', '0'], ['index', '--chunk-overlap', '512'],
    ['query', ' '], ['query', 'Question?', '--top-k', '0'], ['status', '--vault-id', ' '],
])
def test_persistent_cli_rejects_invalid_arguments(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


def test_index_scan_failure_does_not_replace_the_previous_version(workspace, persistent_client, capsys):
    assert main(['index', '--offline']) == 0
    first = json.loads(capsys.readouterr().out)['manifest']['index_version']
    assert main(['index', '--notes-dir', 'missing', '--offline']) == 1
    capsys.readouterr()
    assert main(['status']) == 0
    assert json.loads(capsys.readouterr().out)['active_version'] == first


def test_query_can_generate_from_the_saved_snapshot(persistent_client, capsys):
    assert main(['index', '--offline']) == 0
    capsys.readouterr()
    assert main(['query', 'Question?', '--offline']) == 0
    assert 'Develop one idea' in capsys.readouterr().out
    assert json.loads(persistent_client.chat.call_args.kwargs['messages'][1]['content'])['question'] == 'Question?'


@pytest.fixture(autouse=True)
def generation_counter_adapter(monkeypatch):
    from arkb.context import GenerationCounter
    def load(**kwargs):
        return GenerationCounter(kwargs['model'], 'test-counter',
                                 lambda messages: 12 + sum(len(m['content']) for m in messages))
    factory = Mock(side_effect=load)
    monkeypatch.setattr("arkb.cli.load_generation_counter", factory)
    return factory


def test_persistent_show_context_uses_saved_revision_and_json_remains_retrieval_only(
    workspace, persistent_client, capsys, generation_counter_adapter,
):
    assert main(['index', '--offline']) == 0
    version = json.loads(capsys.readouterr().out)['manifest']['index_version']
    assert main(['query', 'Q?', '--offline', '--json']) == 0
    capsys.readouterr()
    generation_counter_adapter.assert_not_called()
    (workspace / 'example_notes' / 'habits.md').write_text('# Changed\nNew content.')
    assert main(['query', 'Q?', '--source', 'habits.md', '--offline', '--show-context']) == 0
    context = json.loads(capsys.readouterr().out)
    assert 'evidence_blocks' not in context and 'citation_map' not in context
    assert context['citation_sources'][0]['source_id'] == 'S1'
    assert context['citation_sources'][0]['content'] == 'A cue starts a habit.'
    assert context['citation_sources'][0]['origins'][0]['index_version'] == version
    persistent_client.chat.assert_not_called()


@pytest.mark.parametrize('arguments', [
    ['query', 'Q?', '--context-window', '0'], ['query', 'Q?', '--max-output-tokens', '0'],
    ['query', 'Q?', '--context-safety-margin', '-1'], ['query', 'Q?', '--context-window', '100'],
    ['query', 'Q?', '--json', '--show-context'],
    ['query', 'Q?', '--answer-json', '--show-context'], ['query', 'Q?', '--json', '--answer-json'],
    ['query', 'Q?', '--answer-json', '--citation-mode', 'legacy'],
])
def test_context_argument_errors_happen_before_model_calls(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


def test_cli_reports_fixed_prompt_overflow_without_generating(indexed_client, client, capsys):
    assert main(['query', 'Q?', '--context-window', '300', '--max-output-tokens', '100',
                 '--context-safety-margin', '0']) == 1
    output = capsys.readouterr()
    assert 'before adding evidence' in output.err
    assert output.out == ''
    client.chat.assert_not_called()


def test_persistent_answer_json_keeps_saved_source_after_live_file_changes(workspace, persistent_client, capsys):
    assert main(['index', '--offline']) == 0
    version = json.loads(capsys.readouterr().out)['manifest']['index_version']
    (workspace / 'example_notes' / 'habits.md').write_text('# Changed\nDifferent facts.')
    assert main(['query', 'Q?', '--offline', '--source', 'habits.md', '--answer-json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['sources'][0]['content'] == 'A cue starts a habit.'
    assert result['sources'][0]['origins'][0]['index_version'] == version


def test_cli_invalid_citation_fails_without_printing_the_unverified_answer(indexed_client, client, capsys):
    client.chat.return_value.message.content = json.dumps({'status': 'answered', 'claims': [
        {'text': 'Unverified content', 'source_ids': ['S99']}], 'missing_information': []})
    assert main(['query', 'Q?', '--answer-json']) == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert 'invalid_references' in output.err
    assert 'Unverified content' not in output.err


def test_cli_quoted_answer_json_exposes_program_computed_offsets(indexed_client, client, capsys):
    client.chat.return_value.message.content = json.dumps({'status': 'answered', 'claims': [
        {'text': 'Develop one idea per note.', 'source_ids': ['S1'],
         'quotes': [{'source_id': 'S1', 'text': 'one idea'}]}], 'missing_information': []})
    assert main(['query', 'Q?', '--source', 'permanent.md', '--answer-json', '--citation-mode', 'quoted']) == 0
    result = json.loads(capsys.readouterr().out)
    quote = result['validation']['resolved_quotes'][0]
    assert (quote['start_char'], quote['end_char']) == (8, 16)


@pytest.fixture(autouse=True)
def qdrant_connections(tmp_path, monkeypatch):
    from qdrant_client import QdrantClient
    import warnings
    def connect(*args):
        return QdrantClient(path=str(tmp_path / 'qdrant'))
    monkeypatch.setattr('arkb.cli.connect_qdrant', connect)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        yield


@pytest.fixture
def indexed_client(persistent_client, capsys):
    assert main(['index', '--offline']) == 0
    capsys.readouterr()
    persistent_client.embed.reset_mock()
    return persistent_client


@pytest.mark.parametrize('arguments', [[], ['Question?'], ['index', '--backend', 'numpy'],
                                       ['query', 'Q?', '--citation-mode', 'legacy']])
def test_retired_interfaces_are_rejected(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


@pytest.mark.parametrize('arguments', [['--help'], ['index', '--help'], ['query', '--help'], ['status', '--help']])
def test_help_needs_no_services(arguments, client_factory, capsys):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 0
    assert 'usage:' in capsys.readouterr().out
    client_factory.assert_not_called()


@pytest.mark.parametrize('operation,error', [
    ('embed', ConnectionError('Ollama is unavailable.')),
    ('embed', ReadTimeout('Ollama request timed out.')),
    ('chat', ReadError('Ollama connection was interrupted.')),
    ('chat', ResponseError('Model not found.', status_code=404)),
])
def test_query_reports_service_errors_without_an_answer(indexed_client, operation, error, capsys):
    getattr(indexed_client, operation).side_effect = error
    assert main(['query', 'Question?', '--offline']) == 1
    output = capsys.readouterr()
    assert output.out == '' and str(error) in output.err
    assert 'Traceback' not in output.err


def test_query_preserves_unicode_and_whitespace(indexed_client):
    question = "  为什么保留 e\u0301？\r\n"
    assert main(['query', question, '--offline']) == 0
    assert indexed_client.embed.call_args.kwargs['input'][0].endswith('Query:' + question)
    assert json.loads(indexed_client.chat.call_args.kwargs['messages'][1]['content'])['question'] == question


@pytest.mark.parametrize('body', ['', '   ', '{'])
def test_query_rejects_invalid_generated_structure(indexed_client, body, capsys):
    indexed_client.chat.return_value.message.content = body
    assert main(['query', 'Q?', '--offline']) == 1
    output = capsys.readouterr()
    assert output.out == '' and 'invalid_structure' in output.err


def test_query_rejects_invalid_embedding_before_generation(indexed_client, capsys):
    indexed_client.embed.side_effect = [EmbedResponse(embeddings=[[0., 0.]])]
    assert main(['query', 'Q?', '--offline']) == 1
    assert capsys.readouterr().out == ''
    indexed_client.chat.assert_not_called()


@pytest.mark.parametrize('error', [FileNotFoundError('Tokenizer is not cached.'), ReadError('Download failed.')])
def test_index_tokenizer_failure_makes_no_model_calls(tokenizer_download, client_factory, error, capsys):
    tokenizer_download.side_effect = error
    assert main(['index', '--offline']) == 1
    assert str(error) in capsys.readouterr().err
    client_factory.assert_not_called()


def test_index_offline_cache_option(persistent_client, tokenizer_download):
    assert main(['index', '--offline', '--tokenizer-cache', 'cache']) == 0
    assert tokenizer_download.call_args.kwargs['local_files_only'] is True
    assert tokenizer_download.call_args.kwargs['cache_dir'] == Path('cache')


def test_query_rejects_retired_snapshot_before_loading_models(indexed_client, client_factory, capsys):
    from arkb.storage import SQLiteStorage
    with SQLiteStorage(Path('.obsidian-rag/index.sqlite')) as storage:
        manifest = storage.active_manifest('default')
        storage.connection.execute("UPDATE builds SET backend=? WHERE version=?",
                                   (json.dumps({'kind': 'numpy'}), manifest.index_version))
        storage.connection.commit()
    client_factory.reset_mock()
    assert main(['query', 'Q?', '--json']) == 1
    assert 'run arkb index' in capsys.readouterr().err
    client_factory.assert_not_called()


@pytest.mark.parametrize('options', [
    ['--hnsw-m', '1'], ['--index-timeout', 'nan'], ['--full-scan-threshold', '9'],
    ['--require-hnsw', '--indexing-threshold', '0'],
])
def test_cli_qdrant_config_errors_precede_service_calls(options, client_factory):
    with pytest.raises(SystemExit) as error:
        main(['index', *options])
    assert error.value.code == 2
    client_factory.assert_not_called()
