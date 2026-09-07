import json
from pathlib import Path
from unittest.mock import MagicMock, Mock

from httpx import ReadError, ReadTimeout
from ollama import ChatResponse, Client, EmbedResponse, Message, ResponseError
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers, processors

from obsidian_rag.cli import main


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
    monkeypatch.setattr("obsidian_rag.cli.Client", factory)
    return factory


def test_main_answers_using_the_most_relevant_notes(
    client: MagicMock, client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    status = main(["How should I write permanent notes?"])

    output = capsys.readouterr()
    assert status == 0
    assert output.out.startswith("Develop one idea per note. [1]\n")
    assert '[1] permanent.md' in output.out
    assert 'file://' not in output.out
    assert output.err == ""
    client_factory.assert_called_once_with(
        host="http://127.0.0.1:11434", timeout=180.0, trust_env=False
    )
    assert [call.kwargs for call in client.embed.call_args_list] == [
        {
            "model": "qwen3-embedding:0.6b",
            "input": [
                "Habit Stages\n\nA cue starts a habit.",
                "Literature Notes\n\nPreserve the author's meaning.",
                "Permanent Notes\n\nDevelop one idea per note.",
            ],
            "truncate": False,
        },
        {
            "model": "qwen3-embedding:0.6b",
            "input": [
                "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
                "Query:How should I write permanent notes?"
            ],
            "truncate": False,
        },
    ]
    request = client.chat.call_args.kwargs
    assert request["model"] == "qwen3.5:4b"
    assert json.loads(request["messages"][1]["content"]) == {
        "question": "How should I write permanent notes?",
        "notes": [
            {
                "title": "Permanent Notes",
                "content": "Develop one idea per note.",
                "source": "permanent.md",
                "source_id": "S1",
            },
            {
                "title": "Literature Notes",
                "content": "Preserve the author's meaning.",
                "source": "literature.md",
                "source_id": "S2",
            },
        ],
    }


def test_main_accepts_directory_retrieval_model_and_connection_options(
    workspace: Path,
    client: MagicMock,
    client_factory: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (workspace / "example_notes").rename(workspace / "other_notes")

    status = main([
        "How should I write permanent notes?",
        "--notes-dir", "other_notes",
        "--top-k", "1",
        "--embedding-model", "qwen3-embedding:4b",
        "--chunking", "none",
        "--generation-model", "another-local-model",
        "--host", "http://127.0.0.1:11435",
        "--timeout", "15.5",
    ])

    assert status == 0
    assert capsys.readouterr().err == ""
    client_factory.assert_called_once_with(
        host="http://127.0.0.1:11435", timeout=15.5, trust_env=False
    )
    assert [call.kwargs["model"] for call in client.embed.call_args_list] == [
        "qwen3-embedding:4b", "qwen3-embedding:4b"
    ]
    request = client.chat.call_args.kwargs
    assert request["model"] == "another-local-model"
    assert [
        note["source"] for note in json.loads(request["messages"][1]["content"])["notes"]
    ] == ["permanent.md"]


def test_main_preserves_query_whitespace_and_unicode_for_embedding_and_generation(
    client: MagicMock,
) -> None:
    question = "  为什么保留 e\u0301？\r\n"

    assert main([question]) == 0

    assert client.embed.call_args_list[1].kwargs["input"] == [
        "Instruct: Given a question, retrieve relevant notes that help answer it.\n"
        "Query:  为什么保留 e\u0301？\r\n"
    ]
    payload = json.loads(client.chat.call_args.kwargs["messages"][1]["content"])
    assert payload["question"] == question


def test_main_shows_help_without_connecting_to_ollama(
    client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr()
    assert "usage:" in output.out
    assert "--notes-dir" in output.out
    assert "--top-k" in output.out
    assert output.err == ""
    client_factory.assert_not_called()


@pytest.mark.parametrize(
    "arguments",
    [
        [],
        [""],
        [" \n\t"],
        ["Question?", "--top-k", "0"],
        ["Question?", "--top-k", "-1"],
        ["Question?", "--top-k", "1.5"],
        ["Question?", "--timeout", "0"],
        ["Question?", "--timeout", "-1"],
        ["Question?", "--timeout", "nan"],
        ["Question?", "--timeout", "inf"],
        ["Question?", "--unknown"],
    ],
    ids=[
        "missing-question", "empty-question", "whitespace-question",
        "zero-top-k", "negative-top-k", "fractional-top-k",
        "zero-timeout", "negative-timeout", "nan-timeout", "infinite-timeout",
        "unknown-option",
    ],
)
def test_main_rejects_invalid_arguments_before_connecting_to_ollama(
    arguments: list[str], client_factory: Mock, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(arguments)

    assert exit_info.value.code == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert "error:" in output.err
    client_factory.assert_not_called()


@pytest.mark.parametrize("directory", ["missing", "plain.txt", "empty"])
def test_main_reports_unusable_note_directories_without_connecting_to_ollama(
    workspace: Path,
    directory: str,
    client_factory: Mock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (workspace / "plain.txt").write_text("Not a directory.", encoding="utf-8")
    (workspace / "empty").mkdir()

    status = main(["Question?", "--notes-dir", directory])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert "Traceback" not in output.err
    client_factory.assert_not_called()


@pytest.mark.parametrize(
    ("operation", "error"),
    [
        ("embed", ConnectionError("Ollama is unavailable.")),
        ("embed", ReadTimeout("Ollama request timed out.")),
        ("chat", ReadError("Ollama connection was interrupted.")),
        ("chat", ResponseError("Model not found.", status_code=404)),
    ],
    ids=["connection", "timeout", "read-error", "missing-model"],
)
def test_main_reports_service_errors_without_printing_an_answer(
    client: MagicMock,
    operation: str,
    error: Exception,
    capsys: pytest.CaptureFixture[str],
) -> None:
    getattr(client, operation).side_effect = error

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert str(error) in output.err
    assert "Traceback" not in output.err


def test_main_reports_invalid_embeddings_before_generating_an_answer(
    client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    client.embed.side_effect = [EmbedResponse(embeddings=[[1.0, 0.0]])]

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "Error:" in output.err
    assert "one nonempty embedding vector per input text" in output.err
    client.chat.assert_not_called()


def test_main_reports_an_empty_model_answer(
    client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    client.chat.return_value = ChatResponse(
        message=Message(role="assistant", content=" \n")
    )

    status = main(["Question?"])

    assert status == 1
    output = capsys.readouterr()
    assert output.out == ""
    assert "invalid_structure" in output.err


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
    monkeypatch.setattr("obsidian_rag.tokenization.hf_hub_download", download)
    return download


def test_main_embeds_chunks_and_generates_from_the_selected_passage(
    workspace: Path, client: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    directory = workspace / "long_notes"
    directory.mkdir()
    (directory / "long.md").write_text("# Long\n\naaaa\n\nbbbb\n\ncccc")
    client.embed.side_effect = [
        EmbedResponse(embeddings=[[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]]),
        EmbedResponse(embeddings=[[1.0, 0.0]]),
    ]

    status = main(["Which passage?", "--notes-dir", "long_notes", "--top-k", "1",
                   "--chunk-size", "6", "--chunk-overlap", "0"])

    assert status == 0
    assert client.embed.call_args_list[0].kwargs["input"] == [
        "Long\n\naaaa\n\n", "Long\n\nbbbb\n\n", "Long\n\ncccc",
    ]
    payload = json.loads(client.chat.call_args.kwargs["messages"][1]["content"])
    assert payload["notes"] == [{"title": "Long", "content": "bbbb\n\n", "source": "long.md", "source_id": "S1"}]
    assert capsys.readouterr().err == ""


def test_main_whole_note_mode_does_not_load_a_tokenizer(
    tokenizer_download: Mock, client: MagicMock,
) -> None:
    status = main(["A question?", "--chunking", "none",
                   "--embedding-model", "qwen3-embedding:4b"])

    assert status == 0
    tokenizer_download.assert_not_called()
    assert len(client.embed.call_args_list[0].kwargs["input"]) == 3


@pytest.mark.parametrize("options", [
    ["--chunk-size", "0"],
    ["--chunk-overlap", "-1"],
    ["--chunk-size", "64", "--chunk-overlap", "64"],
    ["--embedding-model", "another-model"],
])
def test_main_rejects_invalid_chunking_before_download_or_model_calls(
    options: list[str], tokenizer_download: Mock, client_factory: Mock,
) -> None:
    with pytest.raises(SystemExit) as error:
        main(["A question?", *options])
    assert error.value.code == 2
    tokenizer_download.assert_not_called()
    client_factory.assert_not_called()


def test_main_supports_an_offline_tokenizer_cache(
    workspace: Path, tokenizer_download: Mock,
) -> None:
    status = main(["A question?", "--offline", "--tokenizer-cache", "my_cache"])

    assert status == 0
    assert tokenizer_download.call_args.kwargs["local_files_only"] is True
    assert tokenizer_download.call_args.kwargs["cache_dir"] == Path("my_cache")


@pytest.mark.parametrize("error", [FileNotFoundError("Tokenizer is not cached."),
                                  ReadError("Tokenizer download failed.")])
def test_main_reports_tokenizer_failures_before_model_calls(
    tokenizer_download: Mock, client_factory: Mock,
    capsys: pytest.CaptureFixture[str], error: Exception,
) -> None:
    tokenizer_download.side_effect = error

    assert main(["A question?", "--offline"]) == 1
    assert str(error) in capsys.readouterr().err
    client_factory.assert_not_called()


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
    from obsidian_rag.context import GenerationCounter
    def load(**kwargs):
        return GenerationCounter(kwargs['model'], 'test-counter',
                                 lambda messages: 12 + sum(len(m['content']) for m in messages))
    factory = Mock(side_effect=load)
    monkeypatch.setattr("obsidian_rag.cli.load_generation_counter", factory)
    return factory


def test_show_context_reports_final_payload_budget_and_memory_provenance(client, capsys, generation_counter_adapter):
    assert main(['Q?', '--show-context', '--offline', '--context-window', '2048',
                 '--max-output-tokens', '128', '--context-safety-margin', '64']) == 0
    report = json.loads(capsys.readouterr().out)
    assert report['status'] == 'ready'
    assert report['config'] == {'context_window': 2048, 'max_output_tokens': 128, 'safety_margin': 64}
    assert report['token_usage']['prompt_tokens'] <= 1856
    assert report['evidence_blocks'][0]['origins'][0]['index_version'].startswith('memory:')
    assert report['citation_map'] == {'permanent.md': [0], 'literature.md': [1]}
    assert generation_counter_adapter.call_args.kwargs['local_files_only'] is True
    client.chat.assert_not_called()


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
    assert context['evidence_blocks'][0]['content'] == 'A cue starts a habit.'
    assert context['evidence_blocks'][0]['origins'][0]['index_version'] == version
    persistent_client.chat.assert_not_called()


@pytest.mark.parametrize('arguments', [
    ['Q?', '--context-window', '0'], ['Q?', '--max-output-tokens', '0'],
    ['Q?', '--context-safety-margin', '-1'], ['Q?', '--context-window', '100'],
    ['query', 'Q?', '--context-window', '0'], ['query', 'Q?', '--json', '--show-context'],
    ['Q?', '--answer-json', '--show-context'], ['query', 'Q?', '--json', '--answer-json'],
    ['Q?', '--answer-json', '--citation-mode', 'legacy'],
])
def test_context_argument_errors_happen_before_model_calls(arguments, client_factory):
    with pytest.raises(SystemExit) as error:
        main(arguments)
    assert error.value.code == 2
    client_factory.assert_not_called()


def test_cli_reports_fixed_prompt_overflow_without_generating(client, capsys):
    assert main(['Q?', '--context-window', '300', '--max-output-tokens', '100',
                 '--context-safety-margin', '0']) == 1
    output = capsys.readouterr()
    assert 'before adding evidence' in output.err
    assert output.out == ''
    client.chat.assert_not_called()


def test_memory_answer_json_contains_only_cited_sources_and_validation(client, capsys):
    assert main(['Q?', '--answer-json']) == 0
    output = json.loads(capsys.readouterr().out)
    assert output['answer']['claims'][0]['source_ids'] == ['S1']
    assert len(output['sources']) == 1
    assert output['sources'][0]['source'] == 'permanent.md'
    assert output['sources'][0]['origins'][0]['index_version'].startswith('memory:')
    assert output['validation']['support_status'] == 'not_checked'
    assert output['raw_response'] == client.chat.return_value.message.content


def test_persistent_answer_json_keeps_saved_source_after_live_file_changes(workspace, persistent_client, capsys):
    assert main(['index', '--offline']) == 0
    version = json.loads(capsys.readouterr().out)['manifest']['index_version']
    (workspace / 'example_notes' / 'habits.md').write_text('# Changed\nDifferent facts.')
    assert main(['query', 'Q?', '--offline', '--source', 'habits.md', '--answer-json']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['sources'][0]['content'] == 'A cue starts a habit.'
    assert result['sources'][0]['origins'][0]['index_version'] == version


def test_cli_legacy_protocol_remains_explicitly_available(client, capsys):
    client.chat.return_value.message.content = 'A legacy answer. [permanent.md]'
    assert main(['Q?', '--citation-mode', 'legacy']) == 0
    assert capsys.readouterr().out == 'A legacy answer. [permanent.md]\n'
    assert 'format' not in client.chat.call_args.kwargs


def test_cli_invalid_citation_fails_without_printing_the_unverified_answer(client, capsys):
    client.chat.return_value.message.content = json.dumps({'status': 'answered', 'claims': [
        {'text': 'Unverified content', 'source_ids': ['S99']}], 'missing_information': []})
    assert main(['Q?', '--answer-json']) == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert 'invalid_references' in output.err
    assert 'Unverified content' not in output.err
