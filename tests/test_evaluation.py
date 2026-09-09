from functools import partial

import numpy as np
import pytest

from arkb.knowledge.models import Chunk
from arkb.knowledge.chunking import whole_note_chunks
from arkb.evaluation import compare_retrieval, evidence_statistics, recall_at_k
from arkb.knowledge.models import Note
from arkb.knowledge.qdrant import search_qdrant
from arkb.knowledge.models import ChunkRecord, EmbeddingSpec


def test_neighbor_recall_counts_unique_exact_neighbors_and_handles_no_reference():
    assert recall_at_k(['a', 'b'], ['b', 'c'], 2) == .5
    assert recall_at_k(['a'], ['a'], 10) == 1
    assert recall_at_k([], [], 2) is None
    with pytest.raises(ValueError):
        recall_at_k(['a'], ['a', 'a'], 2)


def test_section_coverage_uses_union_and_not_just_file_hits():
    chunks = [Chunk(content='x' * 6, title='T', source='a.md', chunk_index=i,
                    start_char=start, end_char=start + 6) for i, start in enumerate((0, 4))]
    case = {'required_source_groups': [['a.md', 'equivalent.md'], ['b.md']],
            'evidence_anchors': [{'source': 'a.md', 'body_start_char': 0, 'body_end_char': 12}]}
    metrics = evidence_statistics(chunks, case)
    assert metrics['source_group_recall'] == .5
    assert metrics['section_coverage'] == pytest.approx(10 / 12)
    assert metrics['all_sections_complete'] is False
    assert evidence_statistics([], {})['section_coverage'] is None
    with pytest.raises(ValueError, match='body coordinates'):
        evidence_statistics(chunks, {'evidence_anchors': [{'source': 'a.md', 'start_char': 0, 'end_char': 12}]})


def test_comparison_separates_neighbor_recall_from_evidence_coverage(qdrant):
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2, document_template='title-body-v1')
    notes = [Note(title='T', content='evidence', source=f'{i}.md') for i in range(2)]
    records = [ChunkRecord.from_note(whole_note_chunks([n])[0], note=n, vault_id='v') for n in notes]
    vectors = np.eye(2)
    from arkb.knowledge.qdrant import QdrantIndex
    index = QdrantIndex(qdrant, 'evaluation', spec, vault_id='v', create=True)
    index.upsert(records, vectors)
    search = partial(search_qdrant, qdrant, 'evaluation', spec=spec, vault_id='v')
    cases = [{'id': 'q1', 'question': 'Question?', 'required_source_groups': [['1.md']],
              'evidence_anchors': [{'source': '1.md', 'body_start_char': 0, 'body_end_char': 8}]}]
    result = compare_retrieval(records, vectors, [[1, 0]], cases, spec=spec, search=search, top_k=1)
    assert result['summary']['qdrant_exact']['neighbor_recall_at_k'] == 1
    assert result['summary']['qdrant_exact']['section_coverage'] == 0
    assert result['summary']['qdrant_ann']['neighbor_recall_at_k'] == 1
    assert result['summary']['qdrant_ann']['section_coverage'] == 0
    assert result['settings']['vector_bytes'] == vectors.nbytes
    with pytest.raises(ValueError, match='unique'):
        compare_retrieval(records, vectors, [[1, 0], [1, 0]], cases * 2, spec=spec, search=search)


def test_context_evaluation_measures_packed_span_union_and_coverage_retention():
    from arkb.context import ContextConfig, GenerationCounter, build_context
    from arkb.evaluation import evaluate_context
    from arkb.retrieval.semantic import snapshot_result
    note = Note('Title', 'x' * 300, 'a.md')
    hits = []
    for index, (start, end) in enumerate([(0, 200), (150, 300)]):
        chunk = Chunk(note.content[start:end], note.title, note.source, index, start, end)
        record = ChunkRecord.from_note(chunk, note=note, vault_id='v')
        hits.append(snapshot_result(record, .9 - index / 10, 'snapshot'))
    counter = GenerationCounter('test', 'test-count', lambda messages: 10 + sum(len(m['content']) for m in messages))
    limit = counter(build_context('Q?', hits).messages)
    config = ContextConfig(limit + 20, 20, 0)
    case = {'required_source_groups': [['a.md']],
            'evidence_anchors': [{'source': 'a.md', 'body_start_char': 0, 'body_end_char': 300}]}
    result = evaluate_context('Q?', hits, case, config=config, counter=counter)
    built = result['metrics']
    assert built['section_coverage'] == built['section_coverage_retention'] == 1
    assert built['duplicate_span_fraction'] == 0
    assert built['fits_budget'] and built['prompt_tokens'] == limit
    assert built['body_characters'] == 300 and built['block_count'] == 1
    assert [(s['source_id'], s['source'], s['content']) for s in result['context']['citation_sources']] == [('S1', 'a.md', note.content)]
    from dataclasses import replace
    with pytest.raises(ValueError, match='one snapshot'):
        evaluate_context('Q?', [hits[0], replace(hits[1], metadata={**hits[1].metadata, 'index_version': 'other'})], case, config=config, counter=counter)


def citation_fixture():
    from arkb.context import ContextConfig, GenerationCounter, build_context
    from arkb.retrieval.semantic import snapshot_result
    note = Note('Title', 'Only small datasets were faster.', 'a.md')
    chunk = whole_note_chunks([note])[0]
    counter = GenerationCounter('test', 'chars', lambda m: 10 + sum(len(x['content']) for x in m))
    return build_context('Which datasets were faster?', [snapshot_result(ChunkRecord.from_note(chunk, note=note, vault_id='v'), .9, 'snapshot')],
                          config=ContextConfig(), counter=counter, citation_mode='structured')


def test_citation_metrics_separate_valid_links_from_wrong_claims_and_missing_facts():
    import json
    from arkb.evaluation import citation_statistics
    raw = json.dumps({'status': 'answered', 'claims': [
        {'text': 'Every dataset was faster.', 'source_ids': ['S1']},
        {'text': 'It was also cheaper.', 'source_ids': []}], 'missing_information': []})
    sources = citation_fixture().citation_sources
    unreviewed = citation_statistics(raw, sources)
    assert unreviewed['citation_id_validity'] == 1
    assert unreviewed['claim_reference_coverage'] == .5
    assert unreviewed['references_valid'] is False
    assert unreviewed['supported_claim_rate'] is None
    review = {'reviewer': 'fixture', 'claim_support': ['contradicted', 'insufficient'],
              'answer_correct': False, 'answer_complete': False}
    checked = citation_statistics(raw, sources, review=review)
    assert checked['supported_claim_rate'] == 0 and checked['support_review_coverage'] == 1
    assert checked['answer_correct'] is False
    with pytest.raises(ValueError, match='valid source'):
        citation_statistics(raw, sources, review={**review, 'claim_support': ['contradicted', 'supported']})


def test_citation_metrics_keep_undefined_denominators_and_partial_review_visible():
    import json
    from arkb.evaluation import citation_statistics
    sources = citation_fixture().citation_sources
    abstain = json.dumps({'status': 'insufficient_evidence', 'claims': [], 'missing_information': ['No evidence.']})
    metrics = citation_statistics(abstain, sources)
    assert metrics['citation_id_validity'] is metrics['claim_reference_coverage'] is None
    assert citation_statistics('{', sources)['claim_count'] is None
    raw = json.dumps({'status': 'answered', 'claims': [
        {'text': 'A', 'source_ids': ['S1']}, {'text': 'B', 'source_ids': ['S1']}], 'missing_information': []})
    checked = citation_statistics(raw, sources, review={'reviewer': 'fixture', 'claim_support': ['supported', None]})
    assert checked['supported_claim_rate'] == 1 and checked['support_review_coverage'] == .5
    with pytest.raises(ValueError, match='one support label'):
        citation_statistics(raw, sources, review={'reviewer': 'fixture', 'claim_support': ['supported']})


def test_citation_evaluation_retains_failed_raw_output_and_counts_failures():
    from unittest.mock import Mock
    from ollama import Client, ChatResponse, Message
    from arkb.evaluation import evaluate_citation_context, summarize_citations
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content='{'), done_reason='length')
    row = evaluate_citation_context(citation_fixture(), client=client)
    assert row['success'] is False and row['raw_response'] == '{'
    assert row['error']['code'] == 'truncated_output'
    assert row['context']['citation_sources'][0]['content'] == 'Only small datasets were faster.'
    summary = summarize_citations([row])
    assert summary['error_count'] == 1 and summary['structure_valid'] == 0
    assert summary['supported_claim_rate'] is None
    assert summary['citation_id_validity_defined_cases'] == 0


def test_citation_case_records_prompt_budget_failure_without_calling_model():
    from unittest.mock import Mock
    from arkb.context import ContextConfig
    from arkb.evaluation import evaluate_citation_case
    context = citation_fixture()
    client = Mock()
    row = evaluate_citation_case('Question?', list(context.evidence_blocks[0].origins),
                                 config=ContextConfig(30, 10, 0), counter=context.counter, client=client)
    assert row['error']['code'] == 'context_budget' and row['context'] is None
    client.chat.assert_not_called()


def test_runner_hashes_nested_sources_and_preserves_snapshot_and_artifacts(tmp_path, monkeypatch, capsys, qdrant, qdrant_config):
    import hashlib
    import json
    from unittest.mock import MagicMock

    from ollama import Client, ChatResponse, EmbedResponse, Message
    from tokenizers import Tokenizer, models, pre_tokenizers

    import arkb.evaluation as evaluation
    from arkb.context import GenerationCounter
    from arkb.knowledge.indexing import build_index
    from arkb.knowledge.sqlite import SQLiteStorage

    # Identical basenames in different packages must retain distinct identities.
    package_dir = tmp_path / 'src' / 'arkb'
    sources = {
        '__init__.py': b'"""Package."""\n',
        'evaluation.py': b'"""Evaluation."""\n',
        'indexing/__init__.py': b'"""Indexing."""\n',
        'retrieval/__init__.py': b'"""Retrieval."""\n',
        'retrieval/semantic.py': b'"""Semantic search."""\n',
        'context/builder.py': b'"""Context construction."""\n',
    }
    for name, data in sources.items():
        path = package_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (package_dir / 'unrelated.txt').write_text('Not Python source.')
    monkeypatch.setattr(evaluation, '__file__', str(package_dir / 'evaluation.py'))

    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='test', model_revision='fixed', dimensions=2,
                         document_template='title-body-v1')
    client = MagicMock(spec=Client)
    client.__enter__.return_value = client
    client.embed.side_effect = lambda **kw: EmbedResponse(embeddings=[[1., 0.] for _ in kw['input']])
    client.chat.return_value = ChatResponse(message=Message(role='assistant', content=json.dumps({
        'status': 'answered', 'claims': [{'text': 'A fact.', 'source_ids': ['S1']}],
        'missing_information': [],
    })))
    db = tmp_path / 'index.sqlite'
    with SQLiteStorage(db) as storage:
        build_index(storage, [Note('Title', 'A fact.', 'a.md')], spec=spec, vault_id='default',
                    client=client, tokenizer=tokenizer, max_input_tokens=100, chunking='none',
                    qdrant_client=qdrant, qdrant_config=qdrant_config)
    monkeypatch.setattr('arkb.knowledge.qdrant.connect_qdrant', lambda *a: qdrant)
    before = db.read_bytes()
    client.embed.reset_mock()
    monkeypatch.setattr('ollama.Client', lambda **kw: client)
    monkeypatch.setattr('arkb.knowledge.embeddings.resolve_embedding_spec', lambda *a, **kw: spec)
    monkeypatch.setattr('arkb.knowledge.embeddings.load_tokenizer', lambda **kw: tokenizer)
    counter = GenerationCounter('test-generation', 'test-count',
                                 lambda messages: 10 + sum(len(m['content']) for m in messages))
    monkeypatch.setattr('arkb.generation.load_generation_counter', lambda **kw: counter)
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'fact', 'question': 'What is stated?'}) + '\n')
    output = tmp_path / 'evaluation'
    args = ['--db', str(db), '--cases', str(cases), '--output', str(output),
            '--offline', '--context', '--citations']

    assert evaluation.main(args) == 0
    capsys.readouterr()
    metadata = json.loads((output / 'run_metadata.json').read_text())
    assert metadata['source_hashes'] == {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    row = json.loads((output / 'citation_results.jsonl').read_text())
    assert row['success'] and row['context']['citation_sources'][0]['content'] == 'A fact.'
    assert json.loads((output / 'metrics.json').read_text())['citations']['summary']['success_count'] == 1
    client.embed.assert_called_once()
    assert db.read_bytes() == before
    artifacts = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit) as error:
        evaluation.main(args)
    assert error.value.code == 2
    assert {p.name: p.read_bytes() for p in output.iterdir()} == artifacts
