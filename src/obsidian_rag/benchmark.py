"""Frozen BrowseComp-Plus experiments over the existing RAG pipeline."""

from dataclasses import asdict
from collections import Counter
import argparse
from contextlib import ExitStack, closing
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter

from obsidian_rag.browsecomp import (
    AnswerLabels, BenchmarkQuestion, document_note, file_sha256, read_cases,
    read_corpus, read_jsonl, source_docid,
)
from obsidian_rag.chunking import whole_note_chunks
from obsidian_rag.schema import ChunkRecord
from obsidian_rag.evaluation import qrel_statistics, rank_documents


def _write_json(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def _write_rows(path, rows):
    with path.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def _hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def prepare_benchmark(corpus_path: Path, cases_path: Path, output: Path, *,
                      dataset_revision: str, query_ids=None) -> dict:
    """Freeze a selected query set against the entire supplied corpus.

    Corpus text is referenced by file hash, not copied. Relevance labels never
    select the runtime corpus. A reduced input corpus remains a custom subset.
    """
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError('An identified dataset_revision is required.')
    before = {'corpus': file_sha256(corpus_path), 'cases': file_sha256(cases_path)}
    provenance = None
    download = corpus_path.parent / 'download.json'
    if download.exists():
        provenance = json.loads(download.read_text(encoding='utf-8'))
        if (provenance.get('status') != 'complete'
                or provenance.get('corpus.jsonl_sha256') != before['corpus']
                or provenance.get('cases.jsonl_sha256') != before['cases']):
            raise ValueError('Incomplete or mismatched download hashes.')
    questions, labels = read_cases(cases_path)
    by_id = {q.query_id: (q, label) for q, label in zip(questions, labels)}
    selected = list(query_ids) if query_ids is not None else list(by_id)
    if not selected or len(set(selected)) != len(selected) or any(qid not in by_id for qid in selected):
        raise ValueError('Select nonempty, unique, known query IDs.')
    sources, docids, text_chars, largest = {}, set(), 0, 0
    for doc in read_corpus(corpus_path):
        note = document_note(doc)
        record = ChunkRecord.from_note(whole_note_chunks([note])[0], note=note,
                                       vault_id='browsecomp-' + before['corpus'][:16])
        sources[note.source] = {'docid': doc.docid, 'url': doc.url, 'title': note.title,
                                'text_sha256': _hash_text(doc.text), 'characters': len(doc.text),
                                'document_revision': record.document_revision}
        docids.add(doc.docid)
        text_chars += len(doc.text)
        largest = max(largest, len(doc.text))
    for qid in selected:
        label = by_id[qid][1]
        if not set(label.evidence_docids + label.gold_docids) <= docids:
            raise ValueError('Selected question references evidence missing from the supplied corpus.')
    if before != {'corpus': file_sha256(corpus_path), 'cases': file_sha256(cases_path)}:
        raise ValueError('Inputs changed during preparation.')
    output.mkdir(parents=True, exist_ok=False)
    _write_rows(output / 'questions.jsonl', (asdict(by_id[qid][0]) for qid in selected))
    _write_rows(output / 'labels.jsonl', (asdict(by_id[qid][1]) for qid in selected))
    _write_json(output / 'sources.json', sources)
    manifest = {'format_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
                'dataset_revision': dataset_revision, 'download_provenance': provenance,
                'corpus_scope': 'downloaded_full' if provenance else 'custom',
                'corpus_path': str(corpus_path.resolve()), 'input_hashes': before,
                'corpus_count': len(docids), 'corpus_text_characters': text_chars,
                'largest_document_characters': largest, 'query_ids': selected,
                'vault_id': 'browsecomp-' + before['corpus'][:16],
                'source_scope': 'browsecomp:' + before['corpus'],
                'artifact_hashes': {name: file_sha256(output / name)
                                   for name in ('questions.jsonl', 'labels.jsonl', 'sources.json')}}
    _write_json(output / 'manifest.json', manifest)
    return manifest


def load_prepared(directory: Path) -> tuple[dict, tuple[BenchmarkQuestion, ...], dict]:
    """Validate frozen artifacts and return runtime questions, never answer labels."""
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format_version') != 1:
        raise ValueError('Unsupported benchmark manifest.')
    for name in ('questions.jsonl', 'labels.jsonl', 'sources.json'):
        if file_sha256(directory / name) != manifest['artifact_hashes'].get(name):
            raise ValueError(f'Prepared artifact hash mismatch: {name}.')
    questions = tuple(BenchmarkQuestion(**row) for row in read_jsonl(directory / 'questions.jsonl'))
    if [q.query_id for q in questions] != manifest['query_ids']:
        raise ValueError('Question IDs differ from the frozen selection.')
    sources = json.loads((directory / 'sources.json').read_text(encoding='utf-8'))
    return manifest, questions, sources


def load_labels(directory: Path) -> dict[str, AnswerLabels]:
    manifest, _, _ = load_prepared(directory)
    labels = [AnswerLabels(**{**row, 'evidence_docids': tuple(row['evidence_docids']),
                             'gold_docids': tuple(row['gold_docids'])})
              for row in read_jsonl(directory / 'labels.jsonl')]
    if [label.query_id for label in labels] != manifest['query_ids']:
        raise ValueError('Labels differ from the frozen selection.')
    return {label.query_id: label for label in labels}


def build_benchmark_index(prepared: Path, *, storage, client, tokenizer, spec,
                          max_input_tokens: int, **settings):
    """Use the existing index lifecycle with a corpus-bound benchmark vault."""
    from obsidian_rag.indexing import build_index
    manifest, _, _ = load_prepared(prepared)
    path = Path(manifest['corpus_path'])
    expected = manifest['input_hashes']['corpus']
    if file_sha256(path) != expected:
        raise ValueError('Prepared corpus hash has changed; prepare a new experiment.')
    notes = [document_note(doc) for doc in read_corpus(path)]
    if file_sha256(path) != expected:
        raise ValueError('Prepared corpus hash changed while loading.')
    return build_index(storage, notes, client=client, tokenizer=tokenizer, spec=spec,
                       vault_id=manifest['vault_id'], max_input_tokens=max_input_tokens,
                       source_scope=manifest['source_scope'], **settings)


def _capture_snapshot(storage, manifest, sources, spec, index_version):
    active = storage.get_manifest(index_version) if index_version else storage.active_manifest(manifest['vault_id'])
    if active is None or active.status != 'ready' or active.vault_id != manifest['vault_id']:
        raise ValueError('A ready benchmark snapshot is required.')
    if active.embedding_spec != spec:
        raise ValueError('Runtime embedding specification differs from the snapshot.')
    metadata = storage.build_metadata(active.index_version)
    if metadata['backend'].get('source_scope') != manifest['source_scope']:
        raise ValueError('Snapshot source scope differs from the frozen corpus.')
    seen = set()
    for record in storage.snapshot_records(active.index_version):
        source = sources.get(record.chunk.source)
        if source is None or record.document_revision != source['document_revision']:
            raise ValueError('Snapshot document differs from the frozen corpus.')
        seen.add(record.chunk.source)
    if seen != set(sources):
        raise ValueError('Snapshot does not cover the frozen corpus.')
    return active, metadata


class _EmbeddingTimer:
    """Measure Ollama embedding request time without changing retrieval behavior."""
    def __init__(self, client):
        self.client = client
        self.milliseconds = 0.0

    def embed(self, **kwargs):
        started = perf_counter()
        try:
            return self.client.embed(**kwargs)
        finally:
            self.milliseconds += (perf_counter() - started) * 1000


def score_retrieval(prepared: Path, rows, *, doc_ks) -> dict:
    labels = load_labels(prepared)
    rows = list(rows)
    if [row['query_id'] for row in rows] != list(labels):
        raise ValueError('Results must include each frozen question exactly once, in order.')
    result = {}
    for group in ('evidence', 'gold'):
        result[group] = {}
        for k in doc_ks:
            values = [qrel_statistics([doc['docid'] for doc in row['documents']],
                       getattr(labels[row['query_id']], group + '_docids'), k=k) for row in rows]
            defined = [v for v in values if v['recall'] is not None]
            result[group][str(k)] = {key: sum(v[key] for v in defined) / len(defined) if defined else None
                                     for key in ('recall', 'ndcg')}
            result[group][str(k)]['defined_cases'] = len(defined)
    return result


def _code_identity():
    source = Path(__file__).parent
    git = subprocess.run(['git', '-C', str(source.parents[1]), 'rev-parse', 'HEAD'],
                         capture_output=True, text=True)
    packages = {}
    for name in ('numpy', 'ollama', 'tokenizers', 'qdrant-client', 'huggingface-hub', 'datasets'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'commit': git.stdout.strip() if git.returncode == 0 else None, 'packages': packages,
            'python': platform.python_version(),
            'source_hashes': {path.name: file_sha256(path) for path in sorted(source.glob('*.py'))}}


def run_benchmark(prepared: Path, *, storage, output: Path, client, tokenizer, spec,
                  chunk_top_k: int = 2, doc_ks=(5, 10, 100, 1000),
                  index_version=None, exact: bool = True, qdrant_client=None,
                  generate: bool = False, generation_client=None, generation_counter=None,
                  context_config=None, context_top_k: int | None = None, runtime_metadata=None) -> dict:
    """Retrieve each frozen question once; retain failures and score them too.

    Each row is flushed before continuing. An interrupted run has no summary
    and cannot be mistaken for a complete evaluation. No answer labels enter
    retrieval, and scoring only reads them after all raw results are saved.
    """
    from obsidian_rag.retrieval import search_index
    from obsidian_rag.context import ContextConfig
    if output.exists():
        raise FileExistsError(output)
    if type(chunk_top_k) is not int or chunk_top_k <= 0 or type(exact) is not bool:
        raise ValueError('Use a positive chunk_top_k and boolean exact.')
    if type(generate) is not bool:
        raise ValueError('generate must be boolean.')
    if generate:
        if generation_counter is None:
            raise ValueError('Generation requires an explicit validated generation_counter.')
        context_config = context_config or ContextConfig()
        context_top_k = chunk_top_k if context_top_k is None else context_top_k
        if type(context_top_k) is not int or not 0 < context_top_k <= chunk_top_k:
            raise ValueError('context_top_k must be between 1 and chunk_top_k.')
    elif any(v is not None for v in (generation_client, generation_counter, context_config, context_top_k)):
        raise ValueError('Generation settings require generate=True.')
    doc_ks = tuple(doc_ks)
    if not doc_ks or any(type(k) is not int or k <= 0 for k in doc_ks) or len(set(doc_ks)) != len(doc_ks):
        raise ValueError('doc_ks must contain unique positive cutoffs.')
    manifest, questions, sources = load_prepared(prepared)
    active, metadata = _capture_snapshot(storage, manifest, sources, spec, index_version)
    mapping = {source: value['docid'] for source, value in sources.items()}
    output.mkdir(parents=True, exist_ok=False)
    run = {'format_version': 1, 'created_at': datetime.now(timezone.utc).isoformat(),
           'prepared_path': str(prepared.resolve()), 'prepared_sha256': file_sha256(prepared / 'manifest.json'),
           'query_ids': manifest['query_ids'], 'snapshot': asdict(active), 'build_metadata': metadata,
           'chunk_top_k': chunk_top_k, 'doc_ks': doc_ks, 'exact': exact,
           'generation': {'enabled': generate, 'context_top_k': context_top_k,
                          'config': asdict(context_config) if generate else None,
                          'model': generation_counter.model if generate else None,
                          'counter': generation_counter.identity if generate else None,
                          'citation_mode': 'structured', 'temperature': 0, 'think': False},
           'document_ranking': 'max retrieved chunk score, ties ascending docid',
           'code': _code_identity(),
           'runtime': runtime_metadata or {},
           'timing_scope': 'query_embedding_ms measures embed requests; retrieval_ms includes query preparation, snapshot loading and search.'}
    _write_json(output / 'run.json', run)
    rows = []
    with (output / 'results.jsonl').open('x', encoding='utf-8') as stream:
        for question in questions:
            row = {'query_id': question.query_id, 'question': question.question,
                   'index_version': active.index_version, 'success': False,
                   'retrieval_success': False, 'error': None, 'hits': [], 'documents': [],
                   'context_docids': [], 'cited_docids': [], 'generation': None}
            timer = _EmbeddingTimer(client)
            started = perf_counter()
            try:
                hits = search_index(storage, question.question, vault_id=manifest['vault_id'],
                                    spec=spec, tokenizer=tokenizer, client=timer, top_k=chunk_top_k,
                                    exact=exact, index_version=active.index_version, qdrant_client=qdrant_client)
                row['documents'] = rank_documents(hits, mapping)
                row['hits'] = [{'record': asdict(hit.record), 'score': hit.score,
                                'docid': mapping[hit.chunk.source]} for hit in hits]
                row.update(success=True, retrieval_success=True)
            except Exception as error:
                # Per-query failures stay inspectable; interrupts still propagate.
                row['error'] = {'stage': 'retrieval', 'type': type(error).__name__, 'message': str(error)}
            row['timings'] = {'query_embedding_ms': timer.milliseconds,
                              'retrieval_ms': (perf_counter() - started) * 1000,
                              'context_ms': None, 'generation_validation_ms': None}
            if row['retrieval_success'] and generate:
                _generate_row(row, hits[:context_top_k], mapping, config=context_config,
                              counter=generation_counter, client=generation_client or client)
            row['timings']['total_ms'] = (perf_counter() - started) * 1000
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            rows.append(row)
    summary = {'query_count': len(rows), 'success_count': sum(row['success'] for row in rows),
               'error_count': sum(not row['success'] for row in rows),
               'retrieval': score_retrieval(prepared, rows, doc_ks=doc_ks),
               'results_sha256': file_sha256(output / 'results.jsonl'),
               'run_sha256': file_sha256(output / 'run.json')}
    _write_json(output / 'summary.json', summary)
    return summary


def _generate_row(row, hits, mapping, *, config, counter, client):
    from obsidian_rag.context import ContextBudgetError, build_context
    from obsidian_rag.evaluation import evaluate_citation_context
    started = perf_counter()
    try:
        context = build_context(row['question'], hits, config=config, counter=counter,
                                citation_mode='structured')
        if context.status == 'budget_exhausted':
            raise ContextBudgetError('No evidence fits the generation budget.')
        row['context_docids'] = list(dict.fromkeys(mapping[s.source] for s in context.citation_sources))
    except Exception as error:
        row.update(success=False, error={'stage': 'context', 'type': type(error).__name__, 'message': str(error)})
        return
    finally:
        row['timings']['context_ms'] = (perf_counter() - started) * 1000
    try:
        generated = evaluate_citation_context(context, client=client)
        row['generation'] = generated
        row['timings']['generation_validation_ms'] = generated['generation_ms']
        row['success'] = generated['success']
        if generated['success']:
            row['cited_docids'] = list(dict.fromkeys(mapping[s['source']] for s in generated['result']['sources']))
        else:
            row['error'] = {'stage': 'generation', **generated['error']}
    except Exception as error:
        row.update(success=False, error={'stage': 'generation', 'type': type(error).__name__, 'message': str(error)})


def load_run(directory: Path):
    """Only complete, unmodified runs can be scored or exported."""
    run = json.loads((directory / 'run.json').read_text(encoding='utf-8'))
    summary = json.loads((directory / 'summary.json').read_text(encoding='utf-8'))
    if (run.get('format_version') != 1 or summary.get('run_sha256') != file_sha256(directory / 'run.json')
            or summary.get('results_sha256') != file_sha256(directory / 'results.jsonl')):
        raise ValueError('Run artifact hash mismatch or unsupported format.')
    prepared = Path(run['prepared_path'])
    if file_sha256(prepared / 'manifest.json') != run['prepared_sha256']:
        raise ValueError('Prepared manifest differs from the run.')
    _, questions, _ = load_prepared(prepared)
    rows = list(read_jsonl(directory / 'results.jsonl'))
    if ([r['query_id'] for r in rows] != run['query_ids']
            or [(r['query_id'], r['question']) for r in rows] != [(q.query_id, q.question) for q in questions]):
        raise ValueError('Run does not contain the complete frozen question set.')
    return run, rows, summary


def judge_run(directory: Path, output: Path, *, client, config, runtime_metadata=None) -> dict:
    """Regrade archived answers without loading an index or calling the generator."""
    from obsidian_rag.judging import JUDGE_TEMPLATE_REVISION, JUDGE_TEMPLATE_SHA256, judge_answer, prediction_text
    if output.exists():
        raise FileExistsError(output)
    run, rows, run_summary = load_run(directory)
    if not run['generation']['enabled']:
        raise ValueError('A generation run is required for answer judging.')
    labels = load_labels(Path(run['prepared_path']))
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / 'judge_run.json', {'run_path': str(directory.resolve()),
                'run_sha256': run_summary['run_sha256'], 'results_sha256': run_summary['results_sha256'],
                'query_ids': run['query_ids'], 'config': asdict(config), 'code': _code_identity(),
                'runtime': runtime_metadata or {},
                'template_revision': JUDGE_TEMPLATE_REVISION, 'template_sha256': JUDGE_TEMPLATE_SHA256})
    judgments = []
    with (output / 'judgments.jsonl').open('x', encoding='utf-8') as stream:
        for row in rows:
            if not row['success']:
                judgment = {'status': 'pipeline_error', 'correct': None, 'error': row['error']}
            else:
                prediction = prediction_text(row['generation']['result']['answer'])
                judgment = judge_answer(row['question'], prediction, labels[row['query_id']].answer,
                                        client=client, config=config)
            judgment['query_id'] = row['query_id']
            stream.write(json.dumps(judgment, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            judgments.append(judgment)
    correct = sum(j['status'] == 'graded' and j['correct'] for j in judgments)
    graded = sum(j['status'] == 'graded' for j in judgments)
    errors = sum(j['status'] == 'error' for j in judgments)
    pipeline_errors = sum(j['status'] == 'pipeline_error' for j in judgments)
    total = len(rows)
    summary = {'query_count': total, 'graded_count': graded, 'correct_count': correct,
               'incorrect_count': graded - correct, 'judge_error_count': errors,
               'pipeline_error_count': pipeline_errors,
               'accuracy_on_graded': correct / graded if graded else None,
               'end_to_end_accuracy': correct / total if not errors else None,
               'accuracy_lower_bound': correct / total, 'accuracy_upper_bound': (correct + errors) / total,
               'run_sha256': run_summary['run_sha256'], 'results_sha256': run_summary['results_sha256'],
               'judgments_sha256': file_sha256(output / 'judgments.jsonl'),
               'judge_run_sha256': file_sha256(output / 'judge_run.json'),
               'limits': 'Pipeline failures count as unsuccessful tasks. Judge errors remain unresolved, not incorrect answers. Citation support is not judged.'}
    _write_json(output / 'summary.json', summary)
    return summary


def report_run(directory: Path, output: Path, *, judgments: Path | None = None, doc_ks=None) -> dict:
    """Recompute metrics offline; source runs and judgments remain immutable."""
    import numpy as np
    if output.exists():
        raise FileExistsError(output)
    run, rows, original = load_run(directory)
    cutoffs = tuple(run['doc_ks'] if doc_ks is None else doc_ks)
    if not cutoffs or any(type(k) is not int or k <= 0 for k in cutoffs) or len(set(cutoffs)) != len(cutoffs):
        raise ValueError('doc_ks must contain unique positive cutoffs.')
    report = {**original, 'retrieval': score_retrieval(Path(run['prepared_path']), rows, doc_ks=cutoffs),
              'doc_ks': cutoffs, 'semantic_citation_support': None, 'answer_judgments': None,
              'answer_statuses': dict(Counter(row['generation']['result']['answer']['status'] for row in rows
                                             if row['generation'] and row['generation']['result'])),
              'errors_by_stage': dict(Counter(row['error']['stage'] for row in rows if row['error'])),
              'timings': {}, 'tokens': {}, 'citation_checks': {}}
    for key in ('query_embedding_ms', 'retrieval_ms', 'context_ms', 'generation_validation_ms', 'total_ms'):
        values = [row['timings'][key] for row in rows if row['timings'].get(key) is not None]
        report['timings'][key] = {'defined_cases': len(values),
            'mean': float(np.mean(values)) if values else None,
            'p50': float(np.median(values)) if values else None,
            'p95': float(np.percentile(values, 95)) if values else None}
    generations = [row['generation'] for row in rows if row['generation'] is not None]
    for key in ('structure_valid', 'references_valid', 'citation_id_validity', 'claim_reference_coverage'):
        values = [g['metrics'][key] for g in generations if g['metrics'][key] is not None]
        report['citation_checks'][key] = {'defined_cases': len(values),
                                          'mean': float(np.mean(values)) if values else None}
    usages = [(g['result']['token_usage'] if g['result'] else (g['error'] or {}).get('token_usage'))
              for g in generations]
    for key in ('prompt_tokens', 'actual_prompt_tokens', 'output_tokens'):
        values = [usage[key] for usage in usages if usage and usage.get(key) is not None]
        report['tokens'][key] = {'defined_cases': len(values), 'mean': float(np.mean(values)) if values else None}
    if judgments is not None:
        scored = json.loads((judgments / 'summary.json').read_text(encoding='utf-8'))
        if (scored.get('run_sha256') != original['run_sha256']
                or scored.get('results_sha256') != original['results_sha256']
                or scored.get('judgments_sha256') != file_sha256(judgments / 'judgments.jsonl')
                or scored.get('judge_run_sha256') != file_sha256(judgments / 'judge_run.json')):
            raise ValueError('Judgments do not match this run or their artifact hashes.')
        report['answer_judgments'] = scored
    labels = load_labels(Path(run['prepared_path']))
    per_query = []
    for row in rows:
        label = labels[row['query_id']]
        per_query.append({'query_id': row['query_id'], **{
            group: {str(k): qrel_statistics([doc['docid'] for doc in row['documents']],
                       getattr(label, group + '_docids'), k=k) for k in cutoffs}
            for group in ('evidence', 'gold')}})
    report['limits'] = ('Document qrels do not establish fact coverage in sent chunks or citation entailment. '
                        'Timing excludes corpus preparation, index build, runtime setup and result-file writes. '
                        'The current index builder materializes notes, chunks and vectors in memory.')
    output.mkdir(parents=True, exist_ok=False)
    _write_rows(output / 'retrieval_metrics.jsonl', per_query)
    _write_json(output / 'metrics.json', report)
    return report


def export_run(directory: Path, output: Path) -> dict:
    """Export document TREC ranks and, when generated, official per-query JSON.

    Ordinal TREC scores preserve our deterministic tie order; original cosine
    scores remain in results.jsonl. Only generated claims and limitations enter
    the answer field, not the renderer's appended source passages.
    """
    if output.exists():
        raise FileExistsError(output)
    run, rows, summary = load_run(directory)
    for row in rows:
        if any(any(c.isspace() for c in value) for value in
               [row['query_id'], *(doc['docid'] for doc in row['documents'])]):
            raise ValueError('TREC IDs cannot contain whitespace.')
    output.mkdir(parents=True, exist_ok=False)
    if run['generation']['enabled']:
        (output / 'runs').mkdir()
    with (output / 'run.trec').open('x', encoding='utf-8') as stream:
        for row in rows:
            for i, doc in enumerate(row['documents']):
                stream.write(f"{row['query_id']} Q0 {doc['docid']} {i + 1} {len(row['documents']) - i} obsidian-rag\n")
            if not run['generation']['enabled']:
                continue
            text = ''
            if row['success']:
                generated = row['generation']['result']
                source_ids = {s['source_id']: source_docid(s['source']) for s in generated['sources']}
                answer = generated['answer']
                parts = [claim['text'] + ' ' + ''.join('[' + source_ids[sid] + ']' for sid in claim['source_ids'])
                         for claim in answer['claims']]
                if answer['missing_information']:
                    parts.append('Missing information: ' + ' '.join(answer['missing_information']))
                text = '\n\n'.join(parts)
            payload = {'query_id': row['query_id'], 'tool_call_counts': {'search': 1},
                       'status': 'completed' if row['success'] else 'failed',
                       'retrieved_docids': [doc['docid'] for doc in row['documents']],
                       'result': [{'type': 'output_text', 'output': text}]}
            _write_json(output / 'runs' / (_hash_text(row['query_id']) + '.json'), payload)
    metadata = {'query_count': len(rows), 'run_sha256': summary['run_sha256'],
                'results_sha256': summary['results_sha256'],
                'runs_directory': str((output / 'runs').resolve()) if run['generation']['enabled'] else None,
                'trec_score_policy': 'descending rank ordinals; raw scores are retained in the source run'}
    _write_json(output / 'export.json', metadata)
    return metadata


def _positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    download = commands.add_parser('download', help='Explicitly download pinned official data.')
    download.add_argument('--output', type=Path, required=True)
    download.add_argument('--query-revision', required=True)
    download.add_argument('--corpus-revision', required=True)
    prepare = commands.add_parser('prepare', help='Freeze local data and question selection.')
    for name in ('corpus', 'cases', 'output'):
        prepare.add_argument('--' + name, type=Path, required=True)
    prepare.add_argument('--dataset-revision', required=True)
    prepare.add_argument('--query-ids', nargs='+')
    for name in ('index', 'run'):
        command = commands.add_parser(name)
        command.add_argument('--prepared', type=Path, required=True)
        command.add_argument('--db', type=Path, required=True)
        command.add_argument('--host', default='http://127.0.0.1:11434')
        command.add_argument('--timeout', type=float, default=180)
        command.add_argument('--offline', action='store_true', help='Use only cached tokenizers; Ollama is still contacted.')
        command.add_argument('--tokenizer-cache', type=Path)
        command.add_argument('--embedding-model', default='qwen3-embedding:0.6b' if name == 'index' else None)
        command.add_argument('--qdrant-url')
        if name == 'index':
            command.add_argument('--backend', choices=('numpy', 'qdrant'), default='numpy')
            command.add_argument('--context-length', type=_positive, default=8192)
            command.add_argument('--chunking', choices=('recursive', 'none'), default='recursive')
            command.add_argument('--chunk-size', type=_positive, default=512)
            command.add_argument('--chunk-overlap', type=int, default=64)
            command.add_argument('--batch-size', type=_positive, default=32)
            command.add_argument('--max-batch-tokens', type=_positive)
            command.add_argument('--max-retries', type=int, default=0)
        else:
            command.add_argument('--output', type=Path, required=True)
            command.add_argument('--chunk-top-k', type=_positive, default=2)
            command.add_argument('--doc-ks', type=_positive, nargs='+', default=[5, 10, 100, 1000])
            command.add_argument('--index-version')
            command.add_argument('--ann', action='store_true')
            command.add_argument('--generate', action='store_true')
            command.add_argument('--generation-model')
            command.add_argument('--context-top-k', type=_positive)
            command.add_argument('--context-window', type=_positive)
            command.add_argument('--max-output-tokens', type=_positive)
            command.add_argument('--context-safety-margin', type=int)
    judge = commands.add_parser('judge', help='Grade saved answers; never regenerate them.')
    judge.add_argument('--run', type=Path, required=True)
    judge.add_argument('--output', type=Path, required=True)
    judge.add_argument('--host', default='http://127.0.0.1:11434')
    judge.add_argument('--timeout', type=float, default=180)
    judge.add_argument('--judge-model', default='qwen3:32b')
    judge.add_argument('--context-window', type=_positive, default=16384)
    judge.add_argument('--max-output-tokens', type=_positive, default=4096)
    judge.add_argument('--temperature', type=float, default=.7)
    judge.add_argument('--think', action=argparse.BooleanOptionalAction, default=True)
    for name in ('report', 'export'):
        command = commands.add_parser(name)
        command.add_argument('--run', type=Path, required=True)
        command.add_argument('--output', type=Path, required=True)
        if name == 'report':
            command.add_argument('--judgments', type=Path)
            command.add_argument('--doc-ks', type=_positive, nargs='+')
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        if hasattr(args, 'timeout') and (not math.isfinite(args.timeout) or args.timeout <= 0):
            raise ValueError('timeout must be positive and finite.')
        if hasattr(args, 'output') and args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        if args.command == 'download':
            from obsidian_rag.browsecomp import download_browsecomp
            result = download_browsecomp(args.output, query_revision=args.query_revision, corpus_revision=args.corpus_revision)
        elif args.command == 'prepare':
            result = prepare_benchmark(args.corpus, args.cases, args.output,
                                       dataset_revision=args.dataset_revision, query_ids=args.query_ids)
        elif args.command == 'report':
            result = report_run(args.run, args.output, judgments=args.judgments, doc_ks=args.doc_ks)
        elif args.command == 'export':
            result = export_run(args.run, args.output)
        elif args.command == 'judge':
            from ollama import Client
            from obsidian_rag.judging import JudgeConfig
            load_run(args.run)
            with Client(host=args.host, timeout=args.timeout, trust_env=False) as client:
                matches = [model for model in client.list().models if model.model == args.judge_model]
                if len(matches) != 1 or not matches[0].digest:
                    raise ValueError('Judge model is not installed or has no digest; no weights are downloaded automatically.')
                info = client.show(args.judge_model).modelinfo or {}
                limits = [v for k, v in info.items() if k.endswith('.context_length') and type(v) is int]
                if limits and args.context_window > min(limits):
                    raise ValueError('Judge window exceeds the installed model limit.')
                config = JudgeConfig(args.judge_model, matches[0].digest, context_window=args.context_window,
                                     max_output_tokens=args.max_output_tokens, temperature=args.temperature, think=args.think)
                result = judge_run(args.run, args.output, client=client, config=config,
                                   runtime_metadata=_server_metadata(args.host, args.timeout))
        else:
            result = _index_or_run(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if result.get('error_count', 0) or result.get('judge_error_count', 0) else 0
    except Exception as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


def _index_or_run(args):
    from ollama import Client
    from obsidian_rag.context import ContextConfig, load_generation_counter
    from obsidian_rag.embeddings import resolve_embedding_spec
    from obsidian_rag.retrieval import connect_qdrant
    from obsidian_rag.storage import SQLiteStorage
    from obsidian_rag.tokenization import load_tokenizer
    manifest, _, _ = load_prepared(args.prepared)
    if args.command == 'run' and not args.generate and any(value is not None for value in
            (args.generation_model, args.context_top_k, args.context_window, args.max_output_tokens, args.context_safety_margin)):
        raise ValueError('Generation options require --generate.')
    tokenizer = load_tokenizer(cache_dir=args.tokenizer_cache, local_files_only=args.offline)
    with ExitStack() as resources:
        client = resources.enter_context(Client(host=args.host, timeout=args.timeout, trust_env=False))
        storage = resources.enter_context(SQLiteStorage(args.db, read_only=args.command == 'run'))
        if args.command == 'index':
            spec = resolve_embedding_spec(client, args.embedding_model, context_length=args.context_length)
            backend = {'kind': args.backend}
            qclient = None
            if args.backend == 'qdrant':
                if not args.qdrant_url:
                    raise ValueError('--qdrant-url is required for a Qdrant index.')
                qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url, args.timeout)))
                backend['url'] = args.qdrant_url
            return asdict(build_benchmark_index(args.prepared, storage=storage, client=client,
                tokenizer=tokenizer, spec=spec, max_input_tokens=args.context_length,
                chunking=args.chunking, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap,
                batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens,
                max_retries=args.max_retries, backend=backend, qdrant_client=qclient))
        active = storage.get_manifest(args.index_version) if args.index_version else storage.active_manifest(manifest['vault_id'])
        if active is None:
            raise ValueError('Build a benchmark index before running questions.')
        metadata = storage.build_metadata(active.index_version)['backend']
        spec = resolve_embedding_spec(client, args.embedding_model or active.embedding_spec.model,
                                      context_length=metadata['input']['max_tokens'])
        qclient = None
        if metadata['kind'] == 'qdrant':
            qclient = resources.enter_context(closing(connect_qdrant(args.qdrant_url or metadata['url'], args.timeout)))
        generation = {}
        if args.generate:
            generation = {'generate': True, 'context_top_k': args.context_top_k,
                'generation_counter': load_generation_counter(client=client,
                    model=args.generation_model or 'qwen3.5:4b', cache_dir=args.tokenizer_cache, local_files_only=args.offline),
                'context_config': ContextConfig(args.context_window or 8192, args.max_output_tokens or 1024,
                                               128 if args.context_safety_margin is None else args.context_safety_margin)}
        return run_benchmark(args.prepared, storage=storage, output=args.output, client=client,
            tokenizer=tokenizer, spec=spec, chunk_top_k=args.chunk_top_k, doc_ks=args.doc_ks,
            index_version=active.index_version, exact=not args.ann, qdrant_client=qclient,
            runtime_metadata=_server_metadata(args.host, args.timeout), **generation)


def _server_metadata(host, timeout):
    import httpx
    result = {'ollama_host': host, 'ollama_version': None}
    try:
        with httpx.Client(trust_env=False, timeout=min(timeout, 10)) as client:
            response = client.get(host.rstrip('/') + '/api/version')
            response.raise_for_status()
            result['ollama_version'] = response.json()['version']
    except Exception as error:
        result['version_lookup_error'] = type(error).__name__
    return result


if __name__ == '__main__':
    raise SystemExit(main())
