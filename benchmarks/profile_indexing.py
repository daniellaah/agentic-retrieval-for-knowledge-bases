"""Measure current indexing work without changing its algorithms or using services.

Run from the repository root with `uv run --locked python -B
benchmarks/profile_indexing.py --output /tmp/indexing-overhead.json`.
Uninstrumented repetitions provide wall times. One separate instrumented run
provides call counts and inclusive/exclusive attribution, not a speedup estimate.
All databases and Qdrant Local collections live in temporary storage.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack, closing, contextmanager
from datetime import datetime, timezone
from functools import wraps
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sqlite3
from statistics import median
from tempfile import TemporaryDirectory
from time import perf_counter
from unittest.mock import patch
import warnings

import numpy as np
from ollama import EmbedResponse
from qdrant_client import QdrantClient
from tokenizers import Tokenizer, models, pre_tokenizers

from arkb.knowledge.embeddings import prepare_document
from arkb.knowledge.models import QdrantConfig
from arkb.knowledge.qdrant import QdrantIndex
from arkb.knowledge.indexing import build_index
from arkb.knowledge.chunking import whole_note_chunks
from arkb.knowledge.models import EmbeddingSpec, Note
from arkb.knowledge.sqlite import SQLiteStorage


class Timings:
    def __init__(self):
        self.stack = []
        self.entries = defaultdict(lambda: {'calls': 0, 'inclusive_seconds': 0., 'exclusive_seconds': 0.})
        self.edges = Counter()

    @contextmanager
    def measure(self, name):
        parent = self.stack[-1]['name'] if self.stack else '<root>'
        self.edges[f'{parent} -> {name}'] += 1
        frame = {'name': name, 'children': 0.}
        self.stack.append(frame)
        started = perf_counter()
        try:
            yield
        finally:
            elapsed = perf_counter() - started
            self.stack.pop()
            entry = self.entries[name]
            entry['calls'] += 1
            entry['inclusive_seconds'] += elapsed
            entry['exclusive_seconds'] += elapsed - frame['children']
            if self.stack:
                self.stack[-1]['children'] += elapsed

    def wrap(self, function, name):
        @wraps(function)
        def measured(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return measured


class MeasuredConnection(sqlite3.Connection):
    timings = None

    def execute(self, statement, *args, **kwargs):
        if self.timings is None:
            return super().execute(statement, *args, **kwargs)
        sql = statement.upper()
        if sql.startswith('SELECT'):
            kind = next((f'select_{table.lower()}' for table in
                         ('EMBEDDINGS', 'SNAPSHOT_CHUNKS', 'BUILDS', 'ACTIVE_INDEXES') if f'FROM {table}' in sql), 'select_other')
        elif sql.startswith('INSERT'):
            kind = next((f'insert_{table.lower()}' for table in
                         ('EMBEDDINGS', 'SNAPSHOT_CHUNKS', 'BUILDS', 'ACTIVE_INDEXES') if f'INTO {table}' in sql), 'insert_other')
        else:
            kind = sql.split()[0].lower()
        # Fetch/decoding after execute remains attributed to the calling method.
        with self.timings.measure('sql.' + kind):
            return super().execute(statement, *args, **kwargs)


@contextmanager
def instrument(storage, timings):
    methods = {
        SQLiteStorage: ('get_embedding', 'put_embeddings', 'get_manifest', 'snapshot_records',
                        'add_chunk', 'load_snapshot', 'publish'),
        QdrantIndex: ('upsert', 'verify_snapshot', 'wait_ready', 'check_configuration', 'count'),
    }
    with ExitStack() as stack:
        for cls, names in methods.items():
            for name in names:
                stack.enter_context(patch.object(cls, name, timings.wrap(getattr(cls, name), cls.__name__ + '.' + name)))
        storage.connection.timings = timings
        try:
            yield
        finally:
            storage.connection.timings = None


class PreparedEmbeddings:
    """Return precomputed vectors so model latency cannot hide local work."""

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = 0

    def embed(self, *, input, **kwargs):
        self.calls += 1
        return EmbedResponse(embeddings=[self.vectors[text] for text in input])


def fixture(count, dimensions):
    notes = [Note(f'Note {i}', 'Evidence about indexing, caching, and source snapshots. ' * 16, f'{i:06d}.md')
             for i in range(count)]
    vectors = np.random.default_rng(7).normal(size=(count, dimensions))
    vectors /= np.linalg.norm(vectors, axis=1)[:, None]
    prepared = {prepare_document(chunk): vector.tolist()
                for chunk, vector in zip(whole_note_chunks(notes), vectors)}
    tokenizer = Tokenizer(models.WordLevel({'[UNK]': 0}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    spec = EmbeddingSpec(model='benchmark-precomputed', model_revision='seed-7', dimensions=dimensions,
                         document_template='title-body-v1')
    return notes, prepared, tokenizer, spec


def run_once(data, *, profiled=False):
    notes, vectors, tokenizer, spec = data
    original_connect = sqlite3.connect

    def connect(*args, **kwargs):
        return original_connect(*args, **kwargs, factory=MeasuredConnection)

    # A subclass is used only in the separate profiled run. Normal runs use the
    # application's exact connection implementation and no timing wrappers.
    with ExitStack() as resources:
        if profiled:
            resources.enter_context(patch('sqlite3.connect', connect))
        root = Path(resources.enter_context(TemporaryDirectory(prefix='arkb-indexing-profile-')))
        storage = resources.enter_context(SQLiteStorage(root / 'index.sqlite'))
        qdrant = resources.enter_context(closing(QdrantClient(':memory:')))
        model = PreparedEmbeddings(vectors)
        options = dict(spec=spec, vault_id='benchmark', client=model, tokenizer=tokenizer,
                       max_input_tokens=512, chunking='none', batch_size=32,
                       qdrant_config=QdrantConfig(), qdrant_client=qdrant)
        pragma = {key: storage.connection.execute('PRAGMA ' + key).fetchone()[0]
                  for key in ('journal_mode', 'synchronous', 'wal_autocheckpoint')}
        results = {}
        for phase, force in (('cold', False), ('cached_rebuild', True), ('unchanged', False)):
            timings = Timings()
            model.calls = 0
            with ExitStack() as measurement:
                if profiled:
                    measurement.enter_context(instrument(storage, timings))
                    measurement.enter_context(timings.measure('build_index'))
                started = perf_counter()
                report = build_index(storage, notes, **options, force=force)
                elapsed = perf_counter() - started
            assert report.reused_index == (phase == 'unchanged')
            assert report.embedded_inputs == (len(notes) if phase == 'cold' else 0)
            results[phase] = {'wall_seconds': elapsed, 'build_seconds': report.build_seconds,
                              'embedded_inputs': report.embedded_inputs, 'cached_inputs': report.cached_inputs,
                              'embedding_calls': model.calls, 'sqlite_settings': pragma}
            if profiled:
                results[phase]['timings'] = dict(timings.entries)
                results[phase]['call_edges'] = dict(timings.edges)
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sizes', type=int, nargs='+', default=[256, 2048])
    parser.add_argument('--dimensions', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if min(*args.sizes, args.dimensions, args.repeats) <= 0:
        parser.error('sizes, dimensions and repeats must be positive')
    if args.output.exists():
        parser.error('--output must be a new file')
    package = Path(__import__('arkb').__file__).parent
    output = {'created_at': datetime.now(timezone.utc).isoformat(),
              'environment': {'platform': platform.platform(), 'machine': platform.machine(),
                              'python': platform.python_version(), 'sqlite': sqlite3.sqlite_version,
                              'numpy': np.__version__, 'qdrant_client': version('qdrant-client')},
              'settings': {'dimensions': args.dimensions, 'repeats': args.repeats, 'seed': 7,
                           'batch_size': 32, 'chunking': 'none', 'unique_input_fraction': 1.,
                           'qdrant': 'Local :memory:', 'embedding': 'precomputed; no model inference'},
              'source_hashes': {p.relative_to(package).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(package.rglob('*.py'))},
              'limits': ['API build time excludes file scanning, vector creation, setup and imports.',
                         'This isolates local indexing work; it does not measure real model or Qdrant Server latency.',
                         'Profile times include instrumentation overhead; use unprofiled medians for wall time.',
                         'Inclusive method times overlap. SQL timings cover execute, not later row fetching.',
                         'No batching, memoization, skipped checks or storage changes are applied.'],
              'results': []}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Payload indexes have no effect in the local Qdrant.*')
        run_once(fixture(16, args.dimensions))  # Warm imports and code paths.
        for size in args.sizes:
            data = fixture(size, args.dimensions)
            runs = [run_once(data) for _ in range(args.repeats)]
            profiled = run_once(data, profiled=True)
            phases = {}
            for phase in profiled:
                times = [run[phase]['wall_seconds'] for run in runs]
                phases[phase] = {'unprofiled_runs': [run[phase] for run in runs],
                                  'median_seconds': median(times), 'min_seconds': min(times), 'max_seconds': max(times),
                                  'profiled': profiled[phase]}
            output['results'].append({'chunks': size, 'phases': phases})
            print(json.dumps({'chunks': size, 'median_seconds': {phase: value['median_seconds'] for phase, value in phases.items()}}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as destination:
        json.dump(output, destination, indent=2)
        destination.write('\n')


if __name__ == '__main__':
    main()
