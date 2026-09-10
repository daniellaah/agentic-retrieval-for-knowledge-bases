# Agent Evaluation v1

For the controlled three-model, three-trial experiment, see
[Phase 1: Agent Model Ablation](agent-model-ablation.md).
The completed 360-run experiment is documented in the
[Phase 1 measured results](phase1-agent-model-ablation-results.md).
The current fixed reranker is verified in
[Qwen3 Reranker integration results](reranker-integration.md).

The versioned input is `data/agent_v1.jsonl`: one UTF-8 JSON object per line,
ordered by task type and case ID, with no model output or runtime state.
Annotations were curated against the local `example_notes` bodies, including
the longer notes' individual sections. `notes` records why each source is
relevant. Exact labels also have an executable check against real `match`.

| Task type | Cases |
| --- | ---: |
| exact_lookup | 6 |
| semantic_discovery | 6 |
| direct_read | 6 |
| exploratory_retrieval | 10 |
| knowledge_qa | 6 |
| no_retrieval | 6 |
| Total | 40 |

## Annotations and loading

```python
from pathlib import Path
from arkb.evaluation.datasets import load_agent_eval_dataset

cases = load_agent_eval_dataset(
    Path('evaluation/data/agent_v1.jsonl'), notes_dir=Path('example_notes'),
)
```

`AgentEvalCase` contains `id`, `query`, `task_type`, `expected_sources`, optional
`allowed_tools`, `forbidden_tools`, `max_tool_calls`, `min_source_recall`,
`min_read_sources`, and review `notes`. Source lists become immutable tuples in
Python and JSON arrays on disk. Missing `allowed_tools` allows any tool; an empty
list allows none. These are scoring constraints, never restrictions injected
into the runtime or the model's prompt.

Non-retrieval cases have no expected sources. Other cases require nonempty
expected evidence. Exact lookups and direct reads require full coverage.
Semantic and exploratory labels include multiple useful sources and explicit
recall thresholds, so finding a useful subset can succeed. `min_read_sources`
counts distinct expected sources successfully returned by `read`; it is used
only where the exploratory query explicitly asks to open notes. No annotation
requires a particular tool order. QA labels identify key evidence and do not
attempt to grade prose.

The loader rejects malformed rows, unknown fields/types/tools, duplicate JSON
keys or case IDs, invalid source arrays, and contradictory tool constraints.
`notes_dir` optionally checks files without opening an index. Paths follow the
current flat document contract: knowledge-relative `.md` filenames, with no
directory traversal or external symlinks. An empty dataset is an error.

Exact queries explicitly target body text: current `match` removes the H1 title
from its searchable body and counts occurrences rather than unique sources.
For example, `Agent Memory` occurs only in a title, so it is not labeled as a
positive body lookup. `RAG` includes case-sensitive substrings and source links;
its case requires increasing the match limit or otherwise gathering all sources.

## Implementation batches

1. Dataset, validated annotations, loader, and live `match/read` contract tests.
2. Trace evidence extraction, deterministic case metrics, and aggregates.
3. Runtime runner, independent trials, serialized traces, JSON/Markdown reports.

Each batch runs deterministic cross-module integration checks and the full
deterministic regression suite before its commit. Real Ollama/Qdrant evaluations
remain separate from those tests.

## Deterministic metrics

`arkb.evaluation.agent.evaluate_case(case, trace)` returns an `AgentEvalResult`.
`extract_retrieved_sources` unions exact `source` values from every successful
`match/search` observation's `results` array and `read` observation's `result`.
Repeated chunks, sources and calls contribute once to coverage. Tool arguments,
final citations, unknown tools, missing observations and error payloads are not
evidence. Document-ID and section/range reads use the returned source identity.

Source recall is `len(expected ∩ retrieved) / len(expected)`. It is null for
no-retrieval cases. This measures source membership, not passage coverage or
whether the model understood the evidence.

Every success requires `stop_reason == 'final'`, no forbidden or disallowed
tool requests, and compliance with `max_tool_calls` if present. Further rules:

| Task | Evidence rule |
| --- | --- |
| exact_lookup | All expected sources retrieved, through any valid tools |
| semantic_discovery | Recall meets `min_source_recall` |
| direct_read | Every expected source successfully returned by `read` |
| exploratory_retrieval | Recall meets threshold and `min_read_sources` distinct expected sources read |
| knowledge_qa | Recall meets threshold (all labeled key evidence in v1) |
| no_retrieval | No `match`, `search`, or `read` request, including failed requests |

The annotated read minimum is also checked for any other retrieval task that
sets it. No success check depends on final answer wording or a unique tool order.

Results include expected/retrieved/read sources, ordered tool names, counts by
name (including zero counts for the three knowledge tools), total calls, trace
turns, forbidden/disallowed call counts, tool-budget violation, max-turn failure,
unnecessary retrieval, native stop reason, and simple failed-constraint codes.
Call counts measure **requests** in the trace: a failed batch may include calls
that the runtime never executed. Turns include attempted failed model requests.

`summarize_agent_results(results)` computes trial-weighted macro means overall
and under `by_task_type`. It includes distinct case count, execution count,
success/failure counts, success rate, mean recall/calls/turns, max-turn failure
rate, unnecessary retrieval rate, tool request totals and stop distributions.
Every nullable metric has a `*_defined_trials` denominator. Unnecessary retrieval
uses only observed no-retrieval trials, not all retrieval tasks. Repeated trials
are retained; the success rate is neither pass-at-k nor all-trials-success.

If a runtime exception has no partial trace, `evaluate_case(case, None)` records
failure and leaves unobserved behavior null. Such trials count in task success
and `missing_trace_trials` but cannot establish recall, zero calls, or absence of
unnecessary retrieval. Current native stop reasons are `final`, `max_turns`, and
`error`; distributions also expose `other` and `unavailable`. Error is not guessed
to mean model_error or tool_error: the current trace has no explicit error stage.

## Running an evaluation

```python
from pathlib import Path
from arkb.evaluation.agent_runner import run_agent_evaluation
from arkb.evaluation.models import AgentEvalConfig

run = run_agent_evaluation(AgentEvalConfig(
    dataset_path=Path('evaluation/data/agent_v1.jsonl'),
    notes_dir=Path('example_notes'),
    db=Path('.obsidian-rag/index.sqlite'),
    model='qwen3.5:4b', max_turns=8, num_trials=2,
    output_dir=Path('evaluation/results/my-first-run'),
))
print(run.summary['task_success_rate'])
```

The existing evaluation convention also supplies a thin module CLI:

```sh
uv run --locked python -m arkb.evaluation.agent_runner \
  --dataset evaluation/data/agent_v1.jsonl \
  --notes-dir example_notes --db .obsidian-rag/index.sqlite \
  --generation-model qwen3.5:4b --max-turns 8 --num-trials 2 \
  --output evaluation/results/my-first-run
```

Run this from the repository root, with an index of the same `example_notes`
and the configured model services available. `--host`, `--timeout`, `--offline`,
`--tokenizer-cache`, `--qdrant-url`, `--vault-id`, and `--think/--no-think` configure
the existing Runtime. `--model` aliases `--generation-model`. `num_trials`
defaults to 1. Omitting output creates a unique `evaluation/results/<run-id>`
directory. An existing output directory is always rejected. Generated results
under that default root are gitignored.

The runner validates the complete dataset before calling `Runtime.ask`. It then
executes cases in file order, and trials `0..num_trials-1` for each case. Only the
query and ordinary runtime options are passed to `ask`; labels and constraints
remain in evaluation. Each `ask` creates fresh conversation state. The runner
extracts `AgentResult.trace`, computes case metrics, writes and flushes one row,
then proceeds. It never implements another agent loop or executes tools itself.

Ordinary runtime exceptions keep `error.agent_result.trace` where available and
record exception type/message. Each failed trial remains in the result and the
next trial still runs. There is no automatic retry. Ctrl-C propagates and marks
metadata `interrupted`; already flushed rows remain. Artifact failures are run
errors, not task failures, and metadata marks the run `failed`. A completed run
with failed tasks is still a completed experiment (CLI exit 0).

Pass `runtime=` and optionally `client=` to inject a fake or existing runtime for
tests or application composition. A supplied runtime/client remains caller-owned.
Otherwise the runner owns one Runtime context for the run. The effective runtime
config is recorded when it is available, separately from requested configuration.

## Artifacts and replay

Every completed run contains:

| File | Contents |
| --- | --- |
| `results.jsonl` | One `schema_version`, `case`, zero-based `trial`, complete `trace`, `metrics`, `runtime_metadata`, and nullable `error` per line |
| `summary.json` | Overall metrics, denominators, `by_task_type`, distributions and `failed_trials` with query, sources, sequence and stop reason |
| `report.md` | Aggregate tables plus a separate detail block for every failed case/trial |
| `cases.jsonl` | Exact original dataset bytes |
| `run_metadata.json` | Run status/times, requested/effective settings, dataset hash, Python version, commit and source hashes, knowledge fingerprints before/after |

All JSON is UTF-8 with ordinary arrays/objects and strict finite numbers. Missing
traces and undefined metrics are JSON null; no Python object representations are
saved. Each row's runtime metadata includes model, turn limit, thinking setting,
runtime type, timing, dataset hash and a reference to the run metadata file.
The latter records live Markdown SHA-256 hashes and, if present, the SQLite
manifest, build metadata, and corpus manifest using existing fingerprint helpers.
`knowledge_changed` flags differences at run boundaries.

Saved rows can be scored again without a model or index:

```python
import json
from arkb.agent.state import AgentToolTrace, AgentTrace
from arkb.evaluation.agent import evaluate_case
from arkb.evaluation.models import AgentEvalCase

row = json.loads((run.output_dir / 'results.jsonl').read_text().splitlines()[0])
payload = row['trace']
trace = (AgentTrace(**{**payload, 'tool_calls': [AgentToolTrace(**c) for c in payload['tool_calls']]})
         if payload is not None else None)
metrics = evaluate_case(AgentEvalCase(**row['case']), trace)
```

## Verification and limits

```sh
# Deterministic cross-module integration: real documents, match/read/BM25,
# a persisted SQLite/Qdrant-local index, Runtime.ask, traces and reports.
.venv/bin/python -m pytest -q tests/evaluation/test_agent_integration.py

# Full deterministic suite; never calls real LLMs or external services.
.venv/bin/python -m pytest -q -m 'not integration'
```

The repository's `integration` marker specifically gates real models/external
services. These new local cross-module integration tests run in the deterministic
suite. Scripted model replies verify wiring and metric contracts; their success
rates are not evidence of Qwen performance. A real-model benchmark is the separate
runner command above, not part of the deterministic test suite.

The 40 cases are an initial curated sample of this small English corpus, queried
mostly in Chinese. Relevance labels and thresholds are reviewable judgments,
not exhaustive proof of all possible useful evidence. Flat source sets have no
graded relevance or interchangeable evidence groups. Source recall can reward
wide retrieval and does not penalize irrelevant additional sources. Read checks
establish returned source identity, not passage completeness, reading depth, or
answer correctness. No LLM judge, framework, agent optimization or failure
classification is included.

Inspection before implementation found that the described Agent Evaluation
models were absent from this checkout. The new contracts therefore live only in
`arkb.evaluation.models`. The existing AgentTrace/AgentResult contract is reused
without modifying agent, retrieval or knowledge decisions. Its relevant limits:

- Stops distinguish `final`, `max_turns`, and `error`, with no explicit error stage.
- Requested tool calls can include a failed call and later unexecuted calls in the
  same batch; `result=None` does not distinguish those two situations.
- Some pre-loop failures or exceptions rejecting attributes have no partial trace.
- Traces preserve tool observations and final response, not full provider prompts,
  thinking text, generation token usage, or immutable model digests.
- Match/read use live text; each ask captures the current published search snapshot.
  Keep inputs stable during comparisons. Before/after fingerprints detect boundary
  differences, not every transient change or a change followed by a reversion.
  Tool evidence also retains document revisions. The runner does not pin a run-wide
  snapshot or alter Runtime resource semantics.

Stored observations make deterministic metric replay possible. Model tags,
provider versions, local service state and nondeterminism can still change a
future live run; a saved configuration does not promise identical trajectories.

## Changed files

| Area | Files |
| --- | --- |
| Versioned input | `evaluation/data/agent_v1.jsonl` |
| Contracts and loading | `src/arkb/evaluation/models.py`, `src/arkb/evaluation/datasets.py` |
| Metrics and execution | `src/arkb/evaluation/agent.py`, `src/arkb/evaluation/agent_runner.py` |
| Package description | `src/arkb/evaluation/__init__.py` |
| Deterministic tests | `tests/evaluation/test_agent_dataset.py`, `test_agent_metrics.py`, `test_agent_runner.py`, `test_agent_integration.py` |
| Documentation and output hygiene | `evaluation/README.md`, `README.md`, `.gitignore` |
