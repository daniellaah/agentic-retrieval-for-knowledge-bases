# Agent Evaluation v1

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
