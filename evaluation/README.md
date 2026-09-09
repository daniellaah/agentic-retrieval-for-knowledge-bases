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
