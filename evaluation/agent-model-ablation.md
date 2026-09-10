# Phase 1: Agent Model Ablation

Completed 360-run measurements and interpretation:
[Phase 1 results](phase1-agent-model-ablation-results.md).

Only the Agent generation model varies: `qwen3.5:4b`, `qwen3.5:9b`, and
`qwen3.5:27b`, in increasing capacity order. The default formal matrix is the
unchanged `agent_v1.jsonl` (40 cases) × 3 models × 3 independent trials = **360
calls to the existing `Runtime.ask`**.

`model_ablation.py` composes `run_agent_evaluation`; it does not implement another
Agent loop, retrieval pipeline or scoring framework. Optional namespaced hooks
save experiment metadata and deterministic analysis with each flushed v1 row.
Existing v1 callers and success rules are unchanged. Runtime/model/tool exceptions
remain failed trials; later trials/models continue without retries. Interruptions
keep partial rows and mark reports incomplete. Setup/artifact failures are separate.

## Run

Ollama must have all three exact tags and `qwen3-embedding:0.6b`. The shared SQLite
snapshot, its Qdrant collection and the cached embedding tokenizer must exist.
Preflight verifies embedding digest, tokenizer, collection identity/count and every
saved point/vector. It never downloads models, creates an index, replaces a model,
or falls back to another retrieval mode.

```sh
uv run --locked python -m arkb.evaluation.model_ablation \
  --baseline-metadata evaluation/results/<existing-run>/run_metadata.json \
  --output evaluation/results/agent_model_ablation/<preflight-id> --check-only

uv run --locked python -m arkb.evaluation.model_ablation \
  --baseline-metadata evaluation/results/<existing-run>/run_metadata.json \
  --output evaluation/results/agent_model_ablation/<smoke-id> --phase smoke

uv run --locked python -m arkb.evaluation.model_ablation \
  --baseline-metadata evaluation/results/<existing-run>/run_metadata.json \
  --output evaluation/results/agent_model_ablation/<formal-id> --num-trials 3
```

The baseline option imports its config, except output/trial count; old results
are never pooled. Relative paths resolve against the current working directory,
as in the original runner. Overrides for `--dataset`, `--db`, `--notes-dir`,
`--vault-id`, `--host`, `--qdrant-url`, and `--max-turns` apply uniformly to all
models. `--models` accepts exactly three accurate tags in 4B/9B/27B order;
choosing a different family/capacity set would be a different experiment.
Every output directory must be new.

Smoke uses six original cases, one per task type, one trial, and a separately
saved subset. `--smoke-cases` selects other original IDs without changing labels.
Missing models remain explicit slots in a partial smoke report. Formal execution
is blocked before any Agent calls if a required model or shared prerequisite is
unavailable. Exit status is nonzero for blocked, partial or invalid runs. Inspect
smoke traces/errors before formal execution; a smoke task failure alone does not
imply broken tool calling.

## Controls and provenance

Artifacts record dataset bytes/fingerprint, live corpus hashes, snapshot/chunk
identities, runtime/retrieval settings, exact Agent prompt/effective tool schemas,
git commit/branch/status and Python source hashes, package versions, Ollama version,
model digests, quantization/details and provider parameters. Input/source hashes
are checked before and after each model; drift invalidates comparison and stops
further models. Do not change code, corpus, active index, models or service settings
during a run.

The current shared `Runtime.ask` settings are:

- `think=True`, `max_turns=8`, `temperature=0`, no retries.
- Default search mode `semantic`; the same tools advertise BM25, semantic, hybrid.
- `RetrievalConfig()` supplies existing BM25, candidate and RRF settings.
- Reranking is **disabled** in this Agent path for every model. The separately
  enabled retrieval reranker now uses the pinned `Qwen/Qwen3-Reranker-0.6B`.
- Embedding digest and tokenizer must match the same saved snapshot.
- `num_ctx`, `num_predict`, seed and other sampling options remain existing
  provider defaults. No per-model overrides are added. Metadata distinguishes
  unset options from measured zero; differing provider defaults limit attribution
  to model capacity alone.

Models run sequentially in fixed order. Each trial starts fresh conversation
state through Runtime; provider caches may persist. Latency includes loading,
retrieval and failures, with no warmup correction; it is not pure inference time.

## Measurements and interpretation

Every row keeps v1 metrics and full `AgentTrace`. `analysis` adds match/search/read
counts, latency, attempted model requests (`trace.turns`), evidence timing and
efficiency. Prompt/completion/total tokens are **null**: the existing trace does
not retain reliable provider usage. Token counts are never estimated.

`evidence_sufficient_turn` is the turn returning the first observation satisfying
the existing recall/read labels (all target reads for direct-read tasks). It reuses
`evaluate_case` on a prefix with tool/budget restrictions relaxed and a final stop;
actual trial scores do not change. This measures annotated source coverage,
not answer correctness. No-retrieval/insufficient trajectories are undefined.

`wasted_tool_calls_after_sufficient_evidence` counts completed calls after that
observation, including later calls in the same preselected batch. It does **not**
prove avoidability. If a later call lacks an observation, execution is uncertain
and the value is null. Its mean is conditional on defined trials with a recorded
denominator; low coverage must not look efficient merely because waste is undefined.

`expected_sources_found_per_tool_call` divides distinct expected sources observed
by requested calls. Zero-call, missing-trace and no-retrieval denominators are null.
V1 denominators remain intact: unavailable traces count as task failures but not
observed zero turns/recall. Normal-final/error rates use all trials; recall excludes
no-retrieval; unnecessary retrieval uses observed no-retrieval trials.

Per-case success is successes / observed trials. Empirical pass@1 is the mean of
case probabilities, not pass@k extrapolation. Always-pass/flaky/always-fail require
all configured trials; partial cases are incomplete. Three trials do not establish
statistical significance.

Scaling wins: any larger model has higher case success than 4B. Regressions: an
adjacent larger model has lower case success (may overlap wins). Insensitive:
exactly equal empirical case success, which does not imply equal recall/latency.
Classification requires a complete, control-valid formal matrix. Reports contain
representative cases, all nine focused cases, actual query/read arguments, missing
sources, exceptions and full per-trial measurements.

Unknown-section and other errors retain their exact types/messages/partial traces.
AgentTrace has no error-stage field, so arbitrary exceptions are not heuristically
classified as model/tool/infrastructure failures. Error reductions alone do not
establish reasoning improvement. Compare recall with stopping and success; shared
failures warrant tool/runtime/annotation investigation before another model scale-up.

## Outputs and tests

Each experiment has `config.json`, `run_metadata.json`, original `cases.jsonl`,
per-model `results.jsonl`, `summary.json`, `report.md`, and unified `comparison.json`
/ `comparison.md`. Smoke adds `smoke_cases.jsonl`. Blocked formal cells remain
unavailable. Artifacts use the existing ignored `evaluation/results/` directory.

`load_trial_results(path)` replays traces with v1 metrics and rejects disagreement
with saved scores. Partial rows can be inspected without retrying any trial.

```sh
uv run --locked python -m pytest -q tests/evaluation/test_model_ablation.py
uv run --locked python -m pytest -q -m 'not integration'
```

Tests use scripted runtimes/mock HTTP, including the full 360-call expansion.
Real Ollama executions remain explicit CLI runs outside the deterministic suite.
