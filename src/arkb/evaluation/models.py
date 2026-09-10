"""Validated annotations for agent retrieval evaluations, independent of indexing."""

from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from typing import Literal, get_args

from arkb.agent.state import AgentTrace
from arkb.config import (
    DEFAULT_AGENT_THINK, DEFAULT_DB, DEFAULT_GENERATION_MODEL, DEFAULT_NOTES_DIR,
    RetrievalConfig, RuntimeConfig,
)
from arkb.retrieval.models import SearchResponse, validate_options


TaskType = Literal['exact_lookup', 'semantic_discovery', 'direct_read',
                   'exploratory_retrieval', 'knowledge_qa', 'no_retrieval']
TASK_TYPES = get_args(TaskType)
KNOWLEDGE_TOOLS = frozenset({'match', 'search', 'read'})


def _strings(value, name):
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() or item != item.strip() for item in value
    ):
        raise ValueError(f'{name} must be an array of nonblank strings without surrounding whitespace.')
    if len(set(value)) != len(value):
        raise ValueError(f'{name} must not contain duplicates.')
    return tuple(value)


@dataclass(frozen=True, kw_only=True)
class AgentEvalCase:
    """Outcome labels, never an expected trajectory.

    Sources use the current flat knowledge-relative Markdown filenames. Recall
    is undefined for no_retrieval. min_read_sources counts distinct expected
    sources actually returned by read, regardless of call order or selector.
    """

    id: str
    query: str
    task_type: TaskType
    expected_sources: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] | None = None
    forbidden_tools: tuple[str, ...] = ()
    max_tool_calls: int | None = None
    min_source_recall: float = 1.0
    min_read_sources: int = 0
    notes: str = ''

    def __post_init__(self):
        for name in ('id', 'query'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonblank string.')
        if self.id != self.id.strip():
            raise ValueError('id must not have surrounding whitespace.')
        if self.task_type not in TASK_TYPES:
            raise ValueError(f'invalid task_type: {self.task_type!r}.')
        for name in ('expected_sources', 'forbidden_tools', 'allowed_tools'):
            value = getattr(self, name)
            if name == 'allowed_tools' and value is None:
                continue
            object.__setattr__(self, name, _strings(value, name))
        for source in self.expected_sources:
            if ('/' in source or '\\' in source or '\x00' in source
                    or not source.endswith('.md') or source == '.md'):
                raise ValueError('expected_sources must use flat knowledge-relative .md filenames.')
        for name in ('allowed_tools', 'forbidden_tools'):
            if set(getattr(self, name) or ()) - KNOWLEDGE_TOOLS:
                raise ValueError(f'{name} contains an unknown tool.')
        if set(self.allowed_tools or ()) & set(self.forbidden_tools):
            raise ValueError('allowed_tools and forbidden_tools must not overlap.')
        if self.max_tool_calls is not None and (
            type(self.max_tool_calls) is not int or self.max_tool_calls < 0
        ):
            raise ValueError('max_tool_calls must be a nonnegative integer or null.')
        if (type(self.min_source_recall) not in (int, float)
                or not isfinite(self.min_source_recall) or not 0 < self.min_source_recall <= 1):
            raise ValueError('min_source_recall must be in (0, 1].')
        if (type(self.min_read_sources) is not int
                or not 0 <= self.min_read_sources <= len(self.expected_sources)):
            raise ValueError('min_read_sources must be between zero and the expected source count.')
        if self.task_type == 'no_retrieval':
            if self.expected_sources:
                raise ValueError('no_retrieval cannot have expected_sources.')
        elif not self.expected_sources:
            raise ValueError('retrieval cases require nonempty expected_sources.')
        if self.task_type in ('exact_lookup', 'direct_read') and self.min_source_recall != 1:
            raise ValueError('exact_lookup and direct_read require full source recall.')
        if not isinstance(self.notes, str):
            raise ValueError('notes must be a string.')


@dataclass(frozen=True, kw_only=True)
class AgentEvalResult:
    """Deterministic outcome/behavior metrics for one case execution.

    None denotes an undefined denominator or unavailable trace observation,
    never a measured zero. stop_reason retains the AgentTrace value verbatim;
    evaluation does not introduce a competing stop-reason enum.
    """

    case_id: str
    task_type: TaskType
    success: bool
    source_recall: float | None
    expected_sources: tuple[str, ...]
    retrieved_sources: tuple[str, ...] | None
    read_sources: tuple[str, ...] | None
    tool_calls: tuple[str, ...] | None
    tool_counts: dict[str, int] | None
    tool_call_count: int | None
    turn_count: int | None
    forbidden_tool_violations: dict[str, int] | None
    disallowed_tool_violations: dict[str, int] | None
    max_tool_calls_exceeded: bool | None
    max_turn_failure: bool | None
    unnecessary_retrieval: bool | None
    stop_reason: str | None
    failure_reasons: tuple[str, ...]


@dataclass(frozen=True, kw_only=True)
class AgentEvalConfig:
    dataset_path: Path = Path('evaluation/data/agent_v1.jsonl')
    output_dir: Path | None = None
    model: str = DEFAULT_GENERATION_MODEL
    max_turns: int = 8
    num_trials: int = 1
    db: Path = DEFAULT_DB
    notes_dir: Path = DEFAULT_NOTES_DIR
    vault_id: str = 'default'
    think: bool = DEFAULT_AGENT_THINK
    runtime_config: RuntimeConfig = RuntimeConfig()

    def __post_init__(self):
        for name in ('dataset_path', 'output_dir', 'db', 'notes_dir'):
            value = getattr(self, name)
            if name == 'output_dir' and value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f'{name} must be a nonblank path.')
            object.__setattr__(self, name, Path(value))
        for name in ('max_turns', 'num_trials'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        for name in ('model', 'vault_id'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f'{name} must be a nonblank string.')
        if type(self.think) is not bool:
            raise ValueError('think must be boolean.')
        if not isinstance(self.runtime_config, RuntimeConfig):
            raise ValueError('runtime_config must be a RuntimeConfig.')


@dataclass(frozen=True, kw_only=True)
class AgentEvalTrial:
    case: AgentEvalCase
    trial: int
    trace: AgentTrace | None
    metrics: AgentEvalResult
    runtime_metadata: dict
    error: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class AgentEvalRun:
    output_dir: Path
    results: tuple[AgentEvalTrial, ...]
    summary: dict


BASELINE_MODES = {
    'bm25': ('bm25', False),
    'semantic': ('semantic', False),
    'hybrid': ('hybrid', False),
    'hybrid_rerank': ('hybrid', True),
}


@dataclass(frozen=True, kw_only=True)
class BaselineEvalConfig:
    """One engine request per applicable case/method on a pinned snapshot.

    top_k is the engine's chunk/result limit, not a promise of K unique sources.
    Metrics use source cutoffs 1/3/5/10 up to that limit, plus the limit itself.
    """
    dataset_path: Path = AgentEvalConfig.dataset_path
    output_dir: Path | None = None
    db: Path = DEFAULT_DB
    notes_dir: Path = DEFAULT_NOTES_DIR
    vault_id: str = 'default'
    index_version: str | None = None
    baselines: tuple[str, ...] = tuple(BASELINE_MODES)
    top_k: int = 10
    exact: bool = False
    retrieval_config: RetrievalConfig = RetrievalConfig()
    runtime_config: RuntimeConfig = RuntimeConfig()

    def __post_init__(self):
        for name in ('dataset_path', 'output_dir', 'db', 'notes_dir'):
            value = getattr(self, name)
            if name == 'output_dir' and value is None:
                continue
            if not isinstance(value, (str, Path)) or not str(value).strip():
                raise ValueError(f'{name} must be a nonblank path.')
            object.__setattr__(self, name, Path(value))
        if not isinstance(self.vault_id, str) or not self.vault_id.strip():
            raise ValueError('vault_id must be nonblank.')
        if self.index_version is not None and (not isinstance(self.index_version, str) or not self.index_version.strip()):
            raise ValueError('index_version must be nonblank or null.')
        object.__setattr__(self, 'baselines', _strings(self.baselines, 'baselines'))
        if not self.baselines or set(self.baselines) - BASELINE_MODES.keys():
            raise ValueError(f'baselines must select from {tuple(BASELINE_MODES)}.')
        validate_options(self.top_k, None)
        if type(self.exact) is not bool:
            raise ValueError('exact must be boolean.')
        if not isinstance(self.runtime_config, RuntimeConfig) or not isinstance(self.retrieval_config, RetrievalConfig):
            raise ValueError('Expected RuntimeConfig and RetrievalConfig settings.')
        settings = self.retrieval_config
        if set(self.baselines) & {'hybrid', 'hybrid_rerank'}:
            from arkb.retrieval.hybrid import rrf
            validate_options(settings.candidate_k, None)
            rrf([], k=settings.rrf_k)
            if self.top_k > settings.candidate_k:
                raise ValueError('top_k cannot exceed hybrid candidate_k.')
        if 'hybrid_rerank' in self.baselines:
            validate_options(settings.rerank_candidates, None)
            validate_options(settings.reranker_max_length, None)
            if not self.top_k <= settings.rerank_candidates <= settings.candidate_k:
                raise ValueError('Require top_k <= rerank_candidates <= candidate_k.')

    @property
    def metric_ks(self) -> tuple[int, ...]:
        return tuple(sorted({k for k in (1, 3, 5, 10) if k <= self.top_k} | {self.top_k}))


@dataclass(frozen=True, kw_only=True)
class BaselineEvalResult:
    case_id: str
    query: str
    task_type: TaskType
    expected_sources: tuple[str, ...]
    baseline: str
    mode: str
    rerank: bool
    status: Literal['ok', 'error', 'not_applicable']
    retrieved_sources: tuple[str, ...] | None
    ranked_sources: tuple[str, ...] | None
    metrics: dict[str, float | None]
    latency_ms: float | None
    response: SearchResponse | None
    error: dict[str, str] | None = None


@dataclass(frozen=True, kw_only=True)
class BaselineEvalRun:
    output_dir: Path
    results: tuple[BaselineEvalResult, ...]
    summary: dict
