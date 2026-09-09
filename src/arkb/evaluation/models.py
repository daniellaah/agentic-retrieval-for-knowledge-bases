"""Validated annotations for agent retrieval evaluations, independent of indexing."""

from dataclasses import dataclass
from math import isfinite
from typing import Literal, get_args


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
