"""Conversation and model-turn count for one in-memory agent run."""

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class AgentState:
    messages: list[dict[str, Any]] = field(default_factory=list)
    turn: int = 0

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        """Ordered call history, derived from messages without a second store."""
        return [call for message in self.messages if message['role'] == 'assistant'
                for call in message.get('tool_calls', [])]


@dataclass(frozen=True)
class AgentResult:
    response: str | None
    stop_reason: Literal['final', 'max_turns']
    state: AgentState
