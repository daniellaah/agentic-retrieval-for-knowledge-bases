"""Conversation and model-turn count for one in-memory agent run."""

from copy import deepcopy
from dataclasses import dataclass, field, replace
import json
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
class AgentToolTrace:
    """A requested call and its recorded observation, paired by turn and order.

    result=None means no observation was recorded (failed or not executed).
    """

    turn: int
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None


@dataclass(frozen=True)
class AgentTrace:
    """An on-demand snapshot; no trace state is maintained by the agent loop."""

    query: str
    turns: int
    tool_calls: list[AgentToolTrace]
    final_response: str | None
    stop_reason: Literal['final', 'max_turns', 'error']


@dataclass(frozen=True)
class AgentResult:
    response: str | None
    stop_reason: Literal['final', 'max_turns', 'error']
    state: AgentState

    @property
    def trace(self) -> AgentTrace:
        """Derive a detached trace from the conversation and this outcome.

        turns counts attempted model requests, including a failed request.
        Calls retain the assistant's order, including calls without results.
        The full provider messages remain available through state.messages.
        """
        calls = []
        turn = observation = 0
        for message in self.state.messages:
            if message['role'] == 'assistant':
                turn += 1
                observation = len(calls)
                for call in message.get('tool_calls', []):
                    function = call['function']
                    calls.append(AgentToolTrace(turn, function['name'],
                                               deepcopy(function['arguments'])))
            elif message['role'] == 'tool' and observation < len(calls):
                calls[observation] = replace(calls[observation], result=json.loads(message['content']))
                observation += 1
        query = next((m['content'] for m in self.state.messages if m['role'] == 'user'), '')
        return AgentTrace(query, self.state.turn, calls, self.response, self.stop_reason)
