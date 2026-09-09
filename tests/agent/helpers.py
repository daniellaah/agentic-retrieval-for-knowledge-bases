"""Scripted Ollama responses; no query routing or real model inference."""

from copy import deepcopy

from ollama import ChatResponse


def tool_call(name, **arguments):
    return {'function': {'name': name, 'arguments': arguments}}


def reply(content=None, *, calls=(), done_reason='stop'):
    return ChatResponse(message={'role': 'assistant', 'content': content, 'tool_calls': list(calls)},
                        done=True, done_reason=done_reason)


class ScriptedModel:
    def __init__(self, *steps):
        self.steps = iter(steps)
        self.requests = []

    def chat(self, **request):
        self.requests.append(deepcopy(request))
        step = next(self.steps)
        return step(request['messages']) if callable(step) else step
