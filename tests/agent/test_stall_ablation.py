from dataclasses import asdict

import pytest

from arkb.agent.loop import run_agent, _SEARCH_STALLED_INSTRUCTION
from tests.agent.helpers import ScriptedModel, reply, tool_call
from tests.agent.test_observation import tools, observer


def test_ablation_changes_only_reminder_and_preserves_default():
    steps = [reply(calls=[tool_call('search', query='q')]) for _ in range(3)] + [reply('done')]
    models = [ScriptedModel(*steps) for _ in range(3)]
    results = [run_agent('q', client=m, tools=tools(), model='fake', observer=observer(), **kw)
               for m, kw in zip(models, [{}, {'search_stall_reminder': True}, {'search_stall_reminder': False}])]
    assert models[0].requests == models[1].requests
    assert asdict(results[0].trace) == asdict(results[1].trace) == asdict(results[2].trace)
    assert models[0].requests[:3] == models[2].requests[:3]
    on, off = models[0].requests[-1], models[2].requests[-1]
    assert on['messages'][-1]['content'] == _SEARCH_STALLED_INSTRUCTION
    assert {**on, 'messages': on['messages'][:-1]} == off


@pytest.mark.parametrize('value', [0, 1, None, 'false'])
def test_ablation_requires_boolean(value):
    m = ScriptedModel(reply('done'))
    with pytest.raises(ValueError, match='search_stall_reminder'):
        run_agent('q', client=m, tools=tools(), model='fake', search_stall_reminder=value)
    assert not m.requests
