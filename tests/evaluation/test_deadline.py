"""Timeouts must preserve partial observations and cannot count as answers."""
import signal
import time
import pytest
from httpcore._exceptions import map_exceptions, ReadTimeout

from arkb.agent import run_agent, AgentBudget, AgentObserver
from arkb.evaluation.deadline import evaluation_deadline, EvaluationDeadlineExceeded
from tests.agent.helpers import ScriptedModel, reply, tool_call


@pytest.mark.parametrize('stage', ['model_request', 'tool_execution'])
def test_blocked_call_retains_failure_observation(stage):
    class Tools:
        def tool_definitions(self): return []
        def match(self, **kwargs): time.sleep(10)
        search = read = match

    class BlockedModel:
        def chat(self, **kwargs):
            # Real HTTP transport exception mapping must not hide the harness timer.
            with map_exceptions({TimeoutError: ReadTimeout}):
                time.sleep(10)

    client = BlockedModel() if stage == 'model_request' else ScriptedModel(reply(calls=[tool_call('match',query='absent')]))
    observer = AgentObserver(budget=AgentBudget(), counter=len, counter_identity='fixture')
    previous = signal.getsignal(signal.SIGALRM)
    with pytest.raises(EvaluationDeadlineExceeded) as failure:
        with evaluation_deadline(0.05):
            run_agent('fixture', client=client, tools=Tools(), model='fixture', observer=observer)
    result = failure.value.agent_result
    assert result.stop_reason == 'error' and result.response is None
    assert result.observation['models']
    assert result.observation['error']['stage'] == stage
    assert signal.getsignal(signal.SIGALRM) == previous
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)
