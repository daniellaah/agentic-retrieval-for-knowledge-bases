import pytest

from obsidian_rag.judging import prediction_text


def test_judge_sees_only_model_claims_and_limitations_not_appended_source_text():
    answer = {'status': 'partial', 'claims': [{'text': 'A supported observation.', 'source_ids': ['S1']}],
              'missing_information': ['The exact name is not established.']}
    text = prediction_text(answer)
    assert text == 'A supported observation.\n\nMissing information: The exact name is not established.'
    assert 'S1' not in text


@pytest.mark.parametrize('raw,done,expected', [
    ('extracted_final_answer: Orchid\nreasoning: Equivalent name.\ncorrect: yes\nconfidence: 90', 'stop', True),
    ('**extracted_final_answer:** None\n**reasoning:** Refused.\n**correct:** no\n**confidence:** 100', 'stop', False),
    ('extracted_final_answer: Orchid\nreasoning: Equivalent.\ncorrect: maybe', 'stop', None),
    ('extracted_final_answer: Orchid\nreasoning: Equivalent.\ncorrect: yes\ncorrect: no', 'stop', None),
    ('extracted_final_answer: Orchid\nreasoning: Equivalent.\ncorrect: yes', 'length', None),
])
def test_judge_distinguishes_wrong_answers_from_invalid_or_truncated_grading(raw, done, expected):
    from unittest.mock import Mock
    from ollama import Client, ChatResponse
    from obsidian_rag.judging import JudgeConfig, judge_answer
    client = Mock(spec=Client)
    client.chat.return_value = ChatResponse(message={'role': 'assistant', 'content': raw}, done_reason=done)
    config = JudgeConfig(model='fixture-judge', model_revision='fixed')
    result = judge_answer('Which name?', 'Orchid', 'Orchid', client=client, config=config)
    assert result['correct'] is expected
    assert result['status'] == ('error' if expected is None else 'graded')
    assert result['raw_response'] == raw
    assert result['judge']['model_revision'] == 'fixed'


def test_judge_transport_failure_and_oversized_input_are_not_answer_errors():
    from unittest.mock import Mock
    from ollama import Client
    from obsidian_rag.judging import JudgeConfig, judge_answer
    client = Mock(spec=Client)
    client.chat.side_effect = OSError('offline')
    config = JudgeConfig(model='fixture-judge', model_revision='fixed')
    result = judge_answer('Question?', 'Prediction', 'Reference', client=client, config=config)
    assert result['correct'] is None and result['error']['code'] == 'OSError'
    result = judge_answer('Question?', 'x' * 20000, 'Reference', client=client, config=config)
    assert result['correct'] is None and result['error']['code'] == 'input_budget'
