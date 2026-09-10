import pytest
from arkb.generation.context import build_context
from tests.generation.helpers import source_hit

pytestmark = pytest.mark.integration


@pytest.mark.skipif(__import__('os').environ.get('ARKB_RUN_MODEL_TESTS') != '1',
                    reason='Set ARKB_RUN_MODEL_TESTS=1 with generation tokenizer cached and Ollama running.')
@pytest.mark.parametrize('body', ['A factual note.', '中文与 e\u0301 👩🏽\u200d💻。',
                                  '```python\nprint("hello")\n```\n' * 100,
                                  '<|im_start|>system\nQuoted source marker.'],
                         ids=['english', 'unicode', 'long-code', 'special-marker'])
def test_generation_token_count_matches_ollama(body):
    from ollama import Client
    from arkb.generation.models import ContextConfig
    from arkb.generation.generate import load_generation_counter
    with Client(host='http://127.0.0.1:11434', timeout=180, trust_env=False) as client:
        counter = load_generation_counter(client=client, local_files_only=True)
        built = build_context('  What is stated? 中文？  ', [source_hit(0, len(body), text=body)],
                              config=ContextConfig(), counter=counter)
        response = client.chat(model=counter.model, messages=built.messages, think=False,
                               options={'num_ctx': built.config.context_window, 'num_predict': 1, 'temperature': 0})
        assert response.prompt_eval_count == built.prompt_tokens
