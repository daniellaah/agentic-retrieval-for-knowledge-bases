"""Template/batching contracts, using tokens and tensors but no model weights."""

from types import SimpleNamespace
import sys
from unittest.mock import Mock

import pytest

from arkb.retrieval.qwen_rerank import QWEN_MODEL, QWEN_REVISION, QwenRerankerScorer
from tests.retrieval.test_rerank import candidates


def test_invalid_lengths_and_removed_model_overrides_fail_before_loading():
    for kwargs in ({'max_length': 0}, {'batch_size': False}):
        with pytest.raises(ValueError):
            QwenRerankerScorer(**kwargs)
    for kwargs in ({'model': 'another-model'}, {'revision': 'main'}):
        with pytest.raises(TypeError):
            QwenRerankerScorer(**kwargs)


def test_loading_pins_checkpoint_template_device_and_offline_settings(monkeypatch):
    tokenizer = Mock(unk_token_id=None)
    tokenizer.encode.side_effect = [[1, 2], [3, 4, 5]]
    tokenizer.convert_tokens_to_ids.side_effect = [10, 11]
    token_factory = Mock(return_value=tokenizer)
    model = Mock()
    model.to.return_value = model.eval.return_value = model
    model_factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(float32='float32'))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=token_factory),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=model_factory)))
    scorer = QwenRerankerScorer(max_length=128, batch_size=2,
                               cache_folder='/cache', local_files_only=True)
    options = dict(revision=QWEN_REVISION, cache_dir='/cache', local_files_only=True, trust_remote_code=False)
    token_factory.assert_called_once_with(QWEN_MODEL, padding_side='left', **options)
    model_factory.assert_called_once_with(QWEN_MODEL, dtype='float32', attn_implementation='sdpa', **options)
    model.to.assert_called_once_with('cpu')
    model.eval.assert_called_once_with()
    assert scorer.identity.startswith(f'{QWEN_MODEL}@{QWEN_REVISION}/cpu/float32/')
    assert scorer.score_type == 'yes_no_logit_difference'


@pytest.mark.parametrize('length,tokens,error', [
    (5, [10, 11], 'leave room'), (128, [10, 10], 'distinct yes/no'),
    (128, [None, 11], 'distinct yes/no'),
])
def test_invalid_template_budget_or_label_tokens_fail_before_loading_weights(monkeypatch, length, tokens, error):
    tokenizer = Mock(unk_token_id=None)
    tokenizer.encode.side_effect = [[1, 2], [3, 4, 5]]
    tokenizer.convert_tokens_to_ids.side_effect = tokens
    factory = Mock()
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=factory)))
    with pytest.raises(ValueError, match=error):
        QwenRerankerScorer(max_length=length)
    factory.assert_not_called()


def test_missing_optional_dependency_explains_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    with pytest.raises(ValueError, match='uv sync --locked --extra rerank'):
        QwenRerankerScorer()


def test_truncation_reserves_scoring_suffix_and_retains_title_body_input():
    class Tokenizer:
        def __call__(self, pairs, **options):
            self.pairs, self.options = pairs, options
            return {'input_ids': [[9] * options['max_length'], [8, 8]]}
        def pad(self, values, **options):
            self.padding = options
            return values
    scorer = object.__new__(QwenRerankerScorer)
    scorer.tokenizer = Tokenizer()
    scorer._prefix, scorer._suffix, scorer.max_length = [1, 2], [3, 4, 5], 8
    hits = candidates()
    result = scorer._inputs('Paris?', hits)
    assert result['input_ids'] == [[1, 2, 9, 9, 9, 3, 4, 5], [1, 2, 8, 8, 3, 4, 5]]
    assert scorer.tokenizer.options['max_length'] == 3
    assert scorer.tokenizer.options['truncation'] == 'longest_first'
    assert scorer.tokenizer.padding['return_attention_mask'] is True
    assert '<Query>: Paris?' in scorer.tokenizer.pairs[0]
    assert scorer.tokenizer.pairs[0].endswith(f"{hits[0].metadata.get('title') or ''}\n\n{hits[0].content}")


def test_last_token_yes_no_difference_and_batch_order():
    torch = pytest.importorskip('torch')
    scorer = object.__new__(QwenRerankerScorer)
    scorer.batch_size, scorer._true, scorer._false = 2, 2, 1
    seen = []
    def inputs(query, hits):
        seen.append(list(hits))
        return {'input_ids': torch.tensor([[hit] for hit in hits])}
    def model(input_ids, **options):
        assert options == {'use_cache': False, 'logits_to_keep': 1}
        assert not torch.is_grad_enabled()
        values = input_ids[:, 0].float()
        logits = torch.zeros((len(values), 1, 3))
        logits[:, 0, 1] = 5
        logits[:, 0, 2] = values
        return SimpleNamespace(logits=logits)
    scorer._inputs, scorer._model = inputs, model
    assert scorer.score('query', [2, 8, 6]) == [-3.0, 3.0, 1.0]
    assert seen == [[2, 8], [6]]
    assert scorer.score('query', []) == [] and len(seen) == 2
