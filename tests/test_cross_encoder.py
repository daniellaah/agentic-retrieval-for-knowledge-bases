from types import SimpleNamespace
from unittest.mock import Mock
import sys


def test_cross_encoder_pins_model_uses_query_passage_pairs_and_returns_raw_logits(monkeypatch):
    from arkb.retrieval.cross_encoder import CrossEncoderScorer
    from arkb.retrieval import SearchResult
    predict = Mock(return_value=SimpleNamespace(tolist=lambda: [2., -1.]))
    model = SimpleNamespace(predict=predict, config=SimpleNamespace(num_labels=1))
    factory = Mock(return_value=model)
    monkeypatch.setitem(sys.modules, 'sentence_transformers', SimpleNamespace(CrossEncoder=factory))
    monkeypatch.setitem(sys.modules, 'torch.nn', SimpleNamespace(Identity=lambda: 'raw'))
    scorer = CrossEncoderScorer(model='model', revision='a' * 40, max_length=128, batch_size=2,
                                local_files_only=True)
    hits = [SearchResult(source_id=s, source=s, content=t, method='test', metadata={'title': 'Title'})
            for s, t in [('a', 'evidence'), ('b', 'other')]]
    assert scorer.score('query', hits) == [2., -1.]
    assert factory.call_args.kwargs['revision'] == 'a' * 40
    assert factory.call_args.kwargs['local_files_only'] is True
    assert factory.call_args.kwargs['trust_remote_code'] is False
    assert factory.call_args.kwargs['device'] == 'cpu'
    assert predict.call_args.args == ([('query', 'Title\n\nevidence'), ('query', 'Title\n\nother')],)
    assert predict.call_args.kwargs['activation_fn'] == 'raw'
    assert scorer.score_type == 'cross_encoder_logit'
