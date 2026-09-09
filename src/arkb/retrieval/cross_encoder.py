"""Optional Sentence Transformers adapter. Importing this module loads no model."""

from collections.abc import Sequence
import re

from arkb.retrieval.contracts import SearchResult, validate_options

DEFAULT_MODEL = 'cross-encoder/ms-marco-MiniLM-L6-v2'
DEFAULT_REVISION = '233902d25c440f23af6f7d6e94d2946bac0bee0a'


class CrossEncoderScorer:
    """Pinned single-logit model, CPU inference, title+body passage pairs.

    The model tokenizer truncates pairs to max_length (including special tokens).
    Returned source evidence stays verbatim and untruncated. CPU inference and
    stable tie handling are reproducible within one software/hardware setup;
    floating point results need not match across library or hardware versions.
    """
    score_type = 'cross_encoder_logit'

    def __init__(self, *, model: str = DEFAULT_MODEL, revision: str = DEFAULT_REVISION,
                 max_length: int = 512, batch_size: int = 16,
                 cache_folder: str | None = None, local_files_only: bool = False):
        validate_options(max_length, None)
        validate_options(batch_size, None)
        if not isinstance(model, str) or not model.strip() or not isinstance(revision, str) or not re.fullmatch(r'[0-9a-f]{40}', revision):
            raise ValueError('Cross-encoder requires a model name and pinned 40-character commit revision.')
        try:
            from sentence_transformers import CrossEncoder
            from torch.nn import Identity
        except ImportError as error:
            raise ValueError('Install the optional reranker with: uv sync --locked --extra rerank') from error
        self._model = CrossEncoder(model, revision=revision, device='cpu', max_length=max_length,
                                   cache_folder=cache_folder, local_files_only=local_files_only,
                                   trust_remote_code=False)
        if self._model.config.num_labels != 1:
            raise ValueError('Cross-encoder must return one relevance logit per pair.')
        self._activation = Identity()
        self.batch_size = batch_size
        self.identity = f'{model}@{revision}/cpu/max_length={max_length}/batch_size={batch_size}/title-body-v1/raw-logit'

    def score(self, query: str, candidates: Sequence[SearchResult]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, f"{hit.metadata.get('title') or ''}\n\n{hit.content}") for hit in candidates]
        scores = self._model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False,
                                     activation_fn=self._activation, apply_softmax=False, convert_to_numpy=True)
        return scores.tolist()
