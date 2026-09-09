"""Query-only Ollama adapter reusing the validated embedding input pipeline."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from arkb.embeddings import embed_texts, prepare_query, validate_input_tokens
from arkb.schema import EmbeddingSpec
from arkb.tokenization import tokenizer_fingerprint

if TYPE_CHECKING:
    from ollama import Client
    from tokenizers import Tokenizer


@dataclass(frozen=True, kw_only=True)
class OllamaQueryEmbedder:
    client: 'Client'
    spec: EmbeddingSpec
    tokenizer: 'Tokenizer'
    tokenizer_identity: str
    max_input_tokens: int
    query_instruction: str

    def prepare(self, query: str) -> str:
        """Validate without model calls, including for an empty snapshot."""
        if tokenizer_fingerprint(self.tokenizer) != self.tokenizer_identity:
            raise ValueError('Query tokenizer differs from the indexed tokenizer; rebuild with matching settings.')
        text = prepare_query(query, instruction=self.query_instruction)
        validate_input_tokens(text, tokenizer=self.tokenizer, max_tokens=self.max_input_tokens, source='query')
        return text

    def embed_query(self, query: str) -> list[float]:
        text = self.prepare(query)
        return embed_texts([text], client=self.client, model=self.spec.model,
                          dimensions=self.spec.dimensions, dtype=self.spec.dtype,
                          normalization=self.spec.normalization,
                          context_length=self.max_input_tokens)[0].tolist()
