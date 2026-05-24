"""High-level pipeline for building attention visualization data."""

import numpy as np

from src.embeddings import create_embedding_matrix, embed_tokens
from src.tokenizer import build_vocab, text_to_tokens
from src.attention import compute_attention_weights


DEFAULT_TEXT = "Cat sat on a mat"
DEFAULT_EMBEDDING_DIM = 4


def build_attention_weights(
    text: str = DEFAULT_TEXT,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
) -> tuple[list[str], np.ndarray]:
    """Build character labels and attention weights for visualization."""
    vocab = build_vocab(text)
    tokens = text_to_tokens(text, vocab)

    embedding_matrix = create_embedding_matrix(vocab.size, embedding_dim)
    token_embeddings = embed_tokens(tokens, embedding_matrix)

    weights = compute_attention_weights(emb_dim=embedding_dim, embeddings=token_embeddings)

    return list(text), weights
