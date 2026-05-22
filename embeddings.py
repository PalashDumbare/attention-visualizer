"""Embedding helpers for token ids."""

import numpy as np


def create_embedding_matrix(vocab_size: int, embedding_dim: int) -> np.ndarray:
    """Create a random embedding matrix ,Shape (vocab_size, embedding_dim)."""
    return np.random.randn(vocab_size, embedding_dim)


def embed_tokens(tokens: list[int], embedding_matrix: np.ndarray) -> np.ndarray:
    """Return one embedding vector for each token id."""
    return embedding_matrix[tokens]
