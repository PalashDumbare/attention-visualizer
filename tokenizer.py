"""Character-level tokenization utilities."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Vocabulary:
    """Bidirectional mapping between characters and token ids."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @property
    def size(self) -> int:
        return len(self.stoi)


def build_vocab(text: str) -> Vocabulary:
    """Build a deterministic character vocabulary from text."""
    chars = sorted(set(text))
    stoi = {char: index for index, char in enumerate(chars)}
    itos = {index: char for char, index in stoi.items()}
    return Vocabulary(stoi=stoi, itos=itos)


def text_to_tokens(text: str, vocab: Vocabulary) -> list[int]:
    """Convert text into token ids using an existing vocabulary."""
    return [vocab.stoi[char] for char in text]


def tokens_to_text(tokens: list[int], vocab: Vocabulary) -> str:
    """Convert token ids back into text."""
    return "".join(vocab.itos[token] for token in tokens)
