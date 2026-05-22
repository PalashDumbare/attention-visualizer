"""Command-line entry point for the attention weight visualizer."""

from src.pipeline import build_attention_weights
from src.visualization import plot_attention_weights


def main() -> None:
    labels, weights = build_attention_weights()
    plot_attention_weights(labels, weights)


if __name__ == "__main__":
    main()
