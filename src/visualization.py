"""Plotting helpers for attention weights."""

import matplotlib.pyplot as plt
import numpy as np


def plot_attention_weights(labels: list[str], weights: np.ndarray) -> None:
    """Render attention weights as a heatmap."""
    _, axis = plt.subplots()
    image = axis.imshow(weights, cmap="viridis")

    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Key Tokens")
    axis.set_ylabel("Query Tokens")
    axis.set_title("Character-Level Self Attention")
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(
                j,
                i,
                f"{weights[i, j]:.2f}",
                ha="center",
                va="center",
                color="white",
                fontdict= {
                    "fontsize":6,
                }
            )
    
    plt.colorbar(image, ax=axis)
    plt.tight_layout()
    plt.savefig("assets/non_trained_attention_heatmap.png")
    plt.show()
    
