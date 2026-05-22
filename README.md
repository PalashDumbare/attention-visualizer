# Attention Weight Visualizer

A small Python project for visualizing character-level self-attention weights.

The app takes text, builds a character vocabulary, creates token embeddings,
computes scaled dot-product attention, and renders the attention weights as a
heatmap.

## Project Structure

```text
app.py            # Application entry point
pipeline.py       # Wires tokenization, embeddings, and attention together
tokenizer.py      # Builds vocabulary and converts text to token ids
embeddings.py     # Creates embedding matrices and looks up token embeddings
attention.py      # Computes attention weights
visualization.py  # Plots attention weights as a heatmap
```

## Import Flow

The project keeps imports one-directional:

```text
app.py       ->  pipeline.py
app.py       ->  visualization.py
pipeline.py  ->  tokenizer.py
pipeline.py  ->  embeddings.py
pipeline.py  ->  attention.py
```

`app.py` is the only file that runs the full program. The lower-level modules
provide focused helpers and do not need to know about plotting or app startup.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python app.py
```

By default, the visualizer uses:

```python
"Cat sat on a mat"
```

You can change the default text and embedding dimension in `pipeline.py`.

## Requirements

- Python 3.10+
- NumPy
- Matplotlib
