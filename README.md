# Artificial Intelligence Engines — Companion Code

[![CI](https://github.com/mengrenman/artificial_intelligence_engines/actions/workflows/ci.yml/badge.svg)](https://github.com/mengrenman/artificial_intelligence_engines/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A from-scratch Python implementation of the algorithms in
**James V Stone (2020), *Artificial Intelligence Engines: A Tutorial Introduction
to the Mathematics of Deep Learning*** (Sebtel Press, ISBN 9781916279117).

The aim is pedagogical: every algorithm is built in plain NumPy so the
mathematics is visible, and each chapter has a notebook that walks through
the equations and reproduces a figure from the book.

## Layout

```
.
├── aie/            # Shared code: activations, losses, layers, datasets, gradcheck, plotting
├── scripts/        # Importable, tested algorithm modules, one per chapter
├── notebooks/      # One Jupyter notebook per chapter (01_… through 11_…) + a PyTorch appendix,
│                   # stored with outputs
├── tests/          # pytest suite: gradient checks + behavioural tests
└── data/           # Downloaded / generated data, e.g. MNIST (gitignored)
```

## Setup

Requires Python 3.10+ (3.12 pinned in `.python-version`). With [uv](https://docs.astral.sh/uv/),
which installs the exact versions in `uv.lock`:

```bash
uv sync --extra dev                 # add `--extra torch` for the PyTorch appendix (CPU wheels)
uv run jupyter lab
```

or with plain pip: `python3.12 -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.

**No install:** click the *Open in Colab* badge on any notebook (table below). The first cell
installs this package into the Colab runtime.

Everything runs offline on a CPU by default. Chapters 7–9 use `aie.data.digits()`, a small
synthetic stand-in for MNIST (12×12 noisy, shifted digit glyphs). The real MNIST digits are
available via `aie.data.mnist()` (an 11 MB one-off download, cached in `data/mnist/`, or in
`$AIE_DATA_DIR` if set); notebook 09 has an optional MNIST section that runs once the data is cached:

```bash
uv run python -c "from aie.data import mnist; mnist('train'); mnist('test')"
```

## Checks

```bash
uv run ruff check .                            # lint
uv run pytest                                  # unit tests (~10 s; PyTorch tests skip if torch is absent)
uv run pytest --nbmake notebooks               # execute every notebook (~1–2 min)
```

CI (`.github/workflows/ci.yml`) runs all three on Python 3.10 and 3.12, plus a separate job
for the PyTorch tests and appendix notebook.

## Status

| # | Chapter | Notebook | Script |
|---|---------|----------|--------|
| 1 | Artificial Neural Networks | [`01_introduction`](notebooks/01_introduction.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/01_introduction.ipynb) | — (intro) |
| 2 | Linear Associative Networks | [`02_linear_associative`](notebooks/02_linear_associative.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/02_linear_associative.ipynb) | `linear.py` |
| 3 | Perceptrons | [`03_perceptron`](notebooks/03_perceptron.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/03_perceptron.ipynb) | `perceptron.py` |
| 4 | The Backpropagation Algorithm | [`04_backprop`](notebooks/04_backprop.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/04_backprop.ipynb) | `backprop.py` |
| 5 | Hopfield Nets | [`05_hopfield`](notebooks/05_hopfield.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/05_hopfield.ipynb) | `hopfield.py` |
| 6 | Boltzmann Machines | [`06_boltzmann`](notebooks/06_boltzmann.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/06_boltzmann.ipynb) | `boltzmann.py` |
| 7 | Deep RBMs | [`07_rbm`](notebooks/07_rbm.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/07_rbm.ipynb) | `rbm.py` |
| 8 | Variational Autoencoders | [`08_vae`](notebooks/08_vae.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/08_vae.ipynb) | `vae.py` |
| 9 | Deep Backprop Networks | [`09_deep_backprop`](notebooks/09_deep_backprop.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/09_deep_backprop.ipynb) | `deep_backprop.py` (uses `aie/layers.py`) |
| 10 | Reinforcement Learning | [`10_reinforcement`](notebooks/10_reinforcement.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/10_reinforcement.ipynb) | `reinforcement.py` |
| 11 | The Emperor's New AI? | [`11_discussion`](notebooks/11_discussion.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/11_discussion.ipynb) | — (discussion) |
| A | PyTorch cross-check of Ch 8–9 | [`appendix_pytorch`](notebooks/appendix_pytorch.ipynb) [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mengrenman/artificial_intelligence_engines/blob/main/notebooks/appendix_pytorch.ipynb) | `torch_models.py` (optional `torch` extra) |

## Conventions

- **Loss scaling.** Squared-error models minimise
  E = (1/T) Σ_t ½ Σ_k (y_tk − ŷ_tk)², i.e. the book's Eq 2.5 / 4.8 averaged over the
  T training associations (`aie.losses.mean_sse`). Its gradient is exactly
  `aie.losses.mean_sse_grad`, and every model's gradient is checked against finite
  differences (`aie.gradcheck`) in the test suite.
- **Notation.** The book writes targets in sans-serif y and outputs in italic y; PDF text
  extraction loses the distinction. Read correctly, Eq 3.4 is the standard perceptron rule
  Δw = ε · target · x on misclassified inputs.
- **Output units in Chapter 4.** `BackpropNetwork` uses a linear output layer by default
  (Section 4.3); `output="sigmoid"` gives the sigmoidal output units of the book's XOR
  example (Figs 4.5–4.6), which solves 0/1 XOR in under 40 iterations.
- **Deviations from the book** (typos, unspecified parameters, scaled-down networks) are
  documented in a "Notes" section of each script's docstring.
- **Shapes.** Batch dimension first. Chapters 2–4 use the book's W (K × I) layout; the layer
  library in `aie/layers.py` uses W (n_in × n_out) so batched code needs no transposes.

## Notes

The book's own companion code (Python and MATLAB, contributed from various
sources) lives at <https://github.com/jgvfwstone/ArtificialIntelligenceEngines>.
This repo is independent and has a different goal: to present a single
consistent NumPy implementation alongside the book's exposition.

The book itself is copyrighted and is not part of this repository; `*.pdf` is
gitignored. Keep your own copy alongside the code if you like.

## License

The code is released under the [MIT License](LICENSE). The book text and figures are not.
