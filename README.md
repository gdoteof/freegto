# freegto

Linux-first Texas Hold'em solver prototype with optional NVIDIA GPU acceleration via PyTorch CUDA.

## Features

- Heads-up no-limit Texas Hold'em betting tree (preflop/flop/turn/river).
- Configurable **ranges per position** (BTN/SB/BB, etc.) via common range notation.
- Configurable **actions per street** (`check`, `call`, `fold`, `all-in`, `bet:<fraction>`).
- Monte-Carlo CFR style training with regret-matching.
- CUDA mode is enabled when `torch` is installed with GPU support and `--device cuda` is used.

> This is a practical starter implementation. It uses abstraction/sampling and is not a production-grade full-game solve.

## Quick start (Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
# optional GPU backend
pip install -e .[gpu]
```

Run a solve:

```bash
holdem-solve --config examples/hu_nlhe.json --iterations 2000 --device cuda
```

If no NVIDIA GPU is available, use `--device cpu`.

## Config format

See `examples/hu_nlhe.json`.

Key sections:

- `positions`: names + initial stack.
- `ranges`: hand range by position (`"77+,A2s+,KTo+,AQo+,JTs"`).
- `actions`: legal actions by street.
- `blind_structure`: small blind / big blind.

## Notes

- Uses a built-in 7-card evaluator for showdown resolution.
- Uses sampled chance events during training.
- Exports average strategy to JSON.
