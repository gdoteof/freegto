# freegto

Linux-first Texas Hold'em solver prototype with optional NVIDIA GPU acceleration via PyTorch CUDA.

## Features

- Configurable `2-max`, `6-max`, and `9-max` no-limit Texas Hold'em betting tree (preflop/flop/turn/river).
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

See `examples/hu_nlhe.json`, `examples/6max_nlhe.json`, and `examples/9max_nlhe.json`.

Key sections:

- `max_players`: table format for the spot (`2`, `6`, or `9`).
- `positions`: full seat list in clockwise order, with `BTN` and `BB` required, and `SB` required for `6-max`/`9-max`.
- `ranges`: hand range by position (`"77+,A2s+,KTo+,AQo+,JTs"`).
- `actions`: legal actions by street.
- `blind_structure`: small blind / big blind.

Position order drives turn order and blind posting. For example:

- `2-max`: `["BTN", "BB"]`
- `6-max`: `["UTG", "HJ", "CO", "BTN", "SB", "BB"]`
- `9-max`: `["UTG", "UTG1", "MP", "LJ", "HJ", "CO", "BTN", "SB", "BB"]`

## Notes

- Uses a built-in 7-card evaluator for showdown resolution.
- Uses sampled chance events during training.
- Exports average strategy to JSON.
