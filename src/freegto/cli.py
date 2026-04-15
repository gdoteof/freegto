from __future__ import annotations

import argparse

from .cfr import CFRSolver
from .config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Linux-first NVIDIA GPU Texas Hold'em solver")
    parser.add_argument("--config", required=True, help="Path to JSON config")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--out", default="strategy.json")
    parser.add_argument("--max-depth", type=int, default=10)
    args = parser.parse_args()

    cfg = load_config(args.config)
    solver = CFRSolver(cfg, device=args.device, max_depth=args.max_depth)
    if args.device == "cuda" and solver.device != "cuda":
        print("CUDA requested but unavailable (or torch not installed). Falling back to CPU.")
    print(f"Running on device: {solver.device}")
    solver.train(iterations=args.iterations)
    solver.export_average_strategy(args.out)
    print(f"Wrote average strategy: {args.out}")


if __name__ == "__main__":
    main()
