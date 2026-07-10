"""
Run the active learning comparison experiment.

Usage:
    python scripts/run_active_learning.py              # both strategies
    python scripts/run_active_learning.py --strategy uncertainty_entropy
    python scripts/run_active_learning.py --rounds 5 --query-size 25
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.quiet  # noqa: F401,E402 — sets USE_TF=0 etc. before transformers loads
from src.active_learning.loop import run_comparison, run_active_learning
from configs.config import AL_NUM_ROUNDS, AL_QUERY_BATCH_SIZE, AL_SEED_LABELED_SIZE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=["uncertainty_entropy", "random"], default=None
    )
    parser.add_argument("--rounds", type=int, default=AL_NUM_ROUNDS)
    parser.add_argument("--query-size", type=int, default=AL_QUERY_BATCH_SIZE)
    parser.add_argument("--seed-size", type=int, default=AL_SEED_LABELED_SIZE)
    args = parser.parse_args()

    if args.strategy:
        run_active_learning(
            strategy=args.strategy,
            num_rounds=args.rounds,
            query_batch_size=args.query_size,
            seed_size=args.seed_size,
            verbose=True,
        )
    else:
        run_comparison(
            num_rounds=args.rounds,
            query_batch_size=args.query_size,
            seed_size=args.seed_size,
            verbose=True,
        )


if __name__ == "__main__":
    main()
