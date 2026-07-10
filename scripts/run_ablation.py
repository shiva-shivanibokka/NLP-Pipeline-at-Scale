"""
Run the 3-way training strategy ablation.

Usage:
    python scripts/run_ablation.py                      # all 3 strategies
    python scripts/run_ablation.py --strategy uncertainty_weighted
    python scripts/run_ablation.py --strategy independent
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.quiet  # noqa: F401,E402 — sets USE_TF=0 etc. before transformers loads
from configs.config import TRAINING_STRATEGIES, TrainingConfig
from src.model.trainer import train_strategy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy", choices=list(TRAINING_STRATEGIES.keys()), default=None
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Fast end-to-end validation: 1 epoch, ~500 examples/task.",
    )
    args = parser.parse_args()

    strategies = [args.strategy] if args.strategy else list(TRAINING_STRATEGIES.keys())
    all_results = []

    for strat in strategies:
        if args.smoke:
            cfg = TrainingConfig(
                strategy=strat, num_epochs=1, max_examples_per_task=500
            )
        else:
            cfg = TrainingConfig(strategy=strat)
        result = train_strategy(cfg)
        all_results.append(result)

    # Print comparison table
    print("\n" + "=" * 75)
    print("ABLATION STUDY RESULTS")
    print("=" * 75)
    print(
        f"{'Strategy':25} {'Sentiment F1':>14} {'Emotion F1':>12} {'Toxicity F1':>13} {'Latency p99':>13} {'Params':>8}"
    )
    print("-" * 75)
    for r in all_results:
        m = r.get("metrics", {})
        s_f1 = m.get("sentiment", {}).get("f1_macro", 0)
        e_f1 = m.get("emotion", {}).get("f1_macro", 0)
        t_f1 = m.get("toxicity", {}).get("f1_macro", 0)
        lat = r.get("latency_ms", {}).get("p99_ms", 0)
        params = r.get("param_info", {}).get("total_M", 0)
        print(
            f"{r['strategy']:25} {s_f1:>14.4f} {e_f1:>12.4f} {t_f1:>13.4f} {lat:>12.1f}ms {params:>7.1f}M"
        )
    print("=" * 75)

    summary_path = Path("results/ablation/comparison.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
