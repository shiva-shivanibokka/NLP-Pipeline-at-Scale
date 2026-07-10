"""
Run the throughput vs. latency benchmark.

Usage:
    # Simulation mode (no Kafka required, good for CI/dev)
    python scripts/run_benchmark.py

    # Real Kafka mode (requires Docker Compose running)
    python scripts/run_benchmark.py --real-kafka

    # Quick test with fewer levels
    python scripts/run_benchmark.py --levels 100 500 1000
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.quiet  # noqa: F401,E402 — sets USE_TF=0 etc. before transformers loads
from src.benchmark.throughput import run_benchmark
from configs.config import (
    BENCHMARK_THROUGHPUT_LEVELS,
    BENCHMARK_BATCH_SIZE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-kafka", action="store_true")
    parser.add_argument("--levels", nargs="+", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=BENCHMARK_BATCH_SIZE)
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    levels = args.levels or BENCHMARK_THROUGHPUT_LEVELS

    report = run_benchmark(
        throughput_levels=levels,
        duration_per_level=args.duration,
        batch_size=args.batch_size,
        use_real_kafka=args.real_kafka,
        verbose=True,
    )
    report.print_table()


if __name__ == "__main__":
    main()
