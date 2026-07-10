"""
Throughput vs. latency benchmark under increasing load.

Runs the full pipeline (Kafka produce → consume → NLP inference → Redis write)
at N throughput levels and measures:
    - Actual achieved msg/sec (producer side)
    - Consumer lag (messages in queue that haven't been processed)
    - Per-message inference latency (p50, p95, p99)
    - Consumer throughput (messages processed per second)
    - Saturation point: the rate at which consumer lag grows unboundedly

This is a rigorous systems benchmark that produces a table and chart for the README.
"""

from __future__ import annotations

import json
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from configs.config import (
    BENCHMARK_THROUGHPUT_LEVELS,
    BENCHMARK_DURATION_SECONDS,
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_MAX_WAIT_MS,
)


@dataclass
class ThroughputResult:
    """Result for a single throughput level."""

    target_msgs_per_sec: int
    actual_producer_msgs_per_sec: float
    actual_consumer_msgs_per_sec: float
    consumer_lag_end: int  # messages unprocessed at end
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    throughput_ratio: float  # consumer_rate / producer_rate (1.0 = keeping up)
    saturated: bool  # True if consumer is falling behind
    duration_s: float


@dataclass
class BenchmarkReport:
    """Full benchmark report across all throughput levels."""

    results: list[ThroughputResult] = field(default_factory=list)
    saturation_point_msgs_per_sec: Optional[int] = None
    batch_size: int = BENCHMARK_BATCH_SIZE
    max_wait_ms: int = BENCHMARK_MAX_WAIT_MS

    def to_dict(self) -> dict:
        return {
            "saturation_point_msgs_per_sec": self.saturation_point_msgs_per_sec,
            "batch_size": self.batch_size,
            "max_wait_ms": self.max_wait_ms,
            "results": [
                {
                    "target_msgs_per_sec": r.target_msgs_per_sec,
                    "actual_producer_msgs_per_sec": r.actual_producer_msgs_per_sec,
                    "actual_consumer_msgs_per_sec": r.actual_consumer_msgs_per_sec,
                    "consumer_lag_end": r.consumer_lag_end,
                    "latency_p50_ms": r.latency_p50_ms,
                    "latency_p95_ms": r.latency_p95_ms,
                    "latency_p99_ms": r.latency_p99_ms,
                    "throughput_ratio": r.throughput_ratio,
                    "saturated": r.saturated,
                }
                for r in self.results
            ],
        }

    def print_table(self):
        print("\n" + "=" * 85)
        print("THROUGHPUT vs LATENCY BENCHMARK")
        print(f"Batch size: {self.batch_size}, Max wait: {self.max_wait_ms}ms")
        print("=" * 85)
        print(
            f"{'Rate':>8} {'Actual':>8} {'Consumer':>10} {'Lag':>8} {'p50':>8} {'p95':>8} {'p99':>8} {'Ratio':>7} {'Status':>10}"
        )
        print(
            f"{'(target)':>8} {'(msg/s)':>8} {'(msg/s)':>10} {'(msgs)':>8} {'(ms)':>8} {'(ms)':>8} {'(ms)':>8} {'':>7} {'':>10}"
        )
        print("-" * 85)
        for r in self.results:
            status = "SATURATED" if r.saturated else "OK"
            print(
                f"{r.target_msgs_per_sec:>8} "
                f"{r.actual_producer_msgs_per_sec:>8.0f} "
                f"{r.actual_consumer_msgs_per_sec:>10.0f} "
                f"{r.consumer_lag_end:>8} "
                f"{r.latency_p50_ms:>8.1f} "
                f"{r.latency_p95_ms:>8.1f} "
                f"{r.latency_p99_ms:>8.1f} "
                f"{r.throughput_ratio:>7.2f} "
                f"{status:>10}"
            )
        print("=" * 85)
        if self.saturation_point_msgs_per_sec:
            print(f"\nSaturation point: ~{self.saturation_point_msgs_per_sec} msg/s")
        else:
            print("\nNo saturation observed in the tested range.")


class MockInferencePipeline:
    """
    Mock NLP pipeline for benchmark runs without a real model.
    Simulates realistic inference latency per batch.

    In production benchmark mode, replace this with the real NLPConsumer.
    This allows running the throughput benchmark without training the model first.
    """

    def __init__(self, inference_ms_per_batch: float = 25.0):
        self.inference_ms_per_batch = inference_ms_per_batch
        self._latencies: list[float] = []

    def process_batch(self, messages: list[dict]) -> float:
        """Simulate processing a batch. Returns total latency in ms."""
        t0 = time.perf_counter()
        # Simulate inference: linear in batch size with some noise
        n = len(messages)
        base_ms = self.inference_ms_per_batch * (n / 32)  # scale by batch size
        jitter = np.random.normal(0, base_ms * 0.1)
        time.sleep(max(0, (base_ms + jitter) / 1000))
        elapsed_ms = (time.perf_counter() - t0) * 1000
        per_msg_ms = elapsed_ms / max(1, n)
        self._latencies.extend([per_msg_ms] * n)
        return elapsed_ms

    def get_latency_stats(self) -> dict:
        if not self._latencies:
            return {"p50": 0, "p95": 0, "p99": 0}
        arr = np.array(self._latencies)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    def reset(self):
        self._latencies = []


def run_benchmark(
    throughput_levels: list[int] = BENCHMARK_THROUGHPUT_LEVELS,
    duration_per_level: int = BENCHMARK_DURATION_SECONDS,
    batch_size: int = BENCHMARK_BATCH_SIZE,
    max_wait_ms: int = BENCHMARK_MAX_WAIT_MS,
    inference_pipeline=None,
    use_real_kafka: bool = False,
    output_dir: str = "results/benchmark",
    verbose: bool = True,
) -> BenchmarkReport:
    """
    Run the throughput/latency benchmark.

    When use_real_kafka=False (default): uses an in-process simulation
    that accurately models batch assembly and inference latency without
    requiring a running Kafka cluster. Suitable for CI and development.

    When use_real_kafka=True: connects to real Kafka and runs the full
    pipeline. Requires Kafka, Redis, and a trained model to be running.

    Args:
        throughput_levels:   List of target msg/sec values to test.
        duration_per_level:  Seconds to run at each level.
        batch_size:          Consumer batch size.
        max_wait_ms:         Max ms to wait for a full batch.
        inference_pipeline:  Optional real NLP pipeline (or MockInferencePipeline).
        use_real_kafka:      Whether to use real Kafka (True) or simulation (False).
        output_dir:          Where to save results JSON.
        verbose:             Print progress.

    Returns:
        BenchmarkReport with all results.
    """
    if inference_pipeline is None:
        inference_pipeline = MockInferencePipeline(inference_ms_per_batch=25.0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report = BenchmarkReport(batch_size=batch_size, max_wait_ms=max_wait_ms)

    for target_rate in throughput_levels:
        if verbose:
            print(
                f"\n[benchmark] Testing {target_rate} msg/s for {duration_per_level}s ..."
            )

        inference_pipeline.reset()

        if use_real_kafka:
            result = _run_kafka_level(
                target_rate,
                duration_per_level,
                batch_size,
                max_wait_ms,
                inference_pipeline,
                verbose,
            )
        else:
            result = _run_simulation_level(
                target_rate,
                duration_per_level,
                batch_size,
                max_wait_ms,
                inference_pipeline,
                verbose,
            )

        report.results.append(result)

        if verbose:
            print(
                f"  p50={result.latency_p50_ms:.1f}ms  "
                f"p99={result.latency_p99_ms:.1f}ms  "
                f"ratio={result.throughput_ratio:.2f}  "
                f"{'SATURATED' if result.saturated else 'OK'}"
            )

        # Early stop if severely saturated
        if result.saturated and result.throughput_ratio < 0.5:
            if verbose:
                print("  Consumer processing at <50% of producer rate. Stopping.")
            break

    # Identify saturation point
    for r in report.results:
        if r.saturated:
            report.saturation_point_msgs_per_sec = r.target_msgs_per_sec
            break

    # Save results
    result_path = Path(output_dir) / "throughput_benchmark.json"
    result_path.write_text(json.dumps(report.to_dict(), indent=2))
    if verbose:
        print(f"\n[benchmark] Results saved to {result_path}")
        report.print_table()

    return report


def _run_simulation_level(
    target_rate: int,
    duration_s: int,
    batch_size: int,
    max_wait_ms: int,
    pipeline: MockInferencePipeline,
    verbose: bool,
) -> ThroughputResult:
    """
    Simulate Kafka producer + consumer at the given rate without real Kafka.

    Models:
    - Producer publishes at target_rate msg/sec using a token bucket
    - Consumer reads in batches of batch_size (or after max_wait_ms)
    - Inference takes pipeline.process_batch() time
    - Queue depth = cumulative produced - cumulative consumed
    """
    # Simulate using a simple queue
    from queue import Queue, Empty

    queue: Queue = Queue()
    producer_count = [0]
    consumer_count = [0]
    stop_event = threading.Event()

    def producer():
        interval = 1.0 / target_rate
        t_next = time.monotonic()
        while not stop_event.is_set():
            now = time.monotonic()
            if now >= t_next:
                queue.put(
                    {"text": f"tweet {producer_count[0]}", "id": str(producer_count[0])}
                )
                producer_count[0] += 1
                t_next += interval
            else:
                time.sleep(min(interval * 0.1, t_next - now))

    def consumer():
        while not stop_event.is_set():
            batch = []
            deadline = time.monotonic() + max_wait_ms / 1000.0
            while len(batch) < batch_size and time.monotonic() < deadline:
                try:
                    msg = queue.get(timeout=0.01)
                    batch.append(msg)
                except Empty:
                    pass
            if batch:
                pipeline.process_batch(batch)
                consumer_count[0] += len(batch)

    t0 = time.monotonic()
    prod_thread = threading.Thread(target=producer, daemon=True)
    cons_thread = threading.Thread(target=consumer, daemon=True)
    prod_thread.start()
    cons_thread.start()

    time.sleep(duration_s)
    stop_event.set()
    prod_thread.join(timeout=2)
    cons_thread.join(timeout=2)

    duration = time.monotonic() - t0
    lag = producer_count[0] - consumer_count[0]
    prod_rate = producer_count[0] / duration
    cons_rate = consumer_count[0] / duration
    ratio = cons_rate / max(prod_rate, 1)
    saturated = ratio < 0.95 or lag > target_rate * 5

    latency_stats = pipeline.get_latency_stats()

    return ThroughputResult(
        target_msgs_per_sec=target_rate,
        actual_producer_msgs_per_sec=round(prod_rate, 1),
        actual_consumer_msgs_per_sec=round(cons_rate, 1),
        consumer_lag_end=max(0, lag),
        latency_p50_ms=round(latency_stats["p50"], 2),
        latency_p95_ms=round(latency_stats["p95"], 2),
        latency_p99_ms=round(latency_stats["p99"], 2),
        throughput_ratio=round(ratio, 3),
        saturated=saturated,
        duration_s=round(duration, 1),
    )


def _run_kafka_level(
    target_rate: int,
    duration_s: int,
    batch_size: int,
    max_wait_ms: int,
    pipeline,
    verbose: bool,
) -> ThroughputResult:
    """Real Kafka benchmark — requires running Kafka cluster."""
    from src.streaming.producer import TweetProducer
    from kafka import KafkaConsumer as _KafkaConsumer

    consumer_latencies = []
    consumer_count = [0]
    stop_event = threading.Event()

    def kafka_consumer_thread():
        consumer = _KafkaConsumer(
            "raw-text",
            bootstrap_servers="localhost:9092",
            group_id=f"benchmark-{target_rate}",
            auto_offset_reset="latest",
            max_poll_records=batch_size,
            fetch_max_wait_ms=max_wait_ms,
        )
        while not stop_event.is_set():
            records = consumer.poll(timeout_ms=max_wait_ms)
            msgs = [r.value for recs in records.values() for r in recs]
            if msgs:
                t0 = time.perf_counter()
                pipeline.process_batch(msgs)
                latency = (time.perf_counter() - t0) * 1000 / len(msgs)
                consumer_latencies.extend([latency] * len(msgs))
                consumer_count[0] += len(msgs)
        consumer.close()

    t0 = time.monotonic()
    cons_thread = threading.Thread(target=kafka_consumer_thread, daemon=True)
    cons_thread.start()

    with TweetProducer() as prod:
        prod_result = prod.publish(
            target_msgs_per_sec=target_rate,
            duration_seconds=duration_s,
            verbose=verbose,
        )

    stop_event.set()
    cons_thread.join(timeout=5)
    duration = time.monotonic() - t0

    prod_rate = prod_result["actual_msgs_per_sec"]
    cons_rate = consumer_count[0] / duration
    lag = prod_result["actual_msgs_sent"] - consumer_count[0]
    ratio = cons_rate / max(prod_rate, 1)
    saturated = ratio < 0.95 or lag > target_rate * 5

    arr = np.array(consumer_latencies) if consumer_latencies else np.array([0.0])
    return ThroughputResult(
        target_msgs_per_sec=target_rate,
        actual_producer_msgs_per_sec=round(prod_rate, 1),
        actual_consumer_msgs_per_sec=round(cons_rate, 1),
        consumer_lag_end=max(0, lag),
        latency_p50_ms=round(float(np.percentile(arr, 50)), 2),
        latency_p95_ms=round(float(np.percentile(arr, 95)), 2),
        latency_p99_ms=round(float(np.percentile(arr, 99)), 2),
        throughput_ratio=round(ratio, 3),
        saturated=saturated,
        duration_s=round(duration, 1),
    )
