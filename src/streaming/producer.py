"""
Kafka producer: simulates a social media text firehose.

Reads tweets from cardiffnlp/tweet_eval and publishes them to the
'raw-text' Kafka topic at a configurable throughput (msg/sec).

Each message is a JSON object:
    {
        "id":        "uuid4 string",
        "text":      "tweet text",
        "timestamp": "ISO-8601 UTC",
        "source":    "twitter_simulated",
        "true_sentiment": 0|1|2   # kept for benchmark evaluation only
    }

The producer is rate-limited via a token bucket so throughput is accurate
even when the event loop is uneven.
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from typing import Iterator

from kafka import KafkaProducer

from configs.config import (
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_RAW_TOPIC,
    SENTIMENT_DATASET,
    SENTIMENT_SUBSET,
)


def _tweet_stream(repeat: bool = True) -> Iterator[dict]:
    """Infinite iterator over tweet_eval/sentiment (repeats if repeat=True)."""
    from datasets import load_dataset  # lazy: avoids torch/pyarrow OpenMP clash on import

    ds = load_dataset(SENTIMENT_DATASET, SENTIMENT_SUBSET, split="train")
    texts = list(ds["text"])
    labels = list(ds["label"])
    i = 0
    while True:
        yield {"text": texts[i % len(texts)], "true_sentiment": labels[i % len(labels)]}
        i += 1
        if not repeat and i >= len(texts):
            break


class TweetProducer:
    """
    Kafka producer that publishes tweets at a target throughput.

    Uses a token bucket rate limiter to smooth out bursts and maintain
    stable msg/sec measurements during the benchmark.
    """

    def __init__(
        self,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        topic: str = KAFKA_RAW_TOPIC,
    ):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks="all",
            compression_type="gzip",
            batch_size=16_384,  # 16 KB batch — good balance of latency/throughput
            linger_ms=5,  # wait up to 5ms to fill a batch
            buffer_memory=33_554_432,  # 32 MB producer buffer
        )

    def publish(
        self,
        target_msgs_per_sec: int = 1000,
        duration_seconds: int = 60,
        verbose: bool = True,
    ) -> dict:
        """
        Publish tweets at the target throughput for the given duration.

        Args:
            target_msgs_per_sec: Target publish rate.
            duration_seconds:    How long to run.
            verbose:             Print progress every 5 seconds.

        Returns:
            Dict with actual_msgs_sent, actual_msgs_per_sec, duration_s.
        """
        stream = _tweet_stream(repeat=True)
        interval = 1.0 / target_msgs_per_sec  # seconds between messages
        t_start = time.monotonic()
        t_end = t_start + duration_seconds
        sent = 0
        t_next = t_start
        t_report = t_start + 5.0

        while time.monotonic() < t_end:
            now = time.monotonic()

            # Token bucket: sleep until next token is available
            if now < t_next:
                time.sleep(t_next - now)
            t_next = max(time.monotonic(), t_next) + interval

            tweet = next(stream)
            msg = {
                "id": str(uuid.uuid4()),
                "text": tweet["text"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "source": "twitter_simulated",
                "true_sentiment": tweet["true_sentiment"],
            }
            self.producer.send(self.topic, value=msg)
            sent += 1

            if verbose and time.monotonic() >= t_report:
                elapsed = time.monotonic() - t_start
                print(
                    f"[producer] {sent} msgs in {elapsed:.1f}s "
                    f"({sent / elapsed:.0f} msg/s, target={target_msgs_per_sec})"
                )
                t_report = time.monotonic() + 5.0

        self.producer.flush()
        actual_duration = time.monotonic() - t_start

        return {
            "actual_msgs_sent": sent,
            "actual_msgs_per_sec": round(sent / actual_duration, 1),
            "target_msgs_per_sec": target_msgs_per_sec,
            "duration_s": round(actual_duration, 2),
        }

    def close(self):
        self.producer.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
