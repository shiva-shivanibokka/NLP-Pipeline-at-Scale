"""
Kafka consumer: reads raw tweets, runs multi-task inference, publishes enriched records.

Pipeline per batch:
    1. Read up to BATCH_SIZE messages (or wait MAX_WAIT_MS, whichever comes first)
    2. Tokenise the batch
    3. Multi-task model forward pass → sentiment + emotion + toxicity
    4. NER → entity spans + brand normalization
    5. BERTopic → topic assignment (incremental update every N batches)
    6. Write enriched records to 'enriched-text' Kafka topic
    7. Write aggregated stats to Redis

Key design:
    - Batching is the primary latency knob. Larger batches = higher throughput,
      higher latency. The benchmark varies BATCH_SIZE to measure the tradeoff.
    - The NER model runs as a separate pipeline object (different architecture),
      so it's called sequentially after the multi-task inference.
    - BERTopic is updated every BERTOPIC_UPDATE_EVERY_N_BATCHES to amortise
      the cost of the UMAP+HDBSCAN step.
"""

from __future__ import annotations

import json
import time
from typing import Optional

import torch
from kafka import KafkaConsumer
from transformers import AutoTokenizer

from configs.config import (
    BASE_MODEL_ID,
    KAFKA_BOOTSTRAP_SERVERS,
    KAFKA_CONSUMER_GROUP,
    KAFKA_ENRICHED_TOPIC,
    KAFKA_RAW_TOPIC,
    MAX_SEQ_LENGTH,
    BENCHMARK_BATCH_SIZE,
    BENCHMARK_MAX_WAIT_MS,
    SENTIMENT_LABELS,
    EMOTION_LABELS,
    TOXICITY_LABELS,
)

BERTOPIC_UPDATE_EVERY_N_BATCHES = 20


class NLPConsumer:
    """
    Kafka consumer that enriches raw tweet messages with NLP annotations.

    Designed for minimal latency:
    - Tokenises on CPU (fast), infers on GPU if available
    - NER runs on the same device as the multi-task model
    - BERTopic is updated periodically, not per-batch
    """

    def __init__(
        self,
        multitask_model,
        ner_pipeline,
        topic_model,
        redis_client,
        tokenizer: Optional[AutoTokenizer] = None,
        bootstrap_servers: str = KAFKA_BOOTSTRAP_SERVERS,
        raw_topic: str = KAFKA_RAW_TOPIC,
        enriched_topic: str = KAFKA_ENRICHED_TOPIC,
        batch_size: int = BENCHMARK_BATCH_SIZE,
        max_wait_ms: int = BENCHMARK_MAX_WAIT_MS,
        device: str = "cpu",
    ):
        self.model = multitask_model
        self.ner = ner_pipeline
        self.topic_model = topic_model
        self.redis = redis_client
        self.device = device
        self.batch_size = batch_size
        self.max_wait_ms = max_wait_ms
        self.batch_count = 0
        self._text_buffer = []  # for incremental BERTopic

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(BASE_MODEL_ID)

        self.consumer = KafkaConsumer(
            raw_topic,
            bootstrap_servers=bootstrap_servers,
            group_id=KAFKA_CONSUMER_GROUP,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            auto_offset_reset="latest",
            enable_auto_commit=True,
            max_poll_records=batch_size,
            fetch_max_wait_ms=max_wait_ms,
            session_timeout_ms=30_000,
        )

        from kafka import KafkaProducer

        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            acks=1,  # fire-and-forget for the enriched topic
        )

    def _tokenise_batch(self, texts: list[str]) -> dict:
        enc = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def _run_inference(self, texts: list[str]) -> dict:
        """Multi-task forward pass on a batch of texts."""
        enc = self._tokenise_batch(texts)
        with torch.no_grad():
            preds = self.model.predict(enc["input_ids"], enc["attention_mask"])
        return preds

    def _enrich_record(
        self,
        raw_msg: dict,
        sentiment_pred: int,
        sentiment_probs: list[float],
        emotion_pred: int,
        emotion_probs: list[float],
        toxicity_pred: int,
        toxicity_score: float,
        aggregate_entropy: float,
        entities: list[dict],
        topic_id: int,
        inference_latency_ms: float,
    ) -> dict:
        return {
            "id": raw_msg["id"],
            "text": raw_msg["text"],
            "timestamp": raw_msg["timestamp"],
            "source": raw_msg["source"],
            # Multi-task predictions
            "sentiment": SENTIMENT_LABELS[sentiment_pred],
            "sentiment_score": round(sentiment_probs[sentiment_pred], 4),
            "sentiment_probs": {
                l: round(p, 4) for l, p in zip(SENTIMENT_LABELS, sentiment_probs)
            },
            "emotion": EMOTION_LABELS[emotion_pred],
            "emotion_score": round(emotion_probs[emotion_pred], 4),
            "toxicity": TOXICITY_LABELS[toxicity_pred],
            "toxicity_score": round(toxicity_score, 4),
            "uncertainty": round(aggregate_entropy, 4),
            # NER
            "entities": entities,
            # Topic
            "topic_id": topic_id,
            # Meta
            "inference_latency_ms": round(inference_latency_ms, 2),
        }

    def run_batch(self, messages: list[dict]) -> tuple[list[dict], float]:
        """
        Process one batch of raw messages end-to-end.

        Returns:
            (enriched_records, total_latency_ms)
        """
        texts = [m["text"] for m in messages]
        t0 = time.perf_counter()

        # 1. Multi-task inference
        preds = self._run_inference(texts)

        # 2. NER (per-document, batch if NER pipeline supports it)
        all_entities = self.ner.extract_batch(texts)

        # 3. BERTopic: buffer texts and update periodically
        self._text_buffer.extend(texts)
        if (
            self.batch_count % BERTOPIC_UPDATE_EVERY_N_BATCHES == 0
            and len(self._text_buffer) >= 256
        ):
            self.topic_model.update(self._text_buffer[-1024:])
            self._text_buffer = []
        topic_ids = self.topic_model.transform_batch(texts)

        inference_latency_ms = (time.perf_counter() - t0) * 1000 / len(texts)

        # 4. Build enriched records
        enriched = []
        for i, msg in enumerate(messages):
            s_probs = preds["sentiment_probs"][i].tolist()
            e_probs = preds["emotion_probs"][i].tolist()
            t_probs = preds["toxicity_probs"][i].tolist()

            record = self._enrich_record(
                raw_msg=msg,
                sentiment_pred=int(preds["sentiment_pred"][i]),
                sentiment_probs=s_probs,
                emotion_pred=int(preds["emotion_pred"][i]),
                emotion_probs=e_probs,
                toxicity_pred=int(preds["toxicity_pred"][i]),
                toxicity_score=t_probs[1],  # probability of toxic class
                aggregate_entropy=float(preds["aggregate_entropy"][i]),
                entities=all_entities[i],
                topic_id=int(topic_ids[i]),
                inference_latency_ms=inference_latency_ms,
            )
            enriched.append(record)

            # 5. Redis aggregation
            self.redis.update(record)

        self.batch_count += 1
        total_ms = (time.perf_counter() - t0) * 1000
        return enriched, total_ms

    def consume_forever(self, verbose: bool = True):
        """
        Main consume loop. Reads messages, enriches them, publishes to enriched topic.
        Runs until interrupted.
        """
        batch_buffer = []
        print(
            f"[consumer] Starting — batch_size={self.batch_size}, max_wait_ms={self.max_wait_ms}"
        )

        for msg in self.consumer:
            batch_buffer.append(msg.value)

            if len(batch_buffer) >= self.batch_size:
                enriched, latency_ms = self.run_batch(batch_buffer)
                for record in enriched:
                    self.producer.send(KAFKA_ENRICHED_TOPIC, value=record)
                if verbose:
                    print(
                        f"[consumer] Batch {self.batch_count}: "
                        f"{len(batch_buffer)} msgs, "
                        f"{latency_ms:.1f}ms total, "
                        f"{latency_ms / len(batch_buffer):.1f}ms/msg"
                    )
                batch_buffer = []

    def close(self):
        self.consumer.close()
        self.producer.close()
