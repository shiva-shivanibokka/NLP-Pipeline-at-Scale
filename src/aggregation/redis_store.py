"""
Redis-backed real-time aggregation and SPC anomaly detection.

Maintains per-brand rolling statistics using Redis sorted sets and strings:
    - Exponential moving average (EMA) of sentiment score
    - Sliding window counts (5-min and 30-min)
    - Toxicity rate (rolling average)
    - Message volume

Statistical Process Control (SPC) anomaly detection:
    An EWMA-style control check. The EMA is the smoothed (control) signal; the
    control limits come from the mean/std of the recent sentiment history. An
    alert fires when the EMA sits more than THRESHOLD standard deviations below
    that mean on SPC_CONSECUTIVE_WINDOWS consecutive observations (updates for
    the brand) — not fixed clock-time windows.

    # ponytail: per-observation check over the in-memory history deque, not a
    # true time-bucketed EWMA chart. Fine for a demo; swap in fixed time buckets
    # if you need alerting that's independent of message arrival rate.

Reference:
    Montgomery, D.C. (2009). Introduction to Statistical Quality Control, 6th ed.
    Chapter 9: CUSUM and EWMA Control Charts.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from configs.config import (
    REDIS_DB,
    REDIS_HOST,
    REDIS_PORT,
    SENTIMENT_EMA_ALPHA,
    SHORT_WINDOW_SECONDS,
    LONG_WINDOW_SECONDS,
    SPC_ALERT_THRESHOLD,
    SPC_CONSECUTIVE_WINDOWS,
)

# Sentiment label to numeric score
SENTIMENT_SCORE = {"negative": -1.0, "neutral": 0.0, "positive": 1.0}


@dataclass
class BrandStats:
    """In-memory rolling stats per brand, backed by Redis."""

    brand_id: str
    ema_sentiment: float = 0.0
    sentiment_history: deque = field(default_factory=lambda: deque(maxlen=1000))
    message_count: int = 0
    toxicity_count: int = 0
    alert_streak: int = 0  # consecutive below-threshold windows
    active_alert: bool = False
    last_alert_time: Optional[float] = None


@dataclass
class SentimentAlert:
    brand_id: str
    triggered_at: float
    ema_sentiment: float
    historical_mean: float
    z_score: float
    consecutive_windows: int


class RedisAggregator:
    """
    Manages real-time NLP signal aggregation in Redis.

    Falls back to in-memory only if Redis is not available.
    This allows the pipeline to run without Redis in dev/test environments.
    """

    def __init__(
        self,
        host: str = REDIS_HOST,
        port: int = REDIS_PORT,
        db: int = REDIS_DB,
    ):
        self._brand_stats: dict[str, BrandStats] = {}
        self._alerts: list[SentimentAlert] = []
        self._redis = None
        self._redis_available = False

        # Try to connect to Redis; fall back to in-memory if unavailable
        try:
            import redis

            r = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2)
            r.ping()
            self._redis = r
            self._redis_available = True
            print(f"[redis] Connected to {host}:{port}/db{db}")
        except Exception as e:
            print(f"[redis] Not available ({e}) — using in-memory fallback")

    def update(self, enriched_record: dict) -> Optional[SentimentAlert]:
        """
        Update rolling statistics for all brands mentioned in a record.
        Checks SPC alert conditions.

        Args:
            enriched_record: Output from the consumer's _enrich_record().

        Returns:
            SentimentAlert if an anomaly was detected, else None.
        """
        sentiment_str = enriched_record.get("sentiment", "neutral")
        sentiment_val = SENTIMENT_SCORE.get(sentiment_str, 0.0)
        is_toxic = enriched_record.get("toxicity") == "toxic"
        entities = enriched_record.get("entities", [])
        timestamp = time.time()

        # Collect brand IDs mentioned in this record
        brand_ids = set()
        for ent in entities:
            cid = ent.get("canonical_id")
            if cid and cid.startswith("brand:"):
                brand_ids.add(cid)

        # Always update a "global" aggregate
        brand_ids.add("global")

        alert = None
        for brand_id in brand_ids:
            if brand_id not in self._brand_stats:
                self._brand_stats[brand_id] = BrandStats(brand_id=brand_id)

            stats = self._brand_stats[brand_id]
            stats.message_count += 1
            if is_toxic:
                stats.toxicity_count += 1

            # EMA update: S_t = α * x_t + (1-α) * S_{t-1}
            if stats.message_count == 1:
                stats.ema_sentiment = sentiment_val
            else:
                stats.ema_sentiment = (
                    SENTIMENT_EMA_ALPHA * sentiment_val
                    + (1 - SENTIMENT_EMA_ALPHA) * stats.ema_sentiment
                )

            stats.sentiment_history.append((timestamp, sentiment_val))

            # SPC check
            new_alert = self._check_spc(stats)
            if new_alert:
                alert = new_alert
                self._alerts.append(new_alert)

            # Write to Redis if available
            if self._redis_available:
                self._write_to_redis(stats, timestamp)

        return alert

    def _check_spc(self, stats: BrandStats) -> Optional[SentimentAlert]:
        """
        EWMA control chart check.

        Alert fires when:
            z_score = (current_ema - historical_mean) / historical_std < -THRESHOLD
        for SPC_CONSECUTIVE_WINDOWS consecutive evaluations.
        """
        history = stats.sentiment_history
        if len(history) < 30:  # need at least 30 points to compute stable stats
            return None

        scores = [s for _, s in history]
        historical_mean = sum(scores[:-5]) / len(
            scores[:-5]
        )  # exclude last 5 for comparison
        historical_std = (
            sum((s - historical_mean) ** 2 for s in scores[:-5]) / len(scores[:-5])
        ) ** 0.5

        if historical_std < 1e-6:  # all same value — no meaningful variance
            return None

        current_ema = stats.ema_sentiment
        z_score = (current_ema - historical_mean) / historical_std

        if z_score < -SPC_ALERT_THRESHOLD:
            stats.alert_streak += 1
        else:
            stats.alert_streak = 0
            stats.active_alert = False

        if stats.alert_streak >= SPC_CONSECUTIVE_WINDOWS and not stats.active_alert:
            stats.active_alert = True
            stats.last_alert_time = time.time()
            return SentimentAlert(
                brand_id=stats.brand_id,
                triggered_at=time.time(),
                ema_sentiment=round(current_ema, 4),
                historical_mean=round(historical_mean, 4),
                z_score=round(z_score, 4),
                consecutive_windows=stats.alert_streak,
            )

        return None

    def _write_to_redis(self, stats: BrandStats, timestamp: float) -> None:
        """Persist aggregated stats to Redis with TTL."""
        key = f"brand:{stats.brand_id}"
        data = {
            "ema_sentiment": round(stats.ema_sentiment, 4),
            "message_count": stats.message_count,
            "toxicity_rate": round(
                stats.toxicity_count / max(1, stats.message_count), 4
            ),
            "updated_at": timestamp,
            "active_alert": int(stats.active_alert),
        }
        self._redis.setex(key, 3600, json.dumps(data))  # 1-hour TTL

        # Track 5-minute sentiment timeseries in a sorted set (score=timestamp)
        ts_key = f"brand:{stats.brand_id}:sentiment_ts"
        self._redis.zadd(
            ts_key,
            {
                json.dumps(
                    {"ts": timestamp, "s": round(stats.ema_sentiment, 4)}
                ): timestamp
            },
        )
        # Prune entries older than 30 minutes
        cutoff = timestamp - LONG_WINDOW_SECONDS
        self._redis.zremrangebyscore(ts_key, "-inf", cutoff)
        self._redis.expire(ts_key, 3600)

    # ── Public read API ────────────────────────────────────────────────────────

    def get_brand_stats(self, brand_id: str) -> Optional[dict]:
        """Get current stats for a brand."""
        if brand_id in self._brand_stats:
            stats = self._brand_stats[brand_id]
            return {
                "brand_id": brand_id,
                "ema_sentiment": round(stats.ema_sentiment, 4),
                "message_count": stats.message_count,
                "toxicity_rate": round(
                    stats.toxicity_count / max(1, stats.message_count), 4
                ),
                "active_alert": stats.active_alert,
                "alert_streak": stats.alert_streak,
            }
        return None

    def get_all_brands(self) -> list[dict]:
        """Get stats for all tracked brands, sorted by message count."""
        results = []
        for brand_id, stats in self._brand_stats.items():
            if brand_id == "global":
                continue
            results.append(
                {
                    "brand_id": brand_id,
                    "ema_sentiment": round(stats.ema_sentiment, 4),
                    "message_count": stats.message_count,
                    "toxicity_rate": round(
                        stats.toxicity_count / max(1, stats.message_count), 4
                    ),
                    "active_alert": stats.active_alert,
                }
            )
        return sorted(results, key=lambda x: -x["message_count"])

    def get_global_stats(self) -> dict:
        """Get global (cross-brand) stats."""
        g = self._brand_stats.get("global")
        if not g:
            return {}
        return {
            "ema_sentiment": round(g.ema_sentiment, 4),
            "message_count": g.message_count,
            "toxicity_rate": round(g.toxicity_count / max(1, g.message_count), 4),
        }

    def get_active_alerts(self) -> list[dict]:
        """Return all currently active anomaly alerts."""
        active = [stats for stats in self._brand_stats.values() if stats.active_alert]
        return [
            {
                "brand_id": s.brand_id,
                "ema_sentiment": round(s.ema_sentiment, 4),
                "alert_streak": s.alert_streak,
                "triggered_at": s.last_alert_time,
            }
            for s in active
        ]

    def get_sentiment_timeseries(
        self,
        brand_id: str,
        window_seconds: float = SHORT_WINDOW_SECONDS,
    ) -> list[dict]:
        """Get recent EMA sentiment values for a brand."""
        if brand_id not in self._brand_stats:
            return []
        stats = self._brand_stats[brand_id]
        cutoff = time.time() - window_seconds
        return [
            {"timestamp": ts, "sentiment": s}
            for ts, s in stats.sentiment_history
            if ts >= cutoff
        ]
