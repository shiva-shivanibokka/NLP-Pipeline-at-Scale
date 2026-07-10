"""
Tests for the SPC (statistical process control) anomaly detector.

Runs fully in-memory (no Redis) by pointing the aggregator at an unreachable
port so it uses its documented in-memory fallback.
"""

from src.aggregation.redis_store import RedisAggregator, BrandStats


def _agg():
    # Unreachable port → in-memory fallback (no Redis needed in CI).
    return RedisAggregator(host="127.0.0.1", port=1)


def test_alert_fires_on_sharp_sentiment_drop():
    agg = _agg()
    stats = BrandStats(brand_id="brand:test")
    # Stable positive history (small but nonzero variance).
    for v in [0.8, 0.82, 0.78, 0.81, 0.79] * 7:  # 35 points
        stats.sentiment_history.append((0.0, v))
    stats.ema_sentiment = -0.9  # sharp drop well below the historical mean

    alert = None
    for _ in range(3):  # needs SPC_CONSECUTIVE_WINDOWS breaches in a row
        a = agg._check_spc(stats)
        if a:
            alert = a
    assert alert is not None, "a >2σ drop over 3 windows should fire an alert"
    assert alert.z_score < -2.0


def test_no_alert_on_stable_stream():
    agg = _agg()
    stats = BrandStats(brand_id="brand:stable")
    for v in [0.5, 0.52, 0.48, 0.51, 0.49] * 7:
        stats.sentiment_history.append((0.0, v))
    stats.ema_sentiment = 0.5  # right on the mean

    for _ in range(5):
        assert agg._check_spc(stats) is None


def test_no_alert_with_zero_variance():
    # All-identical history → std ≈ 0 → detector must not divide-by-zero or fire.
    agg = _agg()
    stats = BrandStats(brand_id="brand:flat")
    for _ in range(35):
        stats.sentiment_history.append((0.0, 1.0))
    stats.ema_sentiment = -1.0
    assert agg._check_spc(stats) is None
