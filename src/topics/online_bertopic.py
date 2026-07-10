"""
Online incremental BERTopic for streaming topic discovery.

Standard BERTopic requires a full corpus at fit time — not usable in a
real-time streaming pipeline. BERTopic 0.16+ supports incremental/online
learning via topic_model.partial_fit(), which updates the c-TF-IDF
representations without re-running UMAP or HDBSCAN from scratch.

This module wraps BERTopic's online learning feature with:
1. An initial warm-up phase on a seed corpus (to establish the topic space)
2. Incremental updates as new batches arrive
3. Topic velocity tracking: which topics are gaining/losing share over time
4. A sliding window memory (last N documents per topic) for context
"""

from __future__ import annotations

from collections import defaultdict, deque

from bertopic import BERTopic
from bertopic.vectorizers import OnlineCountVectorizer
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from umap import UMAP

from configs.config import (
    BERTOPIC_EMBEDDING_MODEL,
    BERTOPIC_MIN_TOPIC_SIZE,
    BERTOPIC_UMAP_MIN_DIST,
    BERTOPIC_UMAP_N_COMPONENTS,
    BERTOPIC_UMAP_N_NEIGHBORS,
)


class OnlineBERTopic:
    """
    Streaming-capable BERTopic wrapper.

    Lifecycle:
        1. warm_up(seed_docs)  — fit initial topic model on ~1000 seed docs
        2. update(docs)        — incremental update with new batch
        3. transform_batch(texts) — assign topic IDs to new texts
        4. get_trending_topics() — topic velocity over sliding window
    """

    def __init__(
        self,
        min_topic_size: int = BERTOPIC_MIN_TOPIC_SIZE,
        embedding_model: str = BERTOPIC_EMBEDDING_MODEL,
        seed: int = 42,
    ):
        self.min_topic_size = min_topic_size
        self.seed = seed
        self._fitted = False
        self._update_count = 0
        self._doc_count = 0

        # Sliding window: topic_id → deque of timestamps
        self._topic_timestamps: dict[int, deque] = defaultdict(
            lambda: deque(maxlen=500)
        )

        # Embedding model (shared between fit and transform)
        self._embed_model = SentenceTransformer(embedding_model)

        # UMAP: low n_components for clustering quality
        self._umap = UMAP(
            n_neighbors=BERTOPIC_UMAP_N_NEIGHBORS,
            n_components=BERTOPIC_UMAP_N_COMPONENTS,
            min_dist=BERTOPIC_UMAP_MIN_DIST,
            metric="cosine",
            random_state=seed,
            low_memory=True,
        )

        # HDBSCAN: min_cluster_size controls topic granularity
        self._hdbscan = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        # OnlineCountVectorizer: decays old term counts over time
        # decay=0.01 means terms from 100 batches ago contribute 37% of their original weight
        self._vectorizer = OnlineCountVectorizer(
            stop_words="english",
            decay=0.01,
            min_df=2,
        )

        self.model = BERTopic(
            embedding_model=self._embed_model,
            umap_model=self._umap,
            hdbscan_model=self._hdbscan,
            vectorizer_model=self._vectorizer,
            calculate_probabilities=False,  # faster, not needed for streaming
            verbose=False,
        )

    def warm_up(self, seed_docs: list[str]) -> dict:
        """
        Initial fit on a seed corpus to establish the topic space.
        Should be called once before streaming begins, with ~1000+ docs.

        Returns:
            Dict with num_topics, topic_sizes
        """
        if not seed_docs:
            return {"num_topics": 0, "topic_sizes": {}}

        print(f"[bertopic] Warming up on {len(seed_docs)} seed documents...")
        topics, _ = self.model.fit_transform(seed_docs)
        self._fitted = True
        self._doc_count = len(seed_docs)

        topic_info = self.model.get_topic_info()
        print(
            f"[bertopic] Found {len(topic_info) - 1} initial topics "
            f"(-1 = outlier cluster)"
        )
        return {
            "num_topics": len(topic_info) - 1,
            "topic_sizes": {
                int(row["Topic"]): int(row["Count"])
                for _, row in topic_info.iterrows()
                if row["Topic"] != -1
            },
        }

    def update(self, docs: list[str]) -> None:
        """
        Incremental update with a new batch of documents.
        Uses BERTopic.partial_fit() to update c-TF-IDF without re-fitting UMAP.
        """
        if not docs:
            return

        if not self._fitted:
            self.warm_up(docs)
            return

        self.model.partial_fit(docs)
        self._update_count += 1
        self._doc_count += len(docs)

    def transform_batch(self, texts: list[str]) -> list[int]:
        """
        Assign topic IDs to a batch of new texts.

        Returns:
            List of topic IDs (-1 = outlier/noise).
        """
        if not self._fitted or not texts:
            return [-1] * len(texts)

        try:
            topics, _ = self.model.transform(texts)
            import time as _t

            ts = _t.time()
            for topic_id in topics:
                self._topic_timestamps[int(topic_id)].append(ts)
            return [int(t) for t in topics]
        except Exception:
            return [-1] * len(texts)

    def get_topic_info(self) -> list[dict]:
        """Return topic info: ID, label, top words, count."""
        if not self._fitted:
            return []
        try:
            info = self.model.get_topic_info()
            results = []
            for _, row in info.iterrows():
                topic_id = int(row["Topic"])
                if topic_id == -1:
                    continue
                words = self.model.get_topic(topic_id)
                results.append(
                    {
                        "topic_id": topic_id,
                        "label": str(row.get("Name", f"Topic {topic_id}")),
                        "count": int(row["Count"]),
                        "top_words": [w for w, _ in words[:8]] if words else [],
                    }
                )
            return results
        except Exception:
            return []

    def get_trending_topics(
        self,
        window_seconds: float = 300,
        baseline_seconds: float = 1800,
    ) -> list[dict]:
        """
        Return topics sorted by velocity: (short_window_count - baseline_count) / baseline_count.

        Positive velocity = topic is gaining share.
        """
        import time as _t

        now = _t.time()
        short_cutoff = now - window_seconds
        baseline_cutoff = now - baseline_seconds

        topic_info = {t["topic_id"]: t for t in self.get_topic_info()}
        velocities = []

        for topic_id, timestamps in self._topic_timestamps.items():
            short_count = sum(1 for ts in timestamps if ts >= short_cutoff)
            baseline_count = sum(1 for ts in timestamps if ts >= baseline_cutoff)

            if baseline_count == 0:
                velocity = 0.0
            else:
                velocity = (
                    short_count - baseline_count / (baseline_seconds / window_seconds)
                ) / (baseline_count / (baseline_seconds / window_seconds) + 1)

            info = topic_info.get(topic_id, {})
            velocities.append(
                {
                    "topic_id": topic_id,
                    "label": info.get("label", f"Topic {topic_id}"),
                    "top_words": info.get("top_words", []),
                    "short_count": short_count,
                    "velocity": round(velocity, 4),
                }
            )

        return sorted(velocities, key=lambda x: -x["velocity"])[:10]
