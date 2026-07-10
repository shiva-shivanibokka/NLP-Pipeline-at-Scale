"""
Central configuration for the NLP Pipeline at Scale.
All hyperparameters and infrastructure settings in one place.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# ── Datasets ──────────────────────────────────────────────────────────────────

# Primary dataset: Cardiff NLP tweet_eval — multiple tasks, real tweets
SENTIMENT_DATASET = "cardiffnlp/tweet_eval"
SENTIMENT_SUBSET = "sentiment"  # 3-class: positive/neutral/negative
EMOTION_DATASET = "dair-ai/emotion"  # 6-class: joy/sadness/anger/fear/love/surprise
TOXICITY_DATASET = "cardiffnlp/tweet_eval"
TOXICITY_SUBSET = "hate"  # binary: hate/not-hate (proxy for toxicity)

SENTIMENT_LABELS = ["negative", "neutral", "positive"]
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
TOXICITY_LABELS = ["not_toxic", "toxic"]

NUM_SENTIMENT_CLASSES = 3
NUM_EMOTION_CLASSES = 6
NUM_TOXICITY_CLASSES = 2

# ── Model ─────────────────────────────────────────────────────────────────────

BASE_MODEL_ID = "roberta-base"
MAX_SEQ_LENGTH = 128

# ── Training strategies (ablation) ────────────────────────────────────────────

TRAINING_STRATEGIES = {
    "independent": {
        "description": "Three separate single-task models (baseline)",
        "shared_backbone": False,
        "uncertainty_weighting": False,
    },
    "hard_sharing": {
        "description": "One backbone, three heads, fixed equal loss weights (1/3 each)",
        "shared_backbone": True,
        "uncertainty_weighting": False,
    },
    "uncertainty_weighted": {
        "description": "One backbone, three heads, loss weights learned via Kendall et al. 2018",
        "shared_backbone": True,
        "uncertainty_weighting": True,
    },
}


@dataclass
class TrainingConfig:
    strategy: str = "uncertainty_weighted"
    num_epochs: int = 5
    batch_size: int = 32
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_length: int = MAX_SEQ_LENGTH
    fp16: bool = True
    seed: int = 42
    output_dir: str = "results/ablation"
    mlflow_experiment: str = "nlp_pipeline_multitask"
    mlflow_tracking_uri: str = "mlruns"
    push_to_hub: bool = False
    hub_model_id: str = ""
    # If set, cap examples per task (used by `run_ablation.py --smoke` for a fast
    # end-to-end validation run; None = use the full datasets).
    max_examples_per_task: Optional[int] = None


# ── Kafka ─────────────────────────────────────────────────────────────────────

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"
KAFKA_RAW_TOPIC = "raw-text"
KAFKA_ENRICHED_TOPIC = "enriched-text"
KAFKA_CONSUMER_GROUP = "nlp-pipeline-consumer"

# Producer throughput levels for benchmark
BENCHMARK_THROUGHPUT_LEVELS = [100, 250, 500, 1000, 1500, 2000]  # msg/sec
BENCHMARK_DURATION_SECONDS = 60  # per throughput level
BENCHMARK_BATCH_SIZE = 32  # consumer inference batch
BENCHMARK_MAX_WAIT_MS = 100  # max wait to fill a batch before flushing

# ── NER ───────────────────────────────────────────────────────────────────────

NER_MODEL_ID = "dslim/bert-base-NER"
NER_ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]

# Brand/entity normalization lookup
# Maps surface forms → canonical brand IDs
BRAND_NORMALIZATION = {
    # Apple
    "apple": "brand:apple",
    "apple inc": "brand:apple",
    "aapl": "brand:apple",
    "$aapl": "brand:apple",
    "apple inc.": "brand:apple",
    # Google
    "google": "brand:google",
    "alphabet": "brand:google",
    "googl": "brand:google",
    "$googl": "brand:google",
    "google llc": "brand:google",
    # Meta
    "meta": "brand:meta",
    "facebook": "brand:meta",
    "instagram": "brand:meta",
    "fb": "brand:meta",
    "$meta": "brand:meta",
    # Amazon
    "amazon": "brand:amazon",
    "aws": "brand:amazon",
    "amzn": "brand:amazon",
    "$amzn": "brand:amazon",
    # Microsoft
    "microsoft": "brand:microsoft",
    "msft": "brand:microsoft",
    "$msft": "brand:microsoft",
    "azure": "brand:microsoft",
    # Tesla
    "tesla": "brand:tesla",
    "tsla": "brand:tesla",
    "$tsla": "brand:tesla",
    # Twitter/X
    "twitter": "brand:twitter",
    "x corp": "brand:twitter",
    "elon musk": "brand:twitter",
    # OpenAI
    "openai": "brand:openai",
    "chatgpt": "brand:openai",
    "gpt": "brand:openai",
    "gpt-4": "brand:openai",
    "gpt4": "brand:openai",
}


# ── BERTopic ──────────────────────────────────────────────────────────────────

BERTOPIC_MIN_TOPIC_SIZE = 10
BERTOPIC_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BERTOPIC_UMAP_N_NEIGHBORS = 15
BERTOPIC_UMAP_N_COMPONENTS = 5
BERTOPIC_UMAP_MIN_DIST = 0.0
ONLINE_BATCH_SIZE = 256  # docs per incremental BERTopic update

# ── Redis / aggregation ───────────────────────────────────────────────────────

REDIS_HOST = "localhost"
REDIS_PORT = 6379
REDIS_DB = 0

# Exponential moving average decay for rolling sentiment
SENTIMENT_EMA_ALPHA = 0.1
# Sliding window sizes (in seconds)
SHORT_WINDOW_SECONDS = 300  # 5-minute window
LONG_WINDOW_SECONDS = 1800  # 30-minute window
# SPC anomaly detection threshold (z-score)
SPC_ALERT_THRESHOLD = 2.0
SPC_CONSECUTIVE_WINDOWS = 3  # must breach threshold N times in a row


# ── Active learning ───────────────────────────────────────────────────────────

AL_UNLABELED_POOL_SIZE = 5000  # tweets in unlabeled pool
AL_QUERY_BATCH_SIZE = 50  # examples per active learning round
AL_NUM_ROUNDS = 10  # total annotation rounds
AL_SEED_LABELED_SIZE = 200  # initial labeled set size (per task)
# Acquisition functions to compare
AL_STRATEGIES = ["uncertainty_entropy", "random"]


# ── FastAPI ───────────────────────────────────────────────────────────────────

API_HOST = "0.0.0.0"
API_PORT = 8000
