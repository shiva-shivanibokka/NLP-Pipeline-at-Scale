"""
FastAPI serving layer.

Endpoints:
    POST /analyze              — synchronous single-text multi-task inference
    GET  /brands               — all tracked brands with rolling stats
    GET  /brands/{brand_id}    — stats for a specific brand
    GET  /topics/trending      — top-10 trending topics (last 5 minutes)
    GET  /alerts               — active SPC anomaly alerts
    GET  /health               — health check
    GET  /pipeline/stats       — consumer throughput and lag
"""

from __future__ import annotations

import os

# Quiet TF/Flax probing before transformers is (lazily) imported — keeps logs clean.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

# ── Lazy model loading ─────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_ner = None
_aggregator = None
_topic_model = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        from transformers import AutoTokenizer
        from src.model.multitask_model import MultiTaskRoBERTa
        from configs.config import BASE_MODEL_ID

        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        _model = MultiTaskRoBERTa(uncertainty_weighting=True)

        ckpt = _resolve_checkpoint()
        if ckpt:
            _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
            print(f"[model] Loaded trained weights from {ckpt}")
        else:
            print(
                "[model] WARNING: no trained weights found — serving an untrained "
                "backbone (predictions are meaningless). Train and set HF_MODEL_REPO "
                "or MODEL_CKPT_PATH."
            )
        _model.eval()
    return _model, _tokenizer


def _resolve_checkpoint() -> Optional[str]:
    """
    Locate the trained multi-task checkpoint, in priority order:
      1. MODEL_CKPT_PATH env var (explicit local path)
      2. HF_MODEL_REPO env var → download best_model.pt from the Hugging Face Hub
         (this is how the deployed Space gets weights without committing a .pt)
      3. the default local training output path
    Returns a path, or None if nothing is available.
    """
    local = os.getenv("MODEL_CKPT_PATH")
    if local and os.path.exists(local):
        return local

    repo = os.getenv("HF_MODEL_REPO")
    if repo:
        try:
            from huggingface_hub import hf_hub_download

            filename = os.getenv("HF_MODEL_FILENAME", "best_model.pt")
            return hf_hub_download(repo_id=repo, filename=filename)
        except Exception as e:  # network / missing file → fall through to local
            print(f"[model] HF Hub download from {repo!r} failed: {e}")

    default = "results/ablation/uncertainty_weighted/best_model.pt"
    return default if os.path.exists(default) else None


def _get_ner():
    global _ner
    if _ner is None:
        from src.ner.pipeline import NERPipeline

        _ner = NERPipeline(device=-1)
    return _ner


def _get_aggregator():
    global _aggregator
    if _aggregator is None:
        from src.aggregation.redis_store import RedisAggregator

        _aggregator = RedisAggregator()
    return _aggregator


def _get_topic_model():
    global _topic_model
    if _topic_model is None:
        from src.topics.online_bertopic import OnlineBERTopic

        _topic_model = OnlineBERTopic()
    return _topic_model


# ── App setup ──────────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm up models on startup
    try:
        _get_model()
        _get_ner()
        _get_aggregator()
    except Exception as e:
        print(f"[startup] Model warm-up failed: {e} — will load lazily")
    yield


app = FastAPI(
    title="NLP Pipeline at Scale",
    description=(
        "Real-time multi-task NLP pipeline: sentiment + emotion + toxicity "
        "via a shared RoBERTa backbone with uncertainty-weighted loss. "
        "NER with entity normalization, online BERTopic, Redis aggregation, "
        "and SPC anomaly detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)


# ── Schemas ────────────────────────────────────────────────────────────────────


class AnalyzeRequest(BaseModel):
    text: str
    include_entities: bool = True
    include_topics: bool = True


class AnalyzeResponse(BaseModel):
    text: str
    sentiment: str
    sentiment_score: float
    sentiment_probs: dict[str, float]
    emotion: str
    emotion_score: float
    emotion_probs: dict[str, float]
    toxicity: str
    toxicity_score: float
    toxicity_probs: dict[str, float]
    uncertainty: float
    entities: list[dict]
    topic_id: int
    inference_latency_ms: float


# ── Endpoints ──────────────────────────────────────────────────────────────────


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """Run multi-task NLP inference on a single text."""
    model, tokenizer = _get_model()

    t0 = time.perf_counter()
    enc = tokenizer(
        request.text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    with torch.no_grad():
        preds = model.predict(enc["input_ids"], enc["attention_mask"])

    latency_ms = (time.perf_counter() - t0) * 1000

    from configs.config import SENTIMENT_LABELS, EMOTION_LABELS, TOXICITY_LABELS

    s_pred = int(preds["sentiment_pred"][0])
    e_pred = int(preds["emotion_pred"][0])
    t_pred = int(preds["toxicity_pred"][0])

    # Full per-class distributions so the UI can show every class, not just top-1.
    def _dist(labels, probs):
        return {lab: round(float(probs[i]), 4) for i, lab in enumerate(labels)}

    s_dist = _dist(SENTIMENT_LABELS, preds["sentiment_probs"][0])
    e_dist = _dist(EMOTION_LABELS, preds["emotion_probs"][0])
    t_dist = _dist(TOXICITY_LABELS, preds["toxicity_probs"][0])

    entities = []
    if request.include_entities:
        ner = _get_ner()
        entities = ner.extract(request.text)

    topic_id = -1
    if request.include_topics:
        tm = _get_topic_model()
        topic_ids = tm.transform_batch([request.text])
        topic_id = topic_ids[0] if topic_ids else -1

    # Update aggregator
    record = {
        "text": request.text,
        "sentiment": SENTIMENT_LABELS[s_pred],
        "toxicity": TOXICITY_LABELS[t_pred],
        "entities": entities,
        "id": "api_request",
        "timestamp": str(time.time()),
    }
    agg = _get_aggregator()
    agg.update(record)

    return AnalyzeResponse(
        text=request.text,
        sentiment=SENTIMENT_LABELS[s_pred],
        sentiment_score=round(float(preds["sentiment_probs"][0][s_pred]), 4),
        sentiment_probs=s_dist,
        emotion=EMOTION_LABELS[e_pred],
        emotion_score=round(float(preds["emotion_probs"][0][e_pred]), 4),
        emotion_probs=e_dist,
        toxicity=TOXICITY_LABELS[t_pred],
        toxicity_score=round(float(preds["toxicity_probs"][0][1]), 4),
        toxicity_probs=t_dist,
        uncertainty=round(float(preds["aggregate_entropy"][0]), 4),
        entities=entities,
        topic_id=topic_id,
        inference_latency_ms=round(latency_ms, 2),
    )


@app.get("/brands")
async def list_brands():
    return _get_aggregator().get_all_brands()


@app.get("/brands/{brand_id}")
async def get_brand(brand_id: str):
    stats = _get_aggregator().get_brand_stats(brand_id)
    if not stats:
        raise HTTPException(
            status_code=404, detail=f"Brand '{brand_id}' not tracked yet"
        )
    return stats


@app.get("/topics/trending")
async def trending_topics():
    return _get_topic_model().get_trending_topics()


@app.get("/alerts")
async def active_alerts():
    return _get_aggregator().get_active_alerts()


@app.get("/global")
async def global_stats():
    return _get_aggregator().get_global_stats()
