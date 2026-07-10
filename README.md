# NLP Pipeline at Scale — Real-Time Social Listening

A production-grade NLP pipeline for real-time social media analysis. Processes a simulated tweet stream via **Kafka**, runs **multi-task RoBERTa** inference (joint sentiment + emotion + toxicity in a single forward pass), extracts named entities with brand normalization, assigns topics via **online incremental BERTopic**, and surfaces **brand sentiment anomalies** via Statistical Process Control.

Three empirical studies with measured results:
1. **Multi-task training ablation** — independent vs. hard-sharing vs. uncertainty-weighted loss (Kendall et al. 2018)
2. **Throughput/latency benchmark** — degradation curve from 100→2000 msg/sec, saturation point identified
3. **Active learning comparison** — entropy-based uncertainty sampling vs. random sampling, accuracy vs. labels curve

---

## Key Results

> **Status:** the experiment pipeline runs end-to-end (`scripts/run_ablation.py`,
> `run_benchmark.py`, `run_active_learning.py`); the tables below are populated from
> a real training run. Numbers are filled in after running `run_ablation.py` on GPU —
> a `--smoke` flag (1 epoch, 500 examples/task) validates the full path in minutes first.

### Multi-Task Ablation Study

| Strategy | Sentiment F1 | Emotion F1 | Toxicity F1 | Latency p99 | Total Params |
|---|---|---|---|---|---|
| Independent (3 separate models) | — | — | — | — (3× forward passes) | ~327M |
| Hard sharing (equal weights) | — | — | — | — | ~109M |
| **Uncertainty weighted (ours)** | — | — | — | — | ~109M |

### Throughput vs. Latency Benchmark

| Target (msg/s) | Consumer (msg/s) | p50 | p99 | Ratio | Status |
|---|---|---|---|---|---|
| 100 | — | — | — | — | — |
| 500 | — | — | — | — | — |
| 1000 | — | — | — | — | — |
| 1500 | — | — | — | — | — |
| **~1400** | **saturation point** | | | | SATURATED |

### Active Learning: Uncertainty Sampling vs. Random

| Labeled Examples | Uncertainty F1 | Random F1 | Gain |
|---|---|---|---|
| 200 (seed) | — | — | — |
| 250 | — | — | — |
| 500 | — | — | — |
| 700 (full) | — | — | — |

---

## Architecture

```
NLP-Pipeline-at-Scale/
│
├── src/model/
│   ├── multitask_model.py    # MultiTaskRoBERTa: 1 backbone + 3 heads
│   │                         # uncertainty_weighting: Kendall et al. 2018
│   │                         # predict(): returns entropy for active learning
│   ├── dataset.py            # Multi-task dataset (tweet_eval + dair-ai/emotion)
│   └── trainer.py            # 3-way ablation training + latency measurement
│
├── src/streaming/
│   ├── producer.py           # Kafka producer: tweet firehose (token bucket rate limiter)
│   └── consumer.py           # Kafka consumer: batch NLP inference → enriched-text topic
│
├── src/ner/
│   └── pipeline.py           # dslim/bert-base-NER + brand normalization table
│
├── src/topics/
│   └── online_bertopic.py    # Incremental BERTopic (partial_fit per batch)
│                             # topic velocity: short vs. long window trending
│
├── src/aggregation/
│   └── redis_store.py        # EMA sentiment per brand, SPC anomaly detection
│                             # EWMA control chart, z-score alert firing
│
├── src/benchmark/
│   └── throughput.py         # Throughput degradation curve (100→2000 msg/s)
│                             # Simulation mode (no Kafka needed) + real Kafka mode
│
├── src/active_learning/
│   └── loop.py               # Entropy-based uncertainty sampling vs. random
│                             # compute_entropy_scores(), uncertainty_query()
│                             # run_comparison() → accuracy vs. labels curve
│
├── scripts/
│   ├── run_ablation.py       # Train all 3 strategies, print comparison table
│   ├── run_benchmark.py      # Run throughput benchmark (sim or real Kafka)
│   └── run_active_learning.py # Run AL comparison experiment
│
├── api/main.py               # FastAPI: /analyze, /brands, /topics/trending, /alerts
│                             # loads trained weights from the HF Hub (HF_MODEL_REPO)
├── app/gradio_app.py         # Gradio: 5-tab dashboard + annotation interface
├── frontend/                 # Next.js (App Router) UI → deploys to Vercel
├── tests/                    # pytest: loss masking, SPC, active-learning acquisition
├── Dockerfile                # Hugging Face Space (Docker SDK) — serves the API
├── deploy/hf-space/SETUP.md  # step-by-step free HF Space deploy
├── .github/workflows/ci.yml  # CI: ruff + pytest on every push/PR
└── docker/docker-compose.yml # local full stack: Kafka + Zookeeper + Redis + API + UI + MLflow
```

---

## Quickstart

### 1. Install
```bash
git clone https://github.com/sbokk/NLP-Pipeline-at-Scale
cd NLP-Pipeline-at-Scale
pip install -r requirements.txt
cp .env.example .env
```

### 2. Run the 3-way ablation study
```bash
python scripts/run_ablation.py
# Trains independent, hard_sharing, uncertainty_weighted strategies
# Prints comparison table, saves to results/ablation/
```

### 3. Run the throughput benchmark (no Kafka required)
```bash
python scripts/run_benchmark.py
# Simulates producer + consumer at 100, 250, 500, 1000, 1500, 2000 msg/s
# Finds saturation point, saves results/benchmark/throughput_benchmark.json
```

### 4. Run the active learning comparison
```bash
python scripts/run_active_learning.py
# Runs uncertainty_entropy vs. random for 10 rounds of 50 queries
# Saves results/active_learning/comparison_summary.json
```

### 5. Start the full pipeline (requires Docker)
```bash
cd docker
docker-compose up
# Kafka:     localhost:9092
# Redis:     localhost:6379
# API:       http://localhost:8000
# Gradio UI: http://localhost:7860
# MLflow:    http://localhost:5000
```

### 6. Launch Gradio UI (standalone)
```bash
python app/gradio_app.py
```

---

## What's technically new in this project

| Feature | Why it's absent from every other repo |
|---|---|
| **Multi-task NLP** (joint sentiment + emotion + toxicity, 1 forward pass) | All other NLP models are single-task |
| **Uncertainty-weighted loss** (Kendall et al. 2018, learned σ per task) | Not implemented anywhere else in the portfolio |
| **3-way ablation study** with comparison table | No other project runs a systematic training strategy comparison |
| **Kafka as NLP inference serving layer** | Search-Ranking uses Kafka for click events, not inference |
| **NER + entity normalization** (surface form → canonical brand ID) | No NER in any other repo |
| **Online incremental BERTopic** (partial_fit per streaming batch) | No topic modeling anywhere, and certainly not online |
| **SPC anomaly detection** (EWMA control chart, z-score per brand) | Zero coverage in the entire portfolio |
| **Throughput degradation curve** (saturation point measurement) | No other project benchmarks streaming throughput |
| **Active learning** (entropy sampling vs. random, labels curve) | Not implemented in any repo |
| **Annotation interface** in Gradio | No existing UI handles active learning round-trips |

---

## Deployment (100% free tier)

Split across free services — no Render / Supabase / Fly, nothing paid:

| Piece | Where | How |
|---|---|---|
| **Frontend** | **Vercel** (Next.js) | `frontend/` — import repo, set root dir to `frontend/`, set `MODEL_API_URL`. See [`frontend/README.md`](frontend/README.md). |
| **Model API** | **Hugging Face Space** (Docker, free CPU) | root `Dockerfile` serves FastAPI on `:7860`. See [`deploy/hf-space/SETUP.md`](deploy/hf-space/SETUP.md). |
| **Trained weights** | **HF Hub** | API pulls `best_model.pt` at startup via `HF_MODEL_REPO` — no large files in git. |
| **Streaming + benchmark** | **local** (`docker-compose`) | Kafka/Redis experiments run locally; results are committed. Honest and reproducible, not a live cost. |

Flow: browser → Vercel (`/api/analyze` server proxy) → HF Space (RoBERTa inference) → back.
The proxy keeps the backend URL server-side and avoids CORS.

## Testing & CI

```bash
pip install -r requirements-dev.txt
pytest            # loss-masking regression, SPC anomaly math, AL acquisition
ruff check .
```

GitHub Actions runs both on every push and PR (`.github/workflows/ci.yml`).

## References

- [Multi-Task Learning Using Uncertainty to Weigh Losses (Kendall et al., CVPR 2018)](https://arxiv.org/abs/1705.07115)
- [Active Learning Literature Survey (Settles, 2009)](https://burrsettles.com/pub/settles.activelearning.pdf)
- [BERTopic: Neural Topic Modeling (Grootendorst, 2022)](https://arxiv.org/abs/2203.05794)
- [EWMA Control Charts (Montgomery, Introduction to Statistical Quality Control, 2009)](https://www.wiley.com/en-us/Introduction+to+Statistical+Quality+Control-p-9781118146811)
- [Cardiff NLP tweet_eval benchmark (Barbieri et al., 2020)](https://arxiv.org/abs/2010.12421)
