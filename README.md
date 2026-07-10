# NLP Pipeline at Scale — Real-Time Social Listening

**▶️ Live demo:** [frontend-shiv-a.vercel.app](https://frontend-shiv-a.vercel.app) — type any text, get real-time multi-task predictions.
Frontend on **Vercel** (Next.js) → model API on **Google Cloud Run** (FastAPI + trained RoBERTa), all on free tiers.
*(Scale-to-zero backend: the first request after idle takes ~15–20s to warm up.)*

A production-grade NLP pipeline for real-time social media analysis. Processes a simulated tweet stream via **Kafka**, runs **multi-task RoBERTa** inference (joint sentiment + emotion + toxicity in a single forward pass), extracts named entities with brand normalization, assigns topics via **online incremental BERTopic**, and surfaces **brand sentiment anomalies** via Statistical Process Control.

Three empirical studies with measured results:
1. **Multi-task training ablation** — independent vs. hard-sharing vs. uncertainty-weighted loss (Kendall et al. 2018)
2. **Throughput/latency benchmark** — degradation curve from 100→2000 msg/sec, saturation point identified
3. **Active learning comparison** — entropy-based uncertainty sampling vs. random sampling, accuracy vs. labels curve

---

## Key Results

Real numbers from a full run on a single RTX 4060 (roberta-base, 5 epochs, fp16).
Reproduce with `python scripts/run_ablation.py`, `run_benchmark.py`,
`run_active_learning.py` (add `--smoke` for a fast 1-epoch sanity run).

### Multi-Task Ablation Study

Macro-F1 on the **held-out test split** of each task (`tweet_eval/sentiment`,
`dair-ai/emotion`, `tweet_eval/hate`). All three strategies use a pretrained
`roberta-base` backbone and identical 2-layer MLP heads; the only variable is
backbone **sharing** and loss **weighting**.

| Strategy | Sentiment F1 | Emotion F1 | Toxicity F1 | Latency p99 | Total Params |
|---|---|---|---|---|---|
| Independent (3 separate models) | **0.709** | 0.879 | **0.491** | 27.0 ms (3× fwd) | 373 M |
| Hard sharing (equal weights) | 0.700 | 0.884 | 0.478 | 12.6 ms | 125 M |
| **Uncertainty weighted (ours)** | 0.700 | **0.888** | 0.473 | **12.4 ms** | 125 M |

**Takeaway:** one shared backbone matches three separate models within ~1 F1 point
on every task while using **3× fewer parameters and ~2× lower latency** (single
forward pass). Uncertainty weighting edges out the best emotion F1. The efficiency
win is the headline; accuracy is a wash. (Toxicity F1 is low across the board — the
`tweet_eval/hate` test set has a well-known train→test distribution shift; validation
F1 was ~0.78. We report **test** to stay honest.)

> **Methodology note:** all strategies select the checkpoint on **validation** and
> report **test** F1. The independent baseline selects each task's best epoch
> *independently* (3 separate models), while the multi-task strategies must pick a
> single checkpoint on the *average* validation F1 — a mild structural advantage for
> the independent arm, inherent to joint training.

### Throughput vs. Latency Benchmark

Simulated producer→consumer pipeline (batched inference, batch=32, max_wait=100ms).

| Target (msg/s) | Consumer (msg/s) | p50 | p99 | Ratio | Status |
|---|---|---|---|---|---|
| 100 | 99.9 | 0.83 ms | 1.01 ms | 1.00 | OK |
| 500 | 499.0 | 0.79 ms | 0.97 ms | 1.00 | OK |
| 1000 | 999.8 | 0.79 ms | 0.98 ms | 1.00 | OK |
| 1500 | 1250.1 | 0.80 ms | 0.97 ms | 0.83 | **SATURATED** |
| 2000 | 1251.4 | 0.80 ms | 0.98 ms | 0.63 | **SATURATED** |

**Saturation point ≈ 1500 msg/s** — the consumer keeps up 1:1 up to ~1000 msg/s and
plateaus at a max sustained ~1250 msg/s, after which queue lag grows unbounded.

### Active Learning: Uncertainty Sampling vs. Random

Sentiment task, seed 200 labels, +50/round for 10 rounds; F1 on the validation set.

| Labeled Examples | Uncertainty F1 | Random F1 |
|---|---|---|
| 200 (seed) | 0.202 | 0.202 |
| 350 | 0.533 | 0.536 |
| 500 | 0.607 | 0.616 |
| 650 | 0.632 | 0.611 |
| 700 (full) | 0.631 | 0.624 |

**Takeaway (honest, mixed):** with this small query batch, entropy-based uncertainty
sampling is **statistically on par with random** — it leads late (650 labels) but
trails mid-curve. A useful negative result: uncertainty sampling is not a free win at
this scale, and random is a strong baseline. Both reach ~0.63 F1 with 700 labels.

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
