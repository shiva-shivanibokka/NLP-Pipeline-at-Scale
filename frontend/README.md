# Frontend — Next.js on Vercel

A minimal, dependency-light Next.js (App Router) UI for the multi-task analyzer.
It has **no client-side backend URL**: the browser calls `/api/analyze`, a
server route that proxies to the model API (a free Hugging Face Space). That
keeps the backend URL server-only and sidesteps CORS.

## Local dev

```bash
cd frontend
npm install
cp .env.example .env.local        # point MODEL_API_URL at your API
npm run dev                        # http://localhost:3000
```

You need the model API reachable at `MODEL_API_URL` — either the deployed HF
Space, or a local `uvicorn api.main:app --port 8000` from the repo root.

## Deploy to Vercel (free)

1. Push this repo to GitHub.
2. On https://vercel.com/new → import the repo.
3. **Set the project's Root Directory to `frontend/`** (Vercel setting).
4. Add an Environment Variable:
   | Key | Value |
   |---|---|
   | `MODEL_API_URL` | `https://<your-username>-nlp-pipeline-api.hf.space` |
5. Deploy. Vercel auto-detects Next.js and builds.

That's it — no Render/Supabase/Fly, all free tiers.

## What it shows

Single-text analysis only: sentiment / emotion / toxicity (with confidence),
named entities + canonical brand IDs, prediction entropy, and inference latency.
The brand-monitor, trending-topics, and anomaly-alert features depend on the
streaming pipeline (Kafka + Redis), which runs locally — not on the free Space.
