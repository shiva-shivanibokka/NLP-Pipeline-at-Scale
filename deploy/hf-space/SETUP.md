# Deploy the model API to Hugging Face Spaces (free)

The FastAPI backend runs on a **free** HF Space (Docker SDK, CPU). The Vercel
frontend calls it. Nothing here needs Render / Supabase / Fly.

## One-time setup

1. **Create the Space**
   - Go to https://huggingface.co/new-space
   - Owner: your account · Space name: `nlp-pipeline-api`
   - SDK: **Docker** · Hardware: **CPU basic (free)** · Visibility: Public
   - Create.

2. **Push this repo to the Space** (the Space is just a git repo). From the
   project root:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/nlp-pipeline-api
   git push space main
   ```
   HF auto-detects the root `Dockerfile` and builds. First build takes a few
   minutes (installs torch + transformers). When it's green, the API is live at:
   `https://<your-username>-nlp-pipeline-api.hf.space`

3. **Point it at your trained weights** (after training — Step 3 of the overhaul).
   In the Space → **Settings → Variables and secrets**, add:
   | Name | Value |
   |---|---|
   | `HF_MODEL_REPO` | `<your-username>/nlp-pipeline-multitask` (the model repo you push weights to) |

   The API downloads `best_model.pt` from that repo on startup
   (see `api/main.py::_resolve_checkpoint`). Until then the Space runs but
   predictions are from an untrained backbone (it logs a warning).

## Verify

```bash
curl https://<your-username>-nlp-pipeline-api.hf.space/health
# {"status":"ok","version":"0.1.0"}

curl -X POST https://<your-username>-nlp-pipeline-api.hf.space/analyze \
  -H "Content-Type: application/json" \
  -d '{"text":"I love the new update from Apple!","include_topics":false}'
```

## Notes

- `/analyze`, `/health` work standalone. `/brands`, `/topics/trending`, `/alerts`
  need the streaming pipeline (Kafka+Redis) populated, which runs **locally** —
  they return empty on the Space, by design.
- Free Spaces sleep after inactivity; the first request after a sleep is slow
  (cold model load). That's expected on the free tier.
- To publish trained weights to the Hub, from the machine that trained:
  ```bash
  huggingface-cli login
  huggingface-cli upload <your-username>/nlp-pipeline-multitask \
    results/ablation/uncertainty_weighted/best_model.pt best_model.pt
  ```
