# Cloud Run serving image for the FastAPI model API.
#
# All model artifacts are baked in at BUILD time (pulled from the HF Hub over
# GCP's fast network), so cold starts do ZERO network I/O — the container only
# loads weights from local disk. Listens on Cloud Run's $PORT.
#
# Deploy:  gcloud run deploy nlp-pipeline-api --source . --region us-central1 \
#            --memory 4Gi --cpu 2 --allow-unauthenticated --timeout 600
#
# (HF Docker Spaces now require a paid PRO plan, so this replaced the Space plan.)

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
    MODEL_CKPT_PATH=/app/model/best_model.pt

WORKDIR /app

# CPU-only torch (much smaller than the CUDA build) + slim serving deps.
COPY requirements-api.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements-api.txt

# Pre-bake every model artifact so runtime does no downloads:
#  - trained multi-task weights, - roberta-base backbone + tokenizer, - NER model.
RUN python -c "from huggingface_hub import hf_hub_download; hf_hub_download('shiva-1993/nlp-pipeline-multitask', 'best_model.pt', local_dir='/app/model')" \
 && python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('roberta-base'); AutoModel.from_pretrained('roberta-base')" \
 && python -c "from transformers import pipeline; pipeline('ner', model='dslim/bert-base-NER', aggregation_strategy='simple')"

COPY api ./api
COPY src ./src
COPY configs ./configs

EXPOSE 8080
CMD exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8080}
