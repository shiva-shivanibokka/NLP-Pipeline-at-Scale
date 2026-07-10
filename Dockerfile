# Hugging Face Spaces (Docker SDK) entrypoint — serves the FastAPI model API.
#
# HF Spaces auto-detects a root-level Dockerfile, builds it, and routes public
# traffic to port 7860. This is the free-tier backend the Vercel frontend calls.
# For local multi-service dev (Kafka + Redis + UI), use docker/docker-compose.yml.
#
# Trained weights are pulled at runtime from the HF Hub — set the HF_MODEL_REPO
# variable in the Space settings (see deploy/hf-space/SETUP.md). No .pt in git.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

# HF Spaces run the container as a non-root user (uid 1000). Create it and make
# HOME writable so model/dataset caches don't hit permission errors.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app \
    HF_HOME=/home/user/.cache/huggingface \
    USE_TF=0 \
    USE_FLAX=0 \
    TRANSFORMERS_NO_ADVISORY_WARNINGS=1

WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
