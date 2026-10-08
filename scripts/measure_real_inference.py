"""Measure the REAL per-batch inference cost of the shipped multi-task model.

The committed benchmark uses `MockInferencePipeline`, whose `process_batch` is
`time.sleep(25ms * n/32)`. This measures the same batch shape through the actual
`MultiTaskRoBERTa` forward pass, so the published throughput can be restated.

**Why this reports a range rather than a single number.** An earlier version
reported one median per device and called it "stable to about 1.5%", based on two
runs taken minutes apart. An independent re-run in a later session came back 9%
higher. This is a laptop GPU with unlocked clocks and dynamic boost: the
run-to-run spread *within* a session is small and the spread *across* sessions is
not, so a same-session repeat measures repeatability and says nothing about
reproducibility. Reporting a point estimate from it was wrong.

This script now runs `--reps` independent repetitions per device and reports
median, min and max, and writes them to `results/benchmark/real_inference.json`
so nothing downstream has to hardcode a figure.

Batch shape is the repo's own configured one (BENCHMARK_BATCH_SIZE=32,
MAX_SEQ_LENGTH=128), matching `api/main.py` and `src/streaming/consumer.py`.
Tokenization is excluded, so these are **lower bounds** on end-to-end cost.
Model heads are randomly initialised when no checkpoint is present; this does not
affect timing, because cost is set by architecture rather than weight values.

Run from the repo root.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.quiet  # noqa: F401,E402 -- sets USE_TF=0 before transformers loads
import torch  # noqa: E402

from src.model.multitask_model import MultiTaskRoBERTa  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

BATCH = 32
MAX_LEN = 128
WARMUP = 3
ITERS = 15
MOCK_MS_PER_BATCH = 25.0

OUT_PATH = Path("results/benchmark/real_inference.json")

SAMPLE = (
    "the new update broke my workflow completely and support has not replied "
    "in three days which is honestly unacceptable for a paid product"
)


def _one_rep(model, enc, device: str) -> float:
    """Median ms/batch over ITERS timed forward passes, after WARMUP."""

    def once() -> float:
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(WARMUP):
        once()
    return statistics.median(once() for _ in range(ITERS))


def bench(device: str, reps: int) -> dict:
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = MultiTaskRoBERTa(model_name="roberta-base").to(device).eval()
    enc = tok(
        [SAMPLE] * BATCH,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    reps_ms = [_one_rep(model, enc, device) for _ in range(reps)]
    med = statistics.median(reps_ms)
    return {
        "device": device,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "batch_size": BATCH,
        "max_seq_len": MAX_LEN,
        "reps": reps,
        "iters_per_rep": ITERS,
        "per_batch_ms_median": round(med, 1),
        "per_batch_ms_min": round(min(reps_ms), 1),
        "per_batch_ms_max": round(max(reps_ms), 1),
        "per_batch_ms_all_reps": [round(x, 1) for x in reps_ms],
        "per_message_ms_median": round(med / BATCH, 2),
        "msgs_per_sec_one_worker_median": round(BATCH / (med / 1000.0), 1),
        "msgs_per_sec_one_worker_min": round(BATCH / (max(reps_ms) / 1000.0), 1),
        "msgs_per_sec_one_worker_max": round(BATCH / (min(reps_ms) / 1000.0), 1),
        "ratio_vs_mock_median": round(med / MOCK_MS_PER_BATCH, 1),
        "ratio_vs_mock_min": round(min(reps_ms) / MOCK_MS_PER_BATCH, 1),
        "ratio_vs_mock_max": round(max(reps_ms) / MOCK_MS_PER_BATCH, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--reps",
        type=int,
        default=5,
        help="independent repetitions per device (default 5). More reps widen "
        "the observed range; they do not make a laptop GPU's clocks stable.",
    )
    ap.add_argument("--cpu-only", action="store_true")
    args = ap.parse_args()

    print(
        f"MOCK (as published): {MOCK_MS_PER_BATCH} ms/batch of {BATCH} "
        f"= {MOCK_MS_PER_BATCH / BATCH:.2f} ms/msg "
        f"= {BATCH / (MOCK_MS_PER_BATCH / 1000):.0f} msg/s single worker\n"
    )

    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices.insert(0, "cuda")

    out = []
    for dev in devices:
        r = bench(dev, args.reps)
        out.append(r)
        print(
            f"{dev.upper():>5} ({r['device_name']}): "
            f"median {r['per_batch_ms_median']} ms/batch "
            f"[{r['per_batch_ms_min']}-{r['per_batch_ms_max']} over "
            f"{r['reps']} reps]  "
            f"{r['msgs_per_sec_one_worker_median']} msg/s  "
            f"{r['ratio_vs_mock_min']}-{r['ratio_vs_mock_max']}x the mock"
        )

    payload = {
        "note": (
            "Independent repetitions in ONE process. Across separate sessions "
            "this laptop GPU has shown a further ~10% spread (104-114 ms "
            "observed for the same batch shape), so treat the range as "
            "indicative and never quote a single figure as reproducible."
        ),
        "mock_ms_per_batch": MOCK_MS_PER_BATCH,
        "measurements": out,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
