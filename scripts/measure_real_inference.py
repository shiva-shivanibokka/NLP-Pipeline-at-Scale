"""Measure the REAL per-batch inference cost of the shipped multi-task model.

The committed benchmark used MockInferencePipeline, whose process_batch is
`time.sleep(25ms * n/32)`. This measures the same batch shape through the actual
MultiTaskRoBERTa forward pass so the published throughput can be restated.

Run from the repo root.
"""
from __future__ import annotations

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

SAMPLE = (
    "the new update broke my workflow completely and support has not replied "
    "in three days which is honestly unacceptable for a paid product"
)


def bench(device: str) -> dict:
    tok = AutoTokenizer.from_pretrained("roberta-base")
    model = MultiTaskRoBERTa(model_name="roberta-base").to(device).eval()

    enc = tok(
        [SAMPLE] * BATCH,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    ).to(device)

    def one() -> float:
        t0 = time.perf_counter()
        with torch.no_grad():
            model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        if device == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(WARMUP):
        one()
    samples = [one() for _ in range(ITERS)]

    per_batch = statistics.median(samples)
    return {
        "device": device,
        "batch_size": BATCH,
        "max_seq_len": MAX_LEN,
        "iters": ITERS,
        "per_batch_ms_median": round(per_batch, 1),
        "per_batch_ms_min": round(min(samples), 1),
        "per_batch_ms_max": round(max(samples), 1),
        "per_message_ms": round(per_batch / BATCH, 2),
        "max_msgs_per_sec_single_worker": round(BATCH / (per_batch / 1000.0), 1),
    }


def main() -> int:
    out = []
    mock_per_batch = 25.0
    print(f"MOCK (as published): {mock_per_batch} ms/batch of {BATCH} "
          f"= {mock_per_batch / BATCH:.2f} ms/msg "
          f"= {BATCH / (mock_per_batch / 1000):.0f} msg/s single worker\n")

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.insert(0, "cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    for dev in devices:
        r = bench(dev)
        out.append(r)
        print(
            f"{dev.upper():>5}: {r['per_batch_ms_median']:>8.1f} ms/batch  "
            f"{r['per_message_ms']:>6.2f} ms/msg  "
            f"{r['max_msgs_per_sec_single_worker']:>8.1f} msg/s  "
            f"(overstatement vs mock: "
            f"{r['per_batch_ms_median'] / mock_per_batch:.1f}x)"
        )

    print("\n" + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
