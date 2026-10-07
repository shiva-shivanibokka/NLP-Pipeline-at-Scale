# RESULTS: which numbers in this repo were measurements, and which were not

Branch `sop-eval`, based on `main` at `c08c10c`. Measured on this machine on
**2026-10-07**. Hardware: Windows 11 laptop, RTX 4060 Laptop GPU (8 GB).

**Headline.** The ablation and active-learning results are real experiments on
real data. The **throughput benchmark is not** — it measured `time.sleep`. Three
further claims (a hardcoded "live" badge, a toxicity gap, an "on par" active
learning verdict) overstated what the runs support, and a brand-normalisation
bug mapped **"FBI" to Meta**.

| # | Claim or defect | Status |
|---|---|---|
| 1 | Throughput "~1500 msg/s, p99 ~1.0 ms" came from `time.sleep` | **Restated with real measurements** (§1) |
| 2 | NER mapped "FBI" → `brand:meta`, "Al" → `brand:google` | **Fixed** (§2) |
| 3 | Page hardcoded "live · served from Google Cloud Run" | **Fixed** (§3) |
| 4 | Active learning "statistically on par with random" | **Withdrawn** — no variance estimate exists (§4) |
| 5 | Ablation "within ~1 F1 point on every task" | **Corrected to 1.8** (§5) |
| 6 | Untrained weights invisible to API callers | **Fixed** (§6) |
| 7 | `include_topics=True` 500s in the deployed image | **Fixed** (§6) |
| 8 | Test suite imported `transformers` without the backend shim | **Fixed** (§7) |

Tests: **7 → 38 passed.** The frontend typechecks and builds clean.

---

## 1. The throughput benchmark measured `time.sleep`, not the model

`src/benchmark/throughput.py` runs against `MockInferencePipeline`, whose entire
`process_batch` is:

```python
base_ms = self.inference_ms_per_batch * (n / 32)   # 25.0 ms default
time.sleep(max(0, (base_ms + jitter) / 1000))
```

The committed `results/benchmark/throughput_benchmark.json` was generated with
it and carried **no marker saying so**, so the README's "saturation point ≈ 1500
msg/s … per-message inference p99 ≈ 1.0 ms" read as a property of the pipeline.

The arithmetic gives it away: 25 ms ÷ 32 = **0.78 ms/msg**, and the published p50
was 0.79–0.83 ms. The number was the sleep divided by the batch size.

**It was not even internally consistent.** The claimed saturation point of 1500
msg/s exceeds the mock's own single-worker ceiling of 1280 msg/s.

### Real measurement

Same batch shape through the actual `MultiTaskRoBERTa` (roberta-base backbone,
batch 32, seq len 128, median of 15 iterations after 3 warmups):

| backend | ms / batch of 32 | ms / msg | msg/s, one worker | vs mock |
|---|---|---|---|---|
| `MockInferencePipeline` | 25.0 | 0.78 | 1280 | — |
| **RTX 4060 Laptop GPU** | **104.0** | **3.25** | **308** | **4.2× optimistic** |
| **CPU** | **2517.9** | **78.69** | **12.7** | **101× optimistic** |

Reproduced on a second run at 105.6 ms and 2524.0 ms, so the figures are stable
to about 1.5%.

**These are lower bounds.** Tokenization is excluded (the batch is encoded once
outside the timing loop), and the heads are randomly initialised — which does not
affect timing, since cost is set by architecture rather than weight values, but
is stated so the measurement is not over-read. Reaching 1500 msg/s with real
inference would need roughly five concurrent GPU workers.

Reproduce: `python scripts/measure_real_inference.py`.

### What was changed

Provenance is now structural rather than documentary. Every report carries
`inference_backend` (`"mock-time-sleep"` for the mock), a boolean `simulated`,
and a `warning` string naming the mechanism *and* the real cost — and those three
keys are serialised **first**, ahead of the numbers they qualify. `print_table`
prints `[SIMULATED — NOT A MODEL]` in its title. The committed JSON has been
regenerated so the file in the repo carries its own caveat.

`tests/test_benchmark_provenance.py` (7 tests) enforces all of it, including that
the warning contains the actual measured figures rather than a vague hedge.

### What the benchmark does legitimately show

Batch assembly and back-pressure behave correctly: the consumer tracks the
producer 1:1 until the configured inference budget is exhausted, then lag grows
monotonically rather than thrashing. That is a statement about the streaming
code, and it is worth keeping — stated as such.

---

## 2. Brand normalisation mapped "FBI" to Meta

`_normalize_entity` fell back to a bare substring test in **both** directions:

```python
for brand_key, brand_id in BRAND_NORMALIZATION.items():
    if brand_key in key or key in brand_key:
        return brand_id
```

The table has 34 entries, three of them very short: `fb` → Meta, `aws` → Amazon,
`gpt` → OpenAI. So:

| surface form | mapped to | why |
|---|---|---|
| **FBI** | `brand:meta` | `"fb"` is inside `"fbi"` |
| **Al** | `brand:google` | `"al"` is inside `"alphabet"` |
| Egypt | `brand:openai` | `"gpt"` inside `"egypt"` |
| AWSome | `brand:amazon` | `"aws"` inside |
| Applebee's | `brand:apple` | `"apple"` inside |

Which wrong answer you got depended on **dict iteration order**. A dashboard
aggregating sentiment by `canonical_id` silently attributed every FBI mention to
Meta — invisible in aggregate, indefensible when anyone checks a single row.

**Fix.** The `key in brand_key` direction is gone entirely: a surface form being a
*fragment* of a brand name is not evidence it refers to that brand. The remaining
direction requires word boundaries (`(?<!\w)…(?!\w)`), and candidate keys are
tried longest-first so the result no longer depends on dict ordering. "apple inc"
still resolves to Apple; "Applebee's" does not.

`tests/test_ner_normalization.py` (17 tests) pins the false positives and
verifies every existing table entry still resolves, including with `$`/`#`/`@`
prefixes. 9 go red against the old code.

---

## 3. The page asserted its own backend was live

`frontend/app/page.tsx` hardcoded:

```tsx
<span className="live"><span className="dot" /> live · served from Google Cloud Run</span>
```

Nothing checked. The Cloud Run service no longer exists (the Google billing
account behind it is closed), so the page asserted a liveness it had never
verified and could not have verified, with a pulsing green dot.

**Fix.** A real probe at `/api/health` (10 s timeout, never throws) and a
three-state badge: `checking…` while the probe is in flight — the page claims
nothing until it knows — then either the live badge or **"model backend offline —
the demo below will not return results"** in red, with a non-pulsing dot. The
cold-start note underneath, which promised "the first request after idle wakes
the model (~20s)", is replaced in the offline state by text saying Analyze will
error and pointing the reader at the measured results instead.

Verified by loading the built page with no `MODEL_API_URL`: the badge renders the
offline state and the note reads *"The model backend is not reachable
(MODEL_API_URL not configured), so Analyze will return an error."*

---

## 4. "Statistically on par with random" is not available from this data

The README described entropy-based active learning as **"statistically on par
with random"**. There is **one run per arm, one seed, no replicates**, so no
variance estimate exists and no statistical claim can be made in either
direction.

The underlying curves also do not support the implied tie:

| labels | uncertainty | random | Δ |
|---|---|---|---|
| 300 | 0.2784 | 0.3364 | −0.0580 |
| 400 | 0.5833 | 0.5919 | −0.0086 |
| 500 | 0.6070 | 0.6161 | −0.0091 |
| 550 | 0.6049 | 0.6338 | −0.0289 |
| 650 | 0.6323 | 0.6105 | **+0.0218** |
| 700 | 0.6309 | 0.6237 | **+0.0072** |

Uncertainty sampling is **behind at 8 of 11 checkpoints** and ahead at 2 (the
first two tie, before either has queried anything). The final +0.0072 is the last
point of a noisy single-seed curve.

**Changed to:** the curves are close and uncertainty sampling does not lead;
"no benefit at this scale" is a reasonable reading, a tested equivalence is not.
Seed replication is named as the work that would license anything stronger.

---

## 5. The ablation's "within ~1 F1 point" understated the only real regression

Exact macro-F1 deltas against the independent baseline:

| | Sentiment | Emotion | Toxicity |
|---|---|---|---|
| Hard sharing | −0.92 | **+0.48** | −1.28 |
| **Uncertainty weighted** (shipped) | −0.93 | **+0.92** | **−1.78** |

The claim was "within ~1 F1 point on every task". The actual worst case is
**1.78 points**, on toxicity, in the configuration that ships.

The more interesting correction is that "accuracy is a wash" hid the *shape* of
the trade: emotion **improves** under sharing while toxicity pays the entire
cost. Sharing helps the task with the most data and hurts the one with the least.
That is a better story than a wash, and it is true.

The efficiency numbers were understated if anything and are now exact: **3.0×
fewer parameters** (373 M → 125 M) and **2.2× lower p99 latency** (27.0 → 12.4
ms). Also noted: single run per strategy, so none of these deltas carries an
uncertainty estimate, and the 1.78-point gap sits on a test set with a known
train→test distribution shift.

---

## 6. Two states the API refused to disclose

**Untrained weights.** `_get_model()` printed `WARNING: no trained weights found`
to stdout and served anyway. An untrained RoBERTa still returns a well-formed
probability distribution, so a caller saw confident-looking class scores with
nothing marking them meaningless — and nobody reads a server's stdout.
`/analyze` now returns `model_trained: bool` and a `warnings: list[str]`
(empty in the healthy case, so it does not become noise), and `/health` reports
`model_loaded`, `model_trained`, `model_checkpoint` and `topics_available`
instead of a bare `{"status": "ok"}`.

**Topic assignment in the slim image.** `AnalyzeRequest.include_topics` defaults
to `True`, but `bertopic` is deliberately absent from `requirements-api.txt` (the
Cloud Run serving image). `_get_topic_model()` raised `ImportError` inside the
handler, so **the endpoint's own documented default could not succeed where it
was deployed** — every default-shaped request returned 500. It now returns
`None`, and `/analyze` degrades to `topic_id: -1` with a warning explaining why.

`tests/test_api_honesty.py` (8 tests) covers both, without downloading weights.

---

## 7. The test suite and the scripts behaved differently

Every `scripts/` entrypoint imports `configs.quiet` before `transformers`, which
sets `USE_TF=0`. The test suite did not. On a machine with TensorFlow installed,
`transformers` imports it, and on this setup that fails with
`ValueError: numpy.dtype size changed` — an error with nothing to do with the
test being run. Added `tests/conftest.py`, which applies the same shim and puts
the repo root on `sys.path`.

---

## 8. What is NOT claimed

- **No retraining or re-running of the ablation.** The ablation and active
  learning numbers are the committed ones; only their descriptions changed.
- **The real-inference measurement used randomly-initialised heads.** Valid for
  timing, not for accuracy, and it excludes tokenization — so the GPU/CPU figures
  are lower bounds on real per-batch cost.
- **No end-to-end Kafka run.** `--real-kafka` was not exercised; the provenance
  stamping distinguishes that path but it remains untested here.
- **MLflow `run_id`s in the results files are dangling.** `mlruns/` is gitignored,
  so the recorded run ids (e.g. `6b79d0dd…`) cannot be resolved from a clean
  clone. The metrics themselves are committed; their MLflow lineage is not
  recoverable.
- **Single seed throughout.** Neither the ablation nor the active-learning study
  is replicated, which is stated at each claim rather than once in a footnote.
