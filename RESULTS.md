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

**A related presentation defect, stated correctly.** An earlier draft of this
section accused the published 1500 msg/s of being "not internally consistent"
because it exceeds the mock's 1280 msg/s single-worker ceiling. **That accusation
was wrong**, and an adversarial review caught it.
`saturation_point_msgs_per_sec` is not an achieved throughput: `run_benchmark()`
sets it to the `target_msgs_per_sec` of the *first level at which the consumer
fell behind*. With levels `[100, 250, 500, 1000, 1500, 2000]`, the first failing
level is necessarily above the ceiling -- reporting 1500 is what a *correct*
saturation sweep does. The achieved rate in the committed data confirms it:
`actual_consumer_msgs_per_sec` at the 1500 level is **1234.3** (1250.1 in the
original file), both below 1280.

The real defect is the **field name**: `saturation_point_msgs_per_sec` reads like
an achievable rate, and the README quoted it as one. The number a reader wants is
`actual_consumer_msgs_per_sec` (~1250 under the mock). Accusing the old numbers
of inconsistency, in a document whose thesis is that unsupported claims are the
defect, was precisely the error being documented.

### Real measurement

Same batch shape through the actual `MultiTaskRoBERTa` (roberta-base backbone,
batch 32, seq len 128, median of 15 iterations after 3 warmups):

| backend | ms / batch of 32 | msg/s, one worker | vs mock |
|---|---|---|---|
| `MockInferencePipeline` | 25.0 | 1280 | — |
| **RTX 4060 Laptop GPU** | **104 – 115** | **277 – 308** | **4.2 – 4.6×** |
| **CPU** | **2518 – 3473** | **9.2 – 12.7** | **101 – 139×** |

**Ranges, not point estimates, and the reason matters.** An earlier draft reported
`104.0` and `2517.9` and called them "stable to about 1.5%", on the strength of
two runs taken minutes apart. An independent re-run in a later session came back
**9% higher**. This is a laptop GPU with unlocked clocks and dynamic boost, so a
same-session repeat measures *repeatability* and says nothing about
*reproducibility* — presenting it as the latter was the same error this document
exists to catalogue, committed while cataloguing it.

Worse, those literals were hardcoded into library code **and pinned by a test**,
so the suite enforced a number that did not reproduce, and an honest
re-measurement would have failed CI. The figures now live in
`results/benchmark/real_inference.json`, the warning string carries the range, and
the test asserts a *range and a ratio* rather than a value.

Observed across 7 runs (5 mine, 2 by an independent reviewer on this machine):
GPU 104.0 / 105.6 / 108.6 / 109.3 / 112.9 / 113.7 / 113.7 ms; CPU 2517.9 / 2524.0
/ 2578.9 / 2761.7 / 2766.9 / 2788.9 / 2829.6 ms. **The ratio is the durable
finding; the absolute millisecond figure is not.**

**These are lower bounds.** Tokenization is excluded (the batch is encoded once
outside the timing loop), and the heads are randomly initialised — which does not
affect timing, since cost is set by architecture rather than weight values, but
is stated so the measurement is not over-read.

`1500 / 300 ≈ 5` is arithmetic, not a capacity estimate, so this file does not
claim "five workers would reach 1500 msg/s". At batch 32 / seq 128 one 4060 is
already compute-bound; five concurrent workers on the *same* GPU would contend
for the same SMs and scale sublinearly, and five separate GPUs is a different
proposition entirely. What the measurement supports is a per-worker ceiling, and
nothing about how it aggregates.

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
| Googleplex | `brand:google` | `"google"` inside |
| Metallica | `brand:meta` | `"meta"` inside |
| AWSome | `brand:amazon` | `"aws"` inside |
| Applebee's | `brand:apple` | `"apple"` inside |

Which wrong answer you got depended on **dict iteration order**. A dashboard
aggregating sentiment by `canonical_id` silently attributed every FBI mention to
Meta — invisible in aggregate, indefensible when anyone checks a single row.

**Fix.** The `key in brand_key` direction is gone entirely: a surface form being a
*fragment* of a brand name is not evidence it refers to that brand. The remaining
direction requires word boundaries (`(?<!\w)…(?!\w)`), and candidate keys are
tried longest-first. "apple inc" still resolves to Apple; "Applebee's" does not.

Ordering dependence is **reduced, not eliminated**: `sorted` is stable, so among
equal-length keys (there are six of 4 characters) the dict's insertion order
still decides. Low practical impact — spans are usually a single entity — but
"no longer depends on dict ordering", as an earlier draft put it, was an absolute
claim that is not true.

**Three genuine recall losses**, found by the same review and fixed by listing
them rather than by restoring the fallback: `GOOG` (Alphabet's class-C ticker),
`Insta` and `Elon` all resolved under the old substring rule and stopped
resolving under word-boundary matching. They are now explicit table entries, so
precision and recall both hold: `GOOG`/`Insta`/`Elon` resolve, while
`FBI`/`Al`/`Applebee's`/`Googleplex` do not.

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

Uncertainty sampling is **behind at 7 of the 11 checkpoints**, ahead at 2, and
tied at 2 (the first two, before either has queried anything). The final +0.0072
is the last point of a noisy single-seed curve.

(An earlier draft said "behind at 8 of 11 ... and ahead at 2 (the first two
tie)", which totals 12 of 11 and is impossible on its face. The correct count is
7 / 2 / 2. Flagged by adversarial review.)

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
the trade: emotion **improves** under sharing while toxicity pays the largest
cost.

**A tidy explanation for that shape was proposed, and it is false.** An earlier
draft said "sharing helps the task with the most data and hurts the one with the
least" -- and explicitly certified it as true. The training splits say otherwise:

| task | train rows | dF1 (shipped config) |
|---|---|---|
| sentiment (`tweet_eval/sentiment`) | **45,615** -- most | **-0.93** |
| emotion (`dair-ai/emotion`) | 16,000 -- middle | **+0.92** |
| toxicity (`tweet_eval/hate`) | **9,000** -- least | **-1.78** |

The task with the **most** data is hurt; the task that improves is the **middle**
one. The accurate statement is that sharing hurts both the largest and the
smallest task and helps the middle one -- which has no tidy data-size story. The
sentence was a post-hoc narrative fitted to three numbers, and it was the single
most rhetorically load-bearing claim in this section. Caught by adversarial
review; see §9.

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

## 7b. The CI job that could not run the test added in section 6

Found on 2026-10-08, when the pull request opened and CI ran this branch for the
first time on a clean machine.

`tests/test_api_honesty.py` -- the test added in section 6, which pins the two
states the API must disclose -- imports `api.main`. That module's top-level
imports are `torch`, `dotenv`, `fastapi` and `pydantic`; everything else in the
file is imported inside a function. `requirements-dev.txt`, which is the only
thing the CI job installs, declared `torch` and not the other three.

The result was not a failing test. It was a **collection error**, so pytest
aborted before running anything:

```
ImportError while importing test module '.../tests/test_api_honesty.py'
E   ModuleNotFoundError: No module named 'dotenv'
!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!
1 error in 5.85s
```

Zero of the 46 tests ran. Locally all 46 passed, because this machine has the
full `requirements.txt` environment where `fastapi` and `python-dotenv` are
present -- they are declared in `requirements.txt` and `requirements-api.txt`,
just not in the dev file CI uses.

Fixed by adding `fastapi>=0.110.0` and `python-dotenv>=1.0.0` to
`requirements-dev.txt`, with a comment saying why they are there so a future
trim does not remove them again. `pydantic` arrives as a `fastapi` dependency
and is not pinned separately.

Reproduced before fixing, in a venv built exactly as CI builds one (CPU torch,
then `requirements-dev.txt`):

| `requirements-dev.txt` | result |
| --- | --- |
| as committed | **collection error, 0 of 46 tests ran** |
| plus fastapi + python-dotenv | **46 passed**, `ruff check .` clean |

**Worth being exact about what was wrong.** The test itself was correct and its
assertions hold. But between the commit that added it and this one, the suite
that was supposed to enforce the section-6 contract was not merely failing to
enforce it -- it was not running at all, and the job failed in a way that looked
like a dependency problem rather than like an unchecked contract. The local run
was green throughout. This is the same shape as section 1: the number was
coming from somewhere other than the thing it claimed to measure.

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
