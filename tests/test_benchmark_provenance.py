"""The benchmark must never emit a throughput number without saying what produced it.

`MockInferencePipeline.process_batch` is `time.sleep(25ms * n/32)`. The committed
`results/benchmark/throughput_benchmark.json` was generated with it and carried no
marker, so "saturation ~1500 msg/s, inference p99 ~1.0 ms" read as a measurement
of the pipeline. It was a measurement of `time.sleep`: 25/32 = 0.78 ms per
message, which is exactly the published p50.

Measured real cost for the same batch shape (see scripts/measure_real_inference.py):
104.0 ms/batch of 32 on an RTX 4060 Laptop GPU and 2517.9 ms on CPU -- 4.2x and
101x the mock, and both are lower bounds because tokenization is excluded.

These tests make the provenance structural rather than documentary.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.benchmark.throughput import BenchmarkReport, MockInferencePipeline

RESULTS = Path(__file__).resolve().parents[1] / "results" / "benchmark" / "throughput_benchmark.json"


def test_a_mock_backed_report_is_stamped_as_simulated():
    report = BenchmarkReport(inference_backend=MockInferencePipeline.BACKEND_ID)
    assert report.is_simulated is True
    payload = report.to_dict()
    assert payload["simulated"] is True
    assert payload["inference_backend"] == "mock-time-sleep"
    assert payload["warning"], "a simulated report must carry a warning string"


def test_the_warning_names_the_mechanism_and_the_real_cost():
    """A vague caveat is not enough -- the reader needs the mechanism and the
    magnitude, or they cannot tell how wrong the number is."""
    warning = BenchmarkReport(
        inference_backend=MockInferencePipeline.BACKEND_ID
    ).provenance_warning
    assert warning is not None
    low = warning.lower()
    assert "time.sleep" in low, "must name the mechanism"
    assert "not" in low and "model" in low, "must say no model ran"
    for figure in ("104.0", "2517.9", "308", "12.7"):
        assert figure in warning, f"the real measured cost {figure} must appear"


def test_a_real_backend_is_not_stamped_as_simulated():
    class RealPipeline:
        BACKEND_ID = "multitask-roberta-cuda"

    report = BenchmarkReport(
        inference_backend=RealPipeline.BACKEND_ID, real_kafka=True
    )
    assert report.is_simulated is False
    payload = report.to_dict()
    assert payload["simulated"] is False
    assert payload["warning"] is None
    assert payload["real_kafka"] is True


def test_provenance_is_the_first_thing_in_the_serialised_report():
    """Key order matters for a file people skim. `inference_backend` must not be
    buried under the numbers it qualifies."""
    payload = BenchmarkReport(
        inference_backend=MockInferencePipeline.BACKEND_ID
    ).to_dict()
    assert list(payload)[0] == "inference_backend"
    keys = list(payload)
    assert keys.index("warning") < keys.index("saturation_point_msgs_per_sec")


def test_the_mock_declares_its_backend_id():
    """run_benchmark reads BACKEND_ID off the pipeline via getattr; if the
    attribute disappears the stamp silently degrades to a class name."""
    assert MockInferencePipeline.BACKEND_ID == "mock-time-sleep"
    assert getattr(MockInferencePipeline(), "BACKEND_ID", None) == "mock-time-sleep"


def test_the_mock_docstring_warns_it_is_not_a_model():
    doc = (MockInferencePipeline.__doc__ or "").lower()
    assert "does not run a model" in doc or "not run a model" in doc
    assert "time.sleep" in doc


@pytest.mark.skipif(not RESULTS.exists(), reason="benchmark results not generated")
def test_the_committed_results_file_declares_its_provenance():
    """The file published in the repo is the one a reader will find first."""
    payload = json.loads(RESULTS.read_text(encoding="utf-8"))
    assert "inference_backend" in payload, (
        "the committed benchmark JSON predates provenance stamping; regenerate "
        "it with python scripts/run_benchmark.py"
    )
    if payload.get("simulated"):
        assert payload.get("warning"), "simulated results must carry the warning"
        assert "time.sleep" in payload["warning"]
