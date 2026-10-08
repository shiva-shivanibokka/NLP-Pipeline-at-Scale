"""The API must disclose two states a caller cannot otherwise observe.

1. **Untrained weights.** `_get_model` printed a warning to stdout and served on.
   An untrained RoBERTa still returns a well-formed probability distribution, so
   a client sees confident-looking class scores with nothing marking them as
   meaningless. Nobody reads a server's stdout.

2. **Topic assignment unavailable.** `include_topics` defaults to True, but
   `bertopic` is deliberately absent from `requirements-api.txt` (the slim
   serving image). `_get_topic_model()` raised ImportError inside the handler, so
   in the deployed image the endpoint's own default could not succeed — every
   default-shaped request returned 500.

These tests check the contract, not the model, so they do not download weights.
"""
from __future__ import annotations

import api.main as main


def test_untrained_warning_names_the_consequence_not_just_the_cause():
    w = main.UNTRAINED_WARNING.lower()
    assert "randomly-initialised" in w or "randomly initialised" in w
    assert "meaningless" in w, (
        "the warning must say the predictions are meaningless, not merely that a "
        "checkpoint is missing -- the second reads like a configuration note"
    )
    # It must also say how to fix it.
    assert "HF_MODEL_REPO" in main.UNTRAINED_WARNING
    assert "MODEL_CKPT_PATH" in main.UNTRAINED_WARNING


def test_topics_warning_explains_why_and_what_it_returns():
    w = main.TOPICS_UNAVAILABLE_WARNING
    assert "bertopic" in w.lower()
    assert "requirements-api.txt" in w, "must point at the actual reason"
    assert "-1" in w, "must say what topic_id will be"


def test_response_model_exposes_model_trained_and_warnings():
    fields = main.AnalyzeResponse.model_fields
    assert "model_trained" in fields, (
        "a caller cannot distinguish trained from untrained output without this"
    )
    assert "warnings" in fields
    assert fields["model_trained"].annotation is bool


def test_warnings_defaults_to_empty_so_a_healthy_response_is_quiet():
    """A warnings field that is always populated gets ignored."""
    fields = main.AnalyzeResponse.model_fields
    default = fields["warnings"].get_default(call_default_factory=True)
    assert default == []


def test_get_topic_model_returns_none_instead_of_raising(monkeypatch):
    """Simulate the slim image: bertopic not importable."""
    monkeypatch.setattr(main, "_topic_model", None)

    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("src.topics") or name == "bertopic":
            raise ImportError("No module named 'bertopic'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    assert main._get_topic_model() is None, (
        "must degrade to None so /analyze can return topic_id=-1 with a warning "
        "rather than a 500"
    )


def test_topics_available_reports_a_bool_without_importing_bertopic():
    assert isinstance(main._topics_available(), bool)


def test_health_declares_the_model_state_not_just_ok():
    """`{"status": "ok"}` alone is what let an untrained deployment look fine."""
    import inspect

    src = inspect.getsource(main.health)
    for key in ("model_loaded", "model_trained", "model_checkpoint", "topics_available"):
        assert key in src, f"/health must report {key}"
