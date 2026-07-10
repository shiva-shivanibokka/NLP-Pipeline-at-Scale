"""
Regression test for the multi-task label-masking contract.

Each training example carries a real label for exactly one task; the other two
tasks are set to IGNORE_LABEL (-100). A task's head must ONLY be trained on
examples that actually carry its label. Earlier code clamped labels to 0 before
the loss, which silently trained every head on fake label-0 targets and
corrupted the whole ablation study. These tests fail if that regression returns.
"""

import pytest
import torch
from transformers import RobertaConfig, RobertaModel

import src.model.multitask_model as mm
from src.model.multitask_model import MultiTaskRoBERTa, IGNORE_LABEL


@pytest.fixture
def tiny_model(monkeypatch):
    """MultiTaskRoBERTa with a tiny random backbone (no network download)."""
    cfg = RobertaConfig(
        vocab_size=100,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    monkeypatch.setattr(
        mm.RobertaModel, "from_pretrained", lambda *a, **k: RobertaModel(cfg)
    )
    torch.manual_seed(0)
    return MultiTaskRoBERTa(uncertainty_weighting=False)


def _batch(bsz=4, seqlen=8):
    return {
        "input_ids": torch.randint(0, 100, (bsz, seqlen)),
        "attention_mask": torch.ones(bsz, seqlen, dtype=torch.long),
    }


def test_masked_task_gets_no_gradient(tiny_model):
    """Sentiment fully masked (-100) → sentiment head must receive zero gradient,
    while the emotion head (real labels) must receive nonzero gradient."""
    b = _batch()
    out = tiny_model(
        input_ids=b["input_ids"],
        attention_mask=b["attention_mask"],
        sentiment_labels=torch.full((4,), IGNORE_LABEL),  # all masked
        emotion_labels=torch.tensor([0, 1, 2, 3]),  # real
        toxicity_labels=torch.full((4,), IGNORE_LABEL),  # all masked
    )
    out["loss"].backward()

    s_grad = tiny_model.sentiment_head[0].weight.grad
    e_grad = tiny_model.emotion_head[0].weight.grad

    assert s_grad is None or torch.count_nonzero(s_grad) == 0, (
        "masked sentiment task leaked gradient into its head"
    )
    assert e_grad is not None and torch.count_nonzero(e_grad) > 0, (
        "unmasked emotion task produced no gradient"
    )


def test_all_masked_task_is_zero_loss(tiny_model):
    """A task with no valid example in the batch contributes exactly zero loss
    (not NaN from averaging over zero elements)."""
    b = _batch()
    out = tiny_model(
        input_ids=b["input_ids"],
        attention_mask=b["attention_mask"],
        sentiment_labels=torch.tensor([0, 1, 2, 0]),
        emotion_labels=torch.full((4,), IGNORE_LABEL),  # no valid → zero, not NaN
        toxicity_labels=torch.tensor([0, 1, 0, 1]),
    )
    assert out["loss_emotion"].item() == 0.0
    assert torch.isfinite(out["loss"]).all()
