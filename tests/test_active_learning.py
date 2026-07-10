"""
Tests for the active-learning acquisition function.

Uses a fake model with hand-controlled logits so we can assert that uncertainty
sampling really selects the highest-entropy (most uncertain) examples — no
network or real RoBERTa needed.
"""

from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from src.active_learning.loop import compute_entropy_scores, uncertainty_query


class _FakeModel(torch.nn.Module):
    """logits = [first_token, 0, 0]: a large first token ⇒ confident (low entropy),
    a first token of 0 ⇒ uniform over 3 classes (max entropy)."""

    def forward(self, input_ids, attention_mask):
        first = input_ids[:, 0].float()
        z = torch.zeros_like(first)
        return SimpleNamespace(logits=torch.stack([first, z, z], dim=1))


class _FakeDS(Dataset):
    def __init__(self, first_tokens):
        self.first_tokens = first_tokens

    def __len__(self):
        return len(self.first_tokens)

    def __getitem__(self, i):
        return {
            "input_ids": torch.tensor([self.first_tokens[i], 1, 1], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
            "true_label": torch.tensor(0),
        }


def test_entropy_scores_rank_uncertain_examples_highest():
    # tokens: big=confident(low entropy), 0=uniform(high entropy)
    ds = _FakeDS([10, 0, 5, 0])
    scores = compute_entropy_scores(_FakeModel(), ds)
    # index 1 and 3 (token 0) must be the most uncertain
    assert scores[1] > scores[0]
    assert scores[3] > scores[2]


def test_uncertainty_query_selects_most_uncertain():
    ds = _FakeDS([10, 0, 5, 0])
    picked = uncertainty_query(_FakeModel(), [0, 1, 2, 3], ds, query_size=2)
    assert set(picked) == {1, 3}, "should query the two highest-entropy examples"
