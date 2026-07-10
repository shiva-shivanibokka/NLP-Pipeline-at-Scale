"""
Multi-task dataset: aligns tweet_eval (sentiment, toxicity) and
dair-ai/emotion into a unified format where each example has labels
for all three tasks.

Strategy:
  - tweet_eval/sentiment and dair-ai/emotion are both tweet-domain text.
  - We create a unified dataset by loading all three task datasets,
    tokenising with a shared tokeniser, and yielding batches that
    contain labels for all three tasks simultaneously.
  - For examples from single-task datasets, the missing task labels
    are filled with -100 (ignored in CrossEntropyLoss).

For the "independent" ablation baseline, each task is loaded separately
via load_single_task_dataset().
"""

from __future__ import annotations

from typing import Optional

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

# NOTE: `datasets` (via pyarrow) is imported lazily inside the loader functions.
# Importing it at module top can segfault on Windows when torch/MKL is already
# loaded (OpenMP DLL double-load), and it isn't needed just to import this module.

from configs.config import (
    EMOTION_DATASET,
    MAX_SEQ_LENGTH,
    SENTIMENT_DATASET,
    SENTIMENT_SUBSET,
    TOXICITY_DATASET,
    TOXICITY_SUBSET,
    BASE_MODEL_ID,
)

IGNORE_LABEL = -100


# ── Tokenised multi-task dataset ───────────────────────────────────────────────


class MultiTaskTweetDataset(Dataset):
    """
    Unified dataset for multi-task training.

    Each example has:
        input_ids, attention_mask,
        sentiment_label  (-100 if not available for this example)
        emotion_label    (-100 if not available for this example)
        toxicity_label   (-100 if not available for this example)
    """

    def __init__(self, examples: list[dict]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": torch.tensor(ex["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(ex["attention_mask"], dtype=torch.long),
            "sentiment_label": torch.tensor(ex["sentiment_label"], dtype=torch.long),
            "emotion_label": torch.tensor(ex["emotion_label"], dtype=torch.long),
            "toxicity_label": torch.tensor(ex["toxicity_label"], dtype=torch.long),
        }


def _tokenise(texts: list[str], tokenizer, max_length: int) -> list[dict]:
    enc = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors=None,
    )
    return [
        {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
        for i in range(len(texts))
    ]


def load_multitask_dataset(
    split: str = "train",
    max_length: int = MAX_SEQ_LENGTH,
    tokenizer_name: str = BASE_MODEL_ID,
    limit_per_task: Optional[int] = None,
) -> MultiTaskTweetDataset:
    """
    Load and unify all three task datasets for the given split.

    Args:
        split: "train" | "validation" | "test"
        max_length: Maximum token sequence length.
        tokenizer_name: HuggingFace tokenizer ID.
        limit_per_task: If set, keep at most this many examples per task
            (used by the --smoke run to exercise the pipeline quickly).

    Returns:
        MultiTaskTweetDataset ready for DataLoader.
    """
    from datasets import load_dataset

    def _cap(ds):
        return ds.select(range(min(limit_per_task, len(ds)))) if limit_per_task else ds

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    examples = []

    # ── Sentiment (tweet_eval/sentiment) ──────────────────────────────────────
    # list() materialises the datasets>=5.0 lazy Column into a plain list[str/int].
    sent_ds = _cap(load_dataset(SENTIMENT_DATASET, SENTIMENT_SUBSET, split=split))
    sent_labels = list(sent_ds["label"])
    sent_toks = _tokenise(list(sent_ds["text"]), tokenizer, max_length)
    for i, tok in enumerate(sent_toks):
        examples.append(
            {
                **tok,
                "sentiment_label": int(sent_labels[i]),
                "emotion_label": IGNORE_LABEL,
                "toxicity_label": IGNORE_LABEL,
            }
        )

    # ── Emotion (dair-ai/emotion) ─────────────────────────────────────────────
    emo_ds = _cap(load_dataset(EMOTION_DATASET, split=split))
    emo_labels = list(emo_ds["label"])
    emo_toks = _tokenise(list(emo_ds["text"]), tokenizer, max_length)
    for i, tok in enumerate(emo_toks):
        examples.append(
            {
                **tok,
                "sentiment_label": IGNORE_LABEL,
                "emotion_label": int(emo_labels[i]),
                "toxicity_label": IGNORE_LABEL,
            }
        )

    # ── Toxicity (tweet_eval/hate) ────────────────────────────────────────────
    tox_ds = _cap(load_dataset(TOXICITY_DATASET, TOXICITY_SUBSET, split=split))
    tox_labels = list(tox_ds["label"])
    tox_toks = _tokenise(list(tox_ds["text"]), tokenizer, max_length)
    for i, tok in enumerate(tox_toks):
        examples.append(
            {
                **tok,
                "sentiment_label": IGNORE_LABEL,
                "emotion_label": IGNORE_LABEL,
                "toxicity_label": int(tox_labels[i]),
            }
        )

    return MultiTaskTweetDataset(examples)


def load_single_task_dataset(
    task: str,
    split: str = "train",
    max_length: int = MAX_SEQ_LENGTH,
    tokenizer_name: str = BASE_MODEL_ID,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load a single-task dataset for the independent baseline ablation.

    Args:
        task: "sentiment" | "emotion" | "toxicity"
        split: "train" | "validation" | "test"
        limit: If set, keep at most this many examples (used by --smoke).

    Returns:
        A torch Dataset with input_ids, attention_mask, labels.
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    if task == "sentiment":
        ds = load_dataset(SENTIMENT_DATASET, SENTIMENT_SUBSET, split=split)
    elif task == "emotion":
        ds = load_dataset(EMOTION_DATASET, split=split)
    elif task == "toxicity":
        ds = load_dataset(TOXICITY_DATASET, TOXICITY_SUBSET, split=split)
    else:
        raise ValueError(f"Unknown task: {task}")

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    texts = list(ds["text"])  # materialise datasets>=5.0 lazy Column
    labels = list(ds["label"])
    toks = _tokenise(texts, tokenizer, max_length)

    class _SingleTaskDS(Dataset):
        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(toks[idx]["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    toks[idx]["attention_mask"], dtype=torch.long
                ),
                "labels": torch.tensor(int(labels[idx]), dtype=torch.long),
            }

        def __len__(self):
            return len(toks)

    return _SingleTaskDS()


def load_unlabeled_pool(
    pool_size: int = 5000,
    max_length: int = MAX_SEQ_LENGTH,
    tokenizer_name: str = BASE_MODEL_ID,
    seed: int = 42,
) -> tuple[list[str], Dataset]:
    """
    Load a pool of unlabeled texts for active learning experiments.
    Uses the tweet_eval/sentiment test split as the unlabeled pool
    (labels are withheld from the model during active learning rounds).

    Returns:
        (texts, dataset) — raw texts for display + tokenised dataset for inference
    """
    from datasets import load_dataset

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    ds = load_dataset(SENTIMENT_DATASET, SENTIMENT_SUBSET, split="test")
    ds = ds.shuffle(seed=seed).select(range(min(pool_size, len(ds))))

    texts = list(ds["text"])  # materialise datasets>=5.0 lazy Column
    true_labels = list(ds["label"])  # kept for evaluation, not shown to the model
    toks = _tokenise(texts, tokenizer, max_length)

    class _UnlabeledDS(Dataset):
        def __getitem__(self, idx):
            return {
                "input_ids": torch.tensor(toks[idx]["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    toks[idx]["attention_mask"], dtype=torch.long
                ),
                "true_label": torch.tensor(int(true_labels[idx]), dtype=torch.long),
            }

        def __len__(self):
            return len(toks)

    return list(texts), _UnlabeledDS()
