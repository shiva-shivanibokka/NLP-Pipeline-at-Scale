"""
Active learning loop: entropy-based uncertainty sampling vs. random sampling.

The core insight: in production NLP, labels are expensive. Active learning
lets a model identify which unlabeled examples it is most uncertain about
and prioritise those for annotation — reaching high accuracy with far fewer labels.

This module implements:
1. Entropy-based uncertainty sampling: selects the K examples with highest
   predictive entropy H(x) = -Σ_c p_c * log(p_c) across all tasks
2. Random sampling baseline for comparison
3. Both strategies annotated from the same unlabeled pool
4. Accuracy vs. labeled-set-size curves for both strategies

Key result: the active learning curve shows the crossover point where
uncertainty sampling matches random sampling with X% fewer labels.

Reference:
    Settles, B. (2009). Active Learning Literature Survey.
    Computer Sciences Technical Report 1648, University of Wisconsin-Madison.
    https://burrsettles.com/pub/settles.activelearning.pdf
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Subset

from configs.config import (
    AL_NUM_ROUNDS,
    AL_QUERY_BATCH_SIZE,
    AL_SEED_LABELED_SIZE,
    AL_STRATEGIES,
    AL_UNLABELED_POOL_SIZE,
    BASE_MODEL_ID,
)
from src.model.dataset import load_single_task_dataset, load_unlabeled_pool
from src.model.multitask_model import SingleTaskRoBERTa


@dataclass
class RoundResult:
    """Result of one active learning round."""

    round_num: int
    labeled_size: int
    strategy: str
    val_accuracy: float
    val_f1_macro: float
    queried_indices: list[int]


@dataclass
class ALExperiment:
    """Full active learning experiment across all rounds for one strategy."""

    strategy: str
    seed_size: int
    query_batch_size: int
    rounds: list[RoundResult] = field(default_factory=list)

    def labeled_sizes(self) -> list[int]:
        return [r.labeled_size for r in self.rounds]

    def f1_scores(self) -> list[float]:
        return [r.val_f1_macro for r in self.rounds]

    def accuracies(self) -> list[float]:
        return [r.val_accuracy for r in self.rounds]


# ── Acquisition functions ─────────────────────────────────────────────────────


@torch.no_grad()
def compute_entropy_scores(
    model: SingleTaskRoBERTa,
    unlabeled_dataset,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    """
    Compute predictive entropy for all examples in the unlabeled pool.

    H(x) = -Σ_c p_c * log(p_c)

    Higher entropy = the model assigns more uniform probability across classes
    = the model is more uncertain about this example.

    Args:
        model:             Trained single-task model.
        unlabeled_dataset: Dataset returning {input_ids, attention_mask, true_label}
        device:            Inference device.
        batch_size:        Inference batch size.

    Returns:
        numpy array of shape (N,) with entropy score per example.
    """
    model.eval().to(device)
    loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=False)
    all_entropies = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attn_mask)
        probs = torch.softmax(out.logits, dim=-1)
        entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
        all_entropies.extend(entropy.cpu().numpy())

    return np.array(all_entropies)


def uncertainty_query(
    model: SingleTaskRoBERTa,
    unlabeled_pool_indices: list[int],
    unlabeled_dataset,
    query_size: int,
    device: str = "cpu",
) -> list[int]:
    """
    Select the query_size most uncertain examples from the unlabeled pool.

    Returns indices into unlabeled_pool_indices (not global dataset indices).
    """
    pool_ds = Subset(unlabeled_dataset, unlabeled_pool_indices)
    entropies = compute_entropy_scores(model, pool_ds, device)
    # Top-K most uncertain (highest entropy)
    top_k = np.argsort(entropies)[::-1][:query_size]
    return [unlabeled_pool_indices[i] for i in top_k]


def random_query(
    unlabeled_pool_indices: list[int],
    query_size: int,
    rng: random.Random,
) -> list[int]:
    """Randomly select query_size examples from the unlabeled pool."""
    return rng.sample(
        unlabeled_pool_indices, min(query_size, len(unlabeled_pool_indices))
    )


# ── Training utility ──────────────────────────────────────────────────────────


def _quick_train(
    model: SingleTaskRoBERTa,
    labeled_dataset,
    labeled_indices: list[int],
    device: str,
    num_epochs: int = 3,
    lr: float = 2e-5,
    batch_size: int = 32,
) -> SingleTaskRoBERTa:
    """Quick fine-tuning on the current labeled set."""
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    train_ds = Subset(labeled_dataset, labeled_indices)
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.1), total_steps
    )

    for _ in range(num_epochs):
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

    return model


@torch.no_grad()
def _evaluate(
    model: SingleTaskRoBERTa,
    val_dataset,
    device: str,
    batch_size: int = 64,
) -> tuple[float, float]:
    """Evaluate on validation set. Returns (accuracy, f1_macro)."""
    model.eval().to(device)
    loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    preds, trues = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())

    acc = float(sum(p == t for p, t in zip(preds, trues)) / max(1, len(trues)))
    f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    return acc, f1


# ── Main active learning loop ─────────────────────────────────────────────────


def run_active_learning(
    strategy: str = "uncertainty_entropy",
    task: str = "sentiment",
    num_rounds: int = AL_NUM_ROUNDS,
    query_batch_size: int = AL_QUERY_BATCH_SIZE,
    seed_size: int = AL_SEED_LABELED_SIZE,
    pool_size: int = AL_UNLABELED_POOL_SIZE,
    device: Optional[str] = None,
    seed: int = 42,
    output_dir: str = "results/active_learning",
    verbose: bool = True,
) -> ALExperiment:
    """
    Run one active learning experiment (one strategy, one task).

    Args:
        strategy:         "uncertainty_entropy" | "random"
        task:             "sentiment" (focus task for this experiment)
        num_rounds:       Number of annotation rounds
        query_batch_size: Examples queried per round
        seed_size:        Initial labeled set size
        pool_size:        Total unlabeled pool size
        device:           Inference device (auto-detected if None)
        seed:             Random seed
        output_dir:       Where to save results

    Returns:
        ALExperiment with accuracy/F1 per round.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    rng = random.Random(seed)
    torch.manual_seed(seed)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if verbose:
        print(
            f"\n[active_learning] Strategy={strategy}, task={task}, "
            f"rounds={num_rounds}, query_size={query_batch_size}"
        )

    # Load datasets. The labeled set is drawn from the (labeled) pool via
    # revealed indices — standard pool-based AL — so we only need the pool + val.
    val_ds = load_single_task_dataset(task, "validation")
    _, unlabeled_ds = load_unlabeled_pool(pool_size=pool_size)

    all_indices = list(range(len(unlabeled_ds)))

    # Seed set: randomly select seed_size examples to start
    labeled_indices = rng.sample(all_indices, seed_size)
    unlabeled_indices = [i for i in all_indices if i not in set(labeled_indices)]

    # Build initial model with a PRETRAINED backbone — a from-scratch RoBERTa
    # fine-tuned on a few hundred examples learns nothing, making the AL curve
    # meaningless.
    from configs.config import NUM_SENTIMENT_CLASSES

    model = SingleTaskRoBERTa.build_pretrained(NUM_SENTIMENT_CLASSES, BASE_MODEL_ID)

    experiment = ALExperiment(
        strategy=strategy,
        seed_size=seed_size,
        query_batch_size=query_batch_size,
    )

    for round_num in range(num_rounds + 1):  # round 0 = seed only
        # Train on current labeled set
        model = _quick_train(model, unlabeled_ds, labeled_indices, device)
        acc, f1 = _evaluate(model, val_ds, device)

        result = RoundResult(
            round_num=round_num,
            labeled_size=len(labeled_indices),
            strategy=strategy,
            val_accuracy=round(acc, 4),
            val_f1_macro=round(f1, 4),
            queried_indices=[],
        )
        experiment.rounds.append(result)

        if verbose:
            print(
                f"  Round {round_num:2d}: labeled={len(labeled_indices):4d}, "
                f"acc={acc:.4f}, f1={f1:.4f}"
            )

        if round_num == num_rounds:
            break

        # Query next batch
        if strategy == "uncertainty_entropy":
            queried = uncertainty_query(
                model, unlabeled_indices, unlabeled_ds, query_batch_size, device
            )
        else:  # random
            queried = random_query(unlabeled_indices, query_batch_size, rng)

        result.queried_indices = queried
        labeled_indices.extend(queried)
        queried_set = set(queried)
        unlabeled_indices = [i for i in unlabeled_indices if i not in queried_set]

    # Save results
    result_path = Path(output_dir) / f"{strategy}_{task}.json"
    result_path.write_text(
        json.dumps(
            {
                "strategy": strategy,
                "task": task,
                "labeled_sizes": experiment.labeled_sizes(),
                "f1_scores": experiment.f1_scores(),
                "accuracies": experiment.accuracies(),
            },
            indent=2,
        )
    )

    if verbose:
        total_labels = len(labeled_indices)
        final_f1 = experiment.rounds[-1].val_f1_macro
        print(f"  Final: {total_labels} labeled examples, F1={final_f1:.4f}")
        print(f"  Results saved to {result_path}")

    return experiment


def run_comparison(
    task: str = "sentiment",
    num_rounds: int = AL_NUM_ROUNDS,
    query_batch_size: int = AL_QUERY_BATCH_SIZE,
    seed_size: int = AL_SEED_LABELED_SIZE,
    output_dir: str = "results/active_learning",
    verbose: bool = True,
) -> dict[str, ALExperiment]:
    """
    Run both strategies and return comparison results.

    Returns:
        {"uncertainty_entropy": ALExperiment, "random": ALExperiment}
    """
    results = {}
    for strategy in AL_STRATEGIES:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Running strategy: {strategy}")
            print(f"{'=' * 60}")
        exp = run_active_learning(
            strategy=strategy,
            task=task,
            num_rounds=num_rounds,
            query_batch_size=query_batch_size,
            seed_size=seed_size,
            output_dir=output_dir,
            verbose=verbose,
        )
        results[strategy] = exp

    # Save comparison summary
    summary = {}
    for strategy, exp in results.items():
        summary[strategy] = {
            "labeled_sizes": exp.labeled_sizes(),
            "f1_scores": exp.f1_scores(),
        }

    summary_path = Path(output_dir) / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Print comparison table
    if verbose:
        _print_comparison(results)

    return results


def _print_comparison(results: dict[str, ALExperiment]) -> None:
    print("\n" + "=" * 55)
    print("ACTIVE LEARNING COMPARISON")
    print("=" * 55)
    unc = results.get("uncertainty_entropy")
    rnd = results.get("random")
    if unc and rnd:
        print(f"{'Labels':>8}  {'Uncertainty F1':>15}  {'Random F1':>12}  {'Gain':>8}")
        print("-" * 55)
        for r_u, r_r in zip(unc.rounds, rnd.rounds):
            gain = r_u.val_f1_macro - r_r.val_f1_macro
            print(
                f"{r_u.labeled_size:>8}  {r_u.val_f1_macro:>15.4f}  "
                f"{r_r.val_f1_macro:>12.4f}  {gain:>+8.4f}"
            )
    print("=" * 55)
