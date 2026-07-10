"""
Multi-task RoBERTa with three output heads.

Architecture:
    RoBERTa backbone (shared)
        ├── Head 1: Sentiment   (3-class)
        ├── Head 2: Emotion     (6-class)
        └── Head 3: Toxicity    (binary)

Three training strategies (ablation):
    1. independent        — three separate models, each fine-tuned on one task
    2. hard_sharing       — one backbone, fixed equal loss weights (1/3 each)
    3. uncertainty_weighted — one backbone, loss weights learned as parameters
                              via Kendall et al. (2018) homoscedastic uncertainty

Reference:
    Kendall, A., Gal, Y., & Cipolla, R. (2018).
    "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics."
    CVPR 2018. https://arxiv.org/abs/1705.07115
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from transformers import RobertaModel, RobertaPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

# Label value marking "this task has no annotation for this example".
# Each training example carries a real label for exactly ONE task (the datasets
# are disjoint); the other two tasks are set to this sentinel so their heads are
# not trained toward a fake target. Must match dataset.IGNORE_LABEL.
IGNORE_LABEL = -100


# ── Single-task model (for "independent" ablation baseline) ───────────────────


class SingleTaskRoBERTa(RobertaPreTrainedModel):
    """
    Standard RoBERTa fine-tuned for a single classification task.
    Used as the baseline in the ablation study.
    """

    def __init__(self, config, num_labels: int, dropout: float = 0.1):
        super().__init__(config)
        self.num_labels = num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(dropout)
        # Identical 2-layer MLP head to each MultiTaskRoBERTa task head, so the
        # ablation isolates backbone SHARING and loss weighting — not head capacity.
        hidden_size = config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_labels),
        )
        self.post_init()

    @classmethod
    def build_pretrained(
        cls, num_labels: int, model_name: str = "roberta-base", dropout: float = 0.1
    ) -> "SingleTaskRoBERTa":
        """
        Construct with a PRETRAINED backbone (random classifier head).

        Constructing `SingleTaskRoBERTa(config, ...)` directly leaves the backbone
        randomly initialised. Using HF's `from_pretrained` doesn't work here because
        `num_labels` is a recognised config kwarg and would not reach __init__, so we
        build the shell from config and swap in the pretrained encoder explicitly —
        exactly what MultiTaskRoBERTa does for its shared backbone.
        """
        from transformers import AutoConfig

        model = cls(AutoConfig.from_pretrained(model_name), num_labels, dropout)
        model.roberta = RobertaModel.from_pretrained(
            model_name, add_pooling_layer=False
        )
        return model

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.roberta(input_ids, attention_mask=attention_mask)
        # Mean pooling over non-padding tokens (more robust than [CLS] for RoBERTa)
        token_embeddings = outputs.last_hidden_state  # (B, L, H)
        mask = attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (token_embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


# ── Multi-task model ───────────────────────────────────────────────────────────


class MultiTaskRoBERTa(nn.Module):
    """
    Multi-task RoBERTa with three classification heads sharing a single backbone.

    Supports two loss-weighting strategies:
        hard_sharing:          total_loss = (1/3)*L_sentiment + (1/3)*L_emotion + (1/3)*L_toxicity
        uncertainty_weighted:  total_loss = Σ_i [ (1/2σ_i²)*L_i + log(σ_i) ]
                               where σ_i are learnable parameters (one per task).

    The uncertainty weighting formulation from Kendall et al. 2018:
        For a classification task: L_total = (1/σ²)*L_ce + log(σ)
        This is derived from the log-likelihood of a Boltzmann (Gibbs) distribution
        with temperature σ². At optimum, σ_i captures the task's inherent difficulty.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_sentiment: int = 3,
        num_emotion: int = 6,
        num_toxicity: int = 2,
        dropout: float = 0.1,
        uncertainty_weighting: bool = True,
    ):
        super().__init__()

        self.uncertainty_weighting = uncertainty_weighting

        # Shared backbone
        self.backbone = RobertaModel.from_pretrained(
            model_name, add_pooling_layer=False
        )
        hidden_size = self.backbone.config.hidden_size

        self.dropout = nn.Dropout(dropout)

        # Task-specific classification heads
        self.sentiment_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_sentiment),
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_emotion),
        )
        self.toxicity_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_toxicity),
        )

        # Learnable log(σ) per task for uncertainty weighting
        # Initialised to 0 → σ=1 → equal weights at start of training
        # Using log(σ) parameterisation ensures σ > 0 without constraints
        if uncertainty_weighting:
            self.log_sigma_sentiment = nn.Parameter(torch.zeros(1))
            self.log_sigma_emotion = nn.Parameter(torch.zeros(1))
            self.log_sigma_toxicity = nn.Parameter(torch.zeros(1))

    def _pool(
        self,
        last_hidden_state: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Mean pool over non-padding tokens."""
        mask = attention_mask.unsqueeze(-1).float()
        return (last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentiment_labels: Optional[torch.Tensor] = None,
        emotion_labels: Optional[torch.Tensor] = None,
        toxicity_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass.

        When labels are provided, computes the multi-task loss.
        When labels are None (inference mode), returns only logits.

        Returns a dict with keys:
            logits_sentiment, logits_emotion, logits_toxicity,
            loss (if labels provided),
            loss_sentiment, loss_emotion, loss_toxicity (if labels provided),
            log_sigma_* (if uncertainty_weighting)
        """
        # Backbone
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self._pool(outputs.last_hidden_state, attention_mask)
        pooled = self.dropout(pooled)

        # Heads
        logits_sentiment = self.sentiment_head(pooled)
        logits_emotion = self.emotion_head(pooled)
        logits_toxicity = self.toxicity_head(pooled)

        result = {
            "logits_sentiment": logits_sentiment,
            "logits_emotion": logits_emotion,
            "logits_toxicity": logits_toxicity,
        }

        if (
            sentiment_labels is not None
            and emotion_labels is not None
            and toxicity_labels is not None
        ):
            # ignore_index drops IGNORE_LABEL examples from each head's loss, so a
            # head is only trained on examples that actually carry its task's label.
            # (Do NOT clamp the labels to 0 before this — that would train every
            # head on every example toward a fake label-0 target.)
            ce = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

            def _task_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
                # If a batch happens to contain no valid example for a task, CE over
                # all-ignored labels is NaN; return a graph-connected zero instead.
                if (labels != IGNORE_LABEL).any():
                    return ce(logits, labels)
                return logits.sum() * 0.0

            L_s = _task_loss(logits_sentiment, sentiment_labels)
            L_e = _task_loss(logits_emotion, emotion_labels)
            L_t = _task_loss(logits_toxicity, toxicity_labels)

            result["loss_sentiment"] = L_s
            result["loss_emotion"] = L_e
            result["loss_toxicity"] = L_t

            if self.uncertainty_weighting:
                # Kendall et al. 2018: L_total = Σ_i [ (1/2σ_i²)*L_i + log(σ_i) ]
                # Using log(σ) parametrisation: σ² = exp(2*log_σ)
                # ⟹ (1/2σ²)*L + log(σ) = exp(-2*log_σ)*L/2 + log_σ
                total = (
                    torch.exp(-2 * self.log_sigma_sentiment) * L_s / 2
                    + self.log_sigma_sentiment
                    + torch.exp(-2 * self.log_sigma_emotion) * L_e / 2
                    + self.log_sigma_emotion
                    + torch.exp(-2 * self.log_sigma_toxicity) * L_t / 2
                    + self.log_sigma_toxicity
                ).squeeze()
                result["log_sigma_sentiment"] = self.log_sigma_sentiment.item()
                result["log_sigma_emotion"] = self.log_sigma_emotion.item()
                result["log_sigma_toxicity"] = self.log_sigma_toxicity.item()
            else:
                # Hard sharing: equal loss weights
                total = (L_s + L_e + L_t) / 3.0

            result["loss"] = total

        return result

    @torch.no_grad()
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        """
        Inference mode: returns predicted labels + softmax probabilities + entropy.

        Entropy is used as the acquisition function for active learning:
            H(x) = -Σ_c p_c * log(p_c)
        Higher entropy = the model is more uncertain about this sample.
        """
        self.eval()
        out = self.forward(input_ids, attention_mask)

        def _decode(
            logits: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            probs = torch.softmax(logits, dim=-1)
            pred = probs.argmax(dim=-1)
            # Predictive entropy
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1)
            return pred, probs, entropy

        s_pred, s_probs, s_entropy = _decode(out["logits_sentiment"])
        e_pred, e_probs, e_entropy = _decode(out["logits_emotion"])
        t_pred, t_probs, t_entropy = _decode(out["logits_toxicity"])

        # Aggregate uncertainty: mean entropy across all tasks
        aggregate_entropy = (s_entropy + e_entropy + t_entropy) / 3.0

        return {
            "sentiment_pred": s_pred,
            "sentiment_probs": s_probs,
            "sentiment_entropy": s_entropy,
            "emotion_pred": e_pred,
            "emotion_probs": e_probs,
            "emotion_entropy": e_entropy,
            "toxicity_pred": t_pred,
            "toxicity_probs": t_probs,
            "toxicity_entropy": t_entropy,
            "aggregate_entropy": aggregate_entropy,
        }


# ── Model factory ─────────────────────────────────────────────────────────────


def build_model(strategy: str, model_name: str = "roberta-base") -> nn.Module:
    """
    Build the appropriate model for a given training strategy.

    Args:
        strategy: "independent" | "hard_sharing" | "uncertainty_weighted"
        model_name: HuggingFace model ID for the backbone.

    Returns:
        For "independent": dict of {"sentiment": model, "emotion": model, "toxicity": model}
        For others: MultiTaskRoBERTa instance
    """
    from configs.config import (
        NUM_SENTIMENT_CLASSES,
        NUM_EMOTION_CLASSES,
        NUM_TOXICITY_CLASSES,
    )

    if strategy == "independent":
        # Pretrained backbones — a from-scratch baseline would make the multi-task
        # model win for the wrong reason and invalidate the ablation.
        return {
            "sentiment": SingleTaskRoBERTa.build_pretrained(NUM_SENTIMENT_CLASSES, model_name),
            "emotion": SingleTaskRoBERTa.build_pretrained(NUM_EMOTION_CLASSES, model_name),
            "toxicity": SingleTaskRoBERTa.build_pretrained(NUM_TOXICITY_CLASSES, model_name),
        }
    elif strategy == "hard_sharing":
        return MultiTaskRoBERTa(
            model_name=model_name,
            num_sentiment=NUM_SENTIMENT_CLASSES,
            num_emotion=NUM_EMOTION_CLASSES,
            num_toxicity=NUM_TOXICITY_CLASSES,
            uncertainty_weighting=False,
        )
    elif strategy == "uncertainty_weighted":
        return MultiTaskRoBERTa(
            model_name=model_name,
            num_sentiment=NUM_SENTIMENT_CLASSES,
            num_emotion=NUM_EMOTION_CLASSES,
            num_toxicity=NUM_TOXICITY_CLASSES,
            uncertainty_weighting=True,
        )
    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")


def count_parameters(model) -> dict:
    """Return total and trainable parameter counts."""
    if isinstance(model, dict):
        total = sum(sum(p.numel() for p in m.parameters()) for m in model.values())
        trainable = sum(
            sum(p.numel() for p in m.parameters() if p.requires_grad)
            for m in model.values()
        )
    else:
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total,
        "trainable_params": trainable,
        "total_M": round(total / 1e6, 2),
    }
