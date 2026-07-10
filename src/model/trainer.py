"""
Training loop for all three ablation strategies.

Handles:
- Independent: trains three separate models, one per task
- Hard sharing: trains one multi-task model with equal loss weights
- Uncertainty weighted: trains one multi-task model with learned σ per task

All strategies log to MLflow. Results are saved as JSON for the ablation comparison table.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from configs.config import (
    TrainingConfig,
    TRAINING_STRATEGIES,
    BASE_MODEL_ID,
)
from src.model.multitask_model import build_model, count_parameters
from src.model.dataset import (
    load_multitask_dataset,
    load_single_task_dataset,
    IGNORE_LABEL,
)


def _build_optimizer(model, cfg: TrainingConfig) -> AdamW:
    """AdamW with weight decay — but NOT on the learnable log-sigma uncertainty
    parameters. Decaying log_sigma pulls sigma toward 1 (equal task weighting),
    biasing the uncertainty_weighted strategy back toward hard_sharing and
    partially defeating the very thing the ablation measures.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if "log_sigma" in name else decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return AdamW(groups, lr=cfg.lr)


# ── Multi-task training step ───────────────────────────────────────────────────


def _multitask_train_epoch(
    model,
    loader: DataLoader,
    optimizer,
    scheduler,
    device: str,
    scaler,
) -> dict:
    model.train()
    total_loss = total_s = total_e = total_t = 0.0
    steps = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        # Pass labels as-is: IGNORE_LABEL (-100) marks tasks with no annotation
        # for that example, and the model's CrossEntropyLoss(ignore_index=-100)
        # drops them. Clamping to 0 here would train every head on fake targets.
        s_labels = batch["sentiment_label"].to(device)
        e_labels = batch["emotion_label"].to(device)
        t_labels = batch["toxicity_label"].to(device)

        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda" if "cuda" in device else "cpu",
            dtype=torch.float16,
            enabled=(scaler is not None),
        ):
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment_labels=s_labels,
                emotion_labels=e_labels,
                toxicity_labels=t_labels,
            )

        loss = out["loss"]
        if scaler:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_s += out["loss_sentiment"].item()
        total_e += out["loss_emotion"].item()
        total_t += out["loss_toxicity"].item()
        steps += 1

    return {
        "train_loss": total_loss / steps,
        "train_loss_sentiment": total_s / steps,
        "train_loss_emotion": total_e / steps,
        "train_loss_toxicity": total_t / steps,
    }


@torch.no_grad()
def _multitask_eval(model, loader: DataLoader, device: str) -> dict:
    model.eval()
    all_s_pred, all_s_true = [], []
    all_e_pred, all_e_true = [], []
    all_t_pred, all_t_true = [], []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        s_labels = batch["sentiment_label"]
        e_labels = batch["emotion_label"]
        t_labels = batch["toxicity_label"]

        out = model(input_ids=input_ids, attention_mask=attention_mask)

        s_pred = out["logits_sentiment"].argmax(dim=-1).cpu()
        e_pred = out["logits_emotion"].argmax(dim=-1).cpu()
        t_pred = out["logits_toxicity"].argmax(dim=-1).cpu()

        # Only count examples that had this task's label
        s_mask = s_labels != IGNORE_LABEL
        e_mask = e_labels != IGNORE_LABEL
        t_mask = t_labels != IGNORE_LABEL

        if s_mask.any():
            all_s_pred.extend(s_pred[s_mask].numpy())
            all_s_true.extend(s_labels[s_mask].numpy())
        if e_mask.any():
            all_e_pred.extend(e_pred[e_mask].numpy())
            all_e_true.extend(e_labels[e_mask].numpy())
        if t_mask.any():
            all_t_pred.extend(t_pred[t_mask].numpy())
            all_t_true.extend(t_labels[t_mask].numpy())

    def _metrics(pred, true):
        if not pred:
            return {"accuracy": 0.0, "f1_macro": 0.0}
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
        }

    return {
        "sentiment": _metrics(all_s_pred, all_s_true),
        "emotion": _metrics(all_e_pred, all_e_true),
        "toxicity": _metrics(all_t_pred, all_t_true),
    }


# ── Single-task training (independent baseline) ────────────────────────────────


@torch.no_grad()
def _eval_single(model, loader: DataLoader, device: str) -> tuple[float, float]:
    """Evaluate a single-task model. Returns (f1_macro, accuracy)."""
    model.eval()
    preds, trues = [], []
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        trues.extend(batch["labels"].numpy())
    f1 = float(f1_score(trues, preds, average="macro", zero_division=0))
    acc = float(accuracy_score(trues, preds))
    return f1, acc


def _train_single_task(
    task: str,
    model,
    cfg: TrainingConfig,
    device: str,
) -> dict:
    """Train one single-task model for the 'independent' ablation.

    Selects the best checkpoint on the validation split, then reports metrics on
    the held-out TEST split — identical protocol to the multi-task strategies, so
    the ablation table compares like with like.
    """
    train_ds = load_single_task_dataset(task, "train", limit=cfg.max_examples_per_task)
    val_ds = load_single_task_dataset(task, "validation", limit=cfg.max_examples_per_task)
    test_ds = load_single_task_dataset(task, "test", limit=cfg.max_examples_per_task)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    model = model.to(device)
    optimizer = _build_optimizer(model, cfg)
    total_steps = len(train_loader) * cfg.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * cfg.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda") if cfg.fp16 and "cuda" in device else None

    best_val_f1 = 0.0
    best_state = None
    for epoch in range(cfg.num_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            with torch.autocast(
                device_type="cuda" if "cuda" in device else "cpu",
                dtype=torch.float16,
                enabled=(scaler is not None),
            ):
                out = model(
                    input_ids=input_ids, attention_mask=attn_mask, labels=labels
                )
            loss = out.loss
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        val_f1, val_acc = _eval_single(model, val_loader, device)
        mlflow.log_metrics({f"val_{task}_f1": val_f1, f"val_{task}_acc": val_acc}, step=epoch)
        print(
            f"[independent:{task}] epoch {epoch + 1}/{cfg.num_epochs} "
            f"val_f1={val_f1:.4f} val_acc={val_acc:.4f}",
            flush=True,
        )
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore best-on-val checkpoint and report TEST metrics.
    if best_state is not None:
        model.load_state_dict(best_state)
    test_f1, test_acc = _eval_single(model, test_loader, device)
    mlflow.log_metrics({f"test_{task}_f1": test_f1, f"test_{task}_acc": test_acc})
    print(
        f"[independent:{task}] TEST f1={test_f1:.4f} acc={test_acc:.4f} "
        f"(best val f1={best_val_f1:.4f})",
        flush=True,
    )
    return {"f1_macro": test_f1, "accuracy": test_acc, "model": model}


# ── Main training function ─────────────────────────────────────────────────────


def train_strategy(cfg: TrainingConfig) -> dict:
    """
    Train a model for the given strategy and return evaluation results.
    Logs everything to MLflow.

    Returns a result dict suitable for the ablation comparison table.
    """
    import mlflow

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)

    strategy_info = TRAINING_STRATEGIES[cfg.strategy]
    run_name = f"strategy_{cfg.strategy}"
    output_dir = Path(cfg.output_dir) / cfg.strategy
    output_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(
            {
                "strategy": cfg.strategy,
                "description": strategy_info["description"],
                "base_model": BASE_MODEL_ID,
                "lr": cfg.lr,
                "batch_size": cfg.batch_size,
                "num_epochs": cfg.num_epochs,
                "fp16": cfg.fp16,
            }
        )

        model = build_model(cfg.strategy)
        param_info = count_parameters(model)
        mlflow.log_params(param_info)

        t0 = time.monotonic()

        if cfg.strategy == "independent":
            # Train three separate models
            final_metrics = {}
            for task, m in model.items():
                result = _train_single_task(task, m, cfg, device)
                final_metrics[task] = {
                    "f1_macro": result["f1_macro"],
                    "accuracy": result["accuracy"],
                }

            # Measure inference latency (sum of three forward passes)
            latency_ms = _measure_independent_latency(model, device)

        else:
            # Train one multi-task model
            lim = cfg.max_examples_per_task
            train_ds = load_multitask_dataset("train", limit_per_task=lim)
            val_ds = load_multitask_dataset("validation", limit_per_task=lim)
            test_ds = load_multitask_dataset("test", limit_per_task=lim)
            train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
            )
            val_loader = DataLoader(
                val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
            )
            test_loader = DataLoader(
                test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
            )

            model = model.to(device)
            optimizer = _build_optimizer(model, cfg)
            total_steps = len(train_loader) * cfg.num_epochs
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(total_steps * cfg.warmup_ratio),
                num_training_steps=total_steps,
            )
            scaler = (
                torch.amp.GradScaler("cuda") if cfg.fp16 and "cuda" in device else None
            )

            best_val_f1 = 0.0
            for epoch in range(cfg.num_epochs):
                train_metrics = _multitask_train_epoch(
                    model, train_loader, optimizer, scheduler, device, scaler
                )
                val_metrics = _multitask_eval(model, val_loader, device)

                # Log uncertainty weights if applicable
                if cfg.strategy == "uncertainty_weighted":
                    mlflow.log_metrics(
                        {
                            "log_sigma_sentiment": model.log_sigma_sentiment.item(),
                            "log_sigma_emotion": model.log_sigma_emotion.item(),
                            "log_sigma_toxicity": model.log_sigma_toxicity.item(),
                        },
                        step=epoch,
                    )

                avg_f1 = np.mean(
                    [
                        val_metrics[t]["f1_macro"]
                        for t in ["sentiment", "emotion", "toxicity"]
                    ]
                )
                mlflow.log_metrics(
                    {
                        **train_metrics,
                        "val_avg_f1": avg_f1,
                        **{
                            f"val_{t}_f1": val_metrics[t]["f1_macro"]
                            for t in val_metrics
                        },
                        **{
                            f"val_{t}_acc": val_metrics[t]["accuracy"]
                            for t in val_metrics
                        },
                    },
                    step=epoch,
                )

                print(
                    f"[{cfg.strategy}] epoch {epoch + 1}/{cfg.num_epochs} "
                    f"train_loss={train_metrics['train_loss']:.4f} "
                    f"val_avg_f1={avg_f1:.4f} "
                    f"(sent={val_metrics['sentiment']['f1_macro']:.3f} "
                    f"emo={val_metrics['emotion']['f1_macro']:.3f} "
                    f"tox={val_metrics['toxicity']['f1_macro']:.3f})",
                    flush=True,
                )

                if avg_f1 > best_val_f1:
                    best_val_f1 = avg_f1
                    torch.save(model.state_dict(), output_dir / "best_model.pt")

            # Final test evaluation
            model.load_state_dict(
                torch.load(output_dir / "best_model.pt", map_location=device)
            )
            final_metrics = _multitask_eval(model, test_loader, device)
            mlflow.log_metrics(
                {f"test_{t}_f1": final_metrics[t]["f1_macro"] for t in final_metrics}
            )
            mlflow.log_metrics(
                {f"test_{t}_acc": final_metrics[t]["accuracy"] for t in final_metrics}
            )

            # Log final uncertainty weights
            if cfg.strategy == "uncertainty_weighted":
                final_weights = {
                    "final_log_sigma_sentiment": model.log_sigma_sentiment.item(),
                    "final_log_sigma_emotion": model.log_sigma_emotion.item(),
                    "final_log_sigma_toxicity": model.log_sigma_toxicity.item(),
                }
                mlflow.log_metrics(final_weights)

            # Measure inference latency (single forward pass)
            latency_ms = _measure_multitask_latency(model, device)

        duration_s = time.monotonic() - t0
        mlflow.log_metrics(
            {
                "training_duration_seconds": duration_s,
                "latency_cpu_mean_ms": latency_ms["mean_ms"],
                "latency_cpu_p99_ms": latency_ms["p99_ms"],
            }
        )

        result = {
            "strategy": cfg.strategy,
            "description": strategy_info["description"],
            "metrics": final_metrics,
            "latency_ms": latency_ms,
            "param_info": param_info,
            "training_duration_s": round(duration_s, 1),
            "run_id": run.info.run_id,
        }

        result_path = output_dir / "result.json"
        result_path.write_text(json.dumps(result, indent=2))
        mlflow.log_artifact(str(result_path))

    return result


# ── Latency utilities ─────────────────────────────────────────────────────────


def _measure_multitask_latency(model, device: str, n_warmup=20, n_runs=100) -> dict:
    """Measure single multi-task forward pass latency."""
    model.eval()
    dummy_ids = torch.randint(0, 1000, (1, 128)).to(device)
    dummy_mask = torch.ones(1, 128, dtype=torch.long).to(device)

    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy_ids, dummy_mask)

    import time as _time

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = _time.perf_counter()
            model(dummy_ids, dummy_mask)
            latencies.append((_time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {"mean_ms": float(arr.mean()), "p99_ms": float(np.percentile(arr, 99))}


def _measure_independent_latency(
    model_dict: dict, device: str, n_warmup=20, n_runs=100
) -> dict:
    """Measure sum of three single-task forward passes (independent ablation)."""
    import time as _time

    dummy_ids = torch.randint(0, 1000, (1, 128)).to(device)
    dummy_mask = torch.ones(1, 128, dtype=torch.long).to(device)

    models = list(model_dict.values())
    for m in models:
        m.eval().to(device)
        with torch.no_grad():
            for _ in range(n_warmup):
                m(dummy_ids, dummy_mask)

    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = _time.perf_counter()
            for m in models:
                m(dummy_ids, dummy_mask)
            latencies.append((_time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    return {"mean_ms": float(arr.mean()), "p99_ms": float(np.percentile(arr, 99))}
