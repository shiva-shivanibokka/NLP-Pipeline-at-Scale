"""
Gradio dashboard for the NLP Pipeline at Scale.

Five tabs:
1. Live Text Analysis    — type any text, get real-time multi-task predictions
2. Brand Sentiment       — rolling EMA sentiment per brand, SPC alerts
3. Topic Explorer        — trending topics and top words
4. Benchmark Results     — throughput/latency table and active learning curves
5. Annotation Interface  — label uncertain examples (active learning demo)
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr
import plotly.graph_objects as go
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

import configs.quiet  # noqa: F401,E402 — sets USE_TF=0 etc. before transformers loads

API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


# ── API helpers ────────────────────────────────────────────────────────────────


def _api(endpoint: str, method: str = "GET", json_body=None):
    try:
        if method == "POST":
            r = requests.post(f"{API_BASE}{endpoint}", json=json_body, timeout=10)
        else:
            r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Tab 1: Live Text Analysis ─────────────────────────────────────────────────


def analyze_text(text: str):
    if not text.strip():
        return "Please enter some text.", None, None

    result = _api(
        "/analyze",
        "POST",
        {"text": text, "include_entities": True, "include_topics": True},
    )

    if "error" in result:
        return f"API Error: {result['error']}", None, None

    # Summary text
    summary = (
        f"**Sentiment:** {result['sentiment'].upper()} (score={result['sentiment_score']:.3f})\n"
        f"**Emotion:** {result['emotion'].upper()} (score={result['emotion_score']:.3f})\n"
        f"**Toxicity:** {result['toxicity'].upper()} (score={result['toxicity_score']:.3f})\n"
        f"**Uncertainty:** {result['uncertainty']:.4f} (higher = less confident)\n"
        f"**Topic ID:** {result['topic_id']}\n"
        f"**Inference:** {result['inference_latency_ms']:.1f}ms\n\n"
    )

    if result.get("entities"):
        entity_lines = []
        for ent in result["entities"]:
            cid = f" → `{ent['canonical_id']}`" if ent.get("canonical_id") else ""
            entity_lines.append(
                f"- **{ent['text']}** [{ent['entity_type']}, conf={ent['score']:.3f}]{cid}"
            )
        summary += "**Named Entities:**\n" + "\n".join(entity_lines)

    # Sentiment probability bar
    s_probs = result.get("sentiment_probs", {})
    sentiment_fig = go.Figure(
        go.Bar(
            x=list(s_probs.keys()),
            y=list(s_probs.values()),
            marker_color=["#E15759", "#76B7B2", "#59A14F"],
        )
    )
    sentiment_fig.update_layout(
        title="Sentiment Probability",
        yaxis_range=[0, 1],
        yaxis_title="Probability",
        xaxis_title="Class",
        height=300,
    )

    # Emotion breakdown
    e_probs = result.get("emotion_probs", {})
    if e_probs:
        emotion_fig = go.Figure(
            go.Bar(
                x=list(e_probs.keys()),
                y=list(e_probs.values()),
                marker_color="#4E79A7",
            )
        )
        emotion_fig.update_layout(
            title="Emotion Probability",
            yaxis_range=[0, 1],
            height=300,
        )
    else:
        emotion_fig = None

    return summary, sentiment_fig, emotion_fig


# ── Tab 2: Brand Sentiment Monitor ────────────────────────────────────────────


def get_brand_dashboard():
    brands = _api("/brands")
    if "error" in brands or not brands:
        return "No brand data yet. Run the streaming pipeline first.", None

    # Build DataFrame-style display
    rows = []
    for b in brands[:15]:
        alert = " ⚠️" if b.get("active_alert") else ""
        rows.append(
            f"| `{b['brand_id']}` | {b['ema_sentiment']:+.3f} | "
            f"{b['message_count']:,} | {b['toxicity_rate']:.2%} |{alert}"
        )

    table = "| Brand | Sentiment EMA | Messages | Toxicity Rate | Alert |\n"
    table += "|---|---|---|---|---|\n"
    table += "\n".join(rows)

    alerts = _api("/alerts")
    alerts_text = ""
    if alerts and "error" not in alerts:
        alert_lines = [
            f"- **{a['brand_id']}**: sentiment={a['ema_sentiment']:+.3f}, "
            f"streak={a['alert_streak']} windows"
            for a in alerts
        ]
        alerts_text = (
            "\n\n**Active Alerts:**\n" + "\n".join(alert_lines) if alert_lines else ""
        )

    return table + alerts_text, None


# ── Tab 3: Topic Explorer ──────────────────────────────────────────────────────


def get_trending_topics():
    topics = _api("/topics/trending")
    if "error" in topics or not topics:
        return "No topic data yet. Topics appear after ~256 documents are processed."

    lines = ["**Top Trending Topics (last 5 minutes):**\n"]
    for t in topics[:10]:
        words = ", ".join(t.get("top_words", [])[:6])
        velocity = t.get("velocity", 0)
        trend = "↑" if velocity > 0 else ("↓" if velocity < 0 else "→")
        lines.append(
            f"- **Topic {t['topic_id']}** {trend} velocity={velocity:+.3f}: {words}"
        )

    return "\n".join(lines)


# ── Tab 4: Benchmark Results ───────────────────────────────────────────────────


def load_benchmark_results():
    bench_path = Path("results/benchmark/throughput_benchmark.json")
    al_path = Path("results/active_learning/comparison_summary.json")
    ablation_path = Path("results/ablation/comparison.json")

    output = []

    # Ablation table
    if ablation_path.exists():
        try:
            ablation = json.loads(ablation_path.read_text())
            output.append("## Multi-Task Ablation Study\n")
            output.append(
                "| Strategy | Sentiment F1 | Emotion F1 | Toxicity F1 | Latency p99 |"
            )
            output.append("|---|---|---|---|---|")
            for r in ablation:
                m = r.get("metrics", {})
                s_f1 = m.get("sentiment", {}).get("f1_macro", 0)
                e_f1 = m.get("emotion", {}).get("f1_macro", 0)
                t_f1 = m.get("toxicity", {}).get("f1_macro", 0)
                lat = r.get("latency_ms", {}).get("p99_ms", 0)
                output.append(
                    f"| {r['strategy']} | {s_f1:.4f} | {e_f1:.4f} | {t_f1:.4f} | {lat:.1f}ms |"
                )
            output.append("")
        except Exception:
            pass

    # Throughput benchmark
    if bench_path.exists():
        try:
            bench = json.loads(bench_path.read_text())
            output.append("## Throughput vs. Latency Benchmark\n")
            output.append(
                f"Batch size: {bench.get('batch_size')}, Max wait: {bench.get('max_wait_ms')}ms\n"
            )
            output.append(
                "| Target (msg/s) | Actual Producer | Consumer | Lag | p50 | p99 | Ratio | Status |"
            )
            output.append("|---|---|---|---|---|---|---|---|")
            for r in bench.get("results", []):
                status = "SATURATED" if r.get("saturated") else "OK"
                output.append(
                    f"| {r['target_msgs_per_sec']} | {r['actual_producer_msgs_per_sec']:.0f} "
                    f"| {r['actual_consumer_msgs_per_sec']:.0f} | {r['consumer_lag_end']} "
                    f"| {r['latency_p50_ms']:.1f}ms | {r['latency_p99_ms']:.1f}ms "
                    f"| {r['throughput_ratio']:.2f} | {status} |"
                )
            sat = bench.get("saturation_point_msgs_per_sec")
            if sat:
                output.append(f"\n**Saturation point: ~{sat} msg/s**")
            output.append("")
        except Exception:
            pass

    # Active learning
    if al_path.exists():
        try:
            al = json.loads(al_path.read_text())
            output.append("## Active Learning: Uncertainty Sampling vs. Random\n")
            output.append("| Labeled Examples | Uncertainty F1 | Random F1 | Gain |")
            output.append("|---|---|---|---|")
            unc_f1s = al.get("uncertainty_entropy", {}).get("f1_scores", [])
            rnd_f1s = al.get("random", {}).get("f1_scores", [])
            sizes = al.get("uncertainty_entropy", {}).get("labeled_sizes", [])
            for size, u_f1, r_f1 in zip(sizes, unc_f1s, rnd_f1s):
                gain = u_f1 - r_f1
                output.append(f"| {size} | {u_f1:.4f} | {r_f1:.4f} | {gain:+.4f} |")
            output.append("")
        except Exception:
            pass

    if not output:
        return (
            "No results found yet.\n\n"
            "Run experiments first:\n"
            "```bash\n"
            "python scripts/run_ablation.py\n"
            "python scripts/run_benchmark.py\n"
            "python scripts/run_active_learning.py\n"
            "```"
        )

    return "\n".join(output)


# ── Tab 5: Annotation Interface ────────────────────────────────────────────────

_annotation_state = {"pool": [], "annotations": [], "index": 0}


def _load_annotation_pool():
    """Load high-uncertainty examples from the unlabeled pool."""
    try:
        from src.model.dataset import load_unlabeled_pool
        from src.model.multitask_model import SingleTaskRoBERTa
        from src.active_learning.loop import compute_entropy_scores
        from configs.config import BASE_MODEL_ID, NUM_SENTIMENT_CLASSES

        _, pool_ds = load_unlabeled_pool(pool_size=500)

        # Pretrained backbone — a random-init model ranks uncertainty at chance.
        model = SingleTaskRoBERTa.build_pretrained(NUM_SENTIMENT_CLASSES, BASE_MODEL_ID)
        entropies = compute_entropy_scores(model, pool_ds)

        top_k = 50
        indices = list(reversed(entropies.argsort()))[:top_k]

        # Decode back to text (rough)
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
        decoded = [
            tok.decode(pool_ds[i]["input_ids"], skip_special_tokens=True)
            for i in indices
        ]
        return list(zip(decoded, [float(entropies[i]) for i in indices]))
    except Exception:
        # Fallback: show example texts
        return [
            ("This new product update from Apple has me feeling mixed...", 0.89),
            ("I can't tell if the service improved or got worse", 0.91),
            ("The results were neither good nor bad honestly", 0.88),
        ]


def load_next_example():
    """Get the next high-uncertainty example for annotation."""
    if not _annotation_state["pool"]:
        _annotation_state["pool"] = _load_annotation_pool()
        _annotation_state["index"] = 0

    pool = _annotation_state["pool"]
    idx = _annotation_state["index"]

    if idx >= len(pool):
        return "All examples annotated! Restart for a new batch.", 0.0, 0

    text, entropy = pool[idx]
    return text, round(entropy, 4), idx


def submit_annotation(text: str, label: str):
    """Record a human annotation and move to the next example."""
    _annotation_state["annotations"].append({"text": text, "label": label})
    _annotation_state["index"] += 1
    total = len(_annotation_state["annotations"])
    next_text, entropy, _ = load_next_example()
    return (
        f"Saved! {total} examples annotated so far.",
        next_text,
        f"Entropy: {entropy:.4f} (higher = more uncertain)",
    )


# ── Build the Gradio interface ─────────────────────────────────────────────────

with gr.Blocks(title="NLP Pipeline at Scale") as demo:
    gr.Markdown(
        """# NLP Pipeline at Scale — Real-Time Social Listening

**Multi-task RoBERTa** (joint sentiment + emotion + toxicity, uncertainty-weighted loss) +
**Kafka streaming** + **Online BERTopic** + **Redis SPC anomaly detection** + **Active Learning**

Grounded in: Kendall et al. (2018) uncertainty weighting · Settles (2009) active learning survey
"""
    )

    with gr.Tabs():
        # ── Tab 1 ──────────────────────────────────────────────────────────────
        with gr.TabItem("Live Text Analysis"):
            gr.Markdown(
                "Enter any text to see multi-task NLP predictions in real time."
            )
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Apple's new product launch generated a lot of mixed reactions on social media...",
                lines=3,
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
            analysis_output = gr.Markdown()
            with gr.Row():
                sentiment_plot = gr.Plot(label="Sentiment Probabilities")
                emotion_plot = gr.Plot(label="Emotion Probabilities")

            analyze_btn.click(
                fn=analyze_text,
                inputs=[text_input],
                outputs=[analysis_output, sentiment_plot, emotion_plot],
            )

            gr.Examples(
                examples=[
                    [
                        "Apple's quarterly earnings exceeded expectations but the stock fell anyway."
                    ],
                    [
                        "I love how fast the new Google search is, but I hate the layout change."
                    ],
                    ["This is completely unacceptable, I want a refund immediately!"],
                    ["Not sure what to make of the latest Tesla announcement..."],
                ],
                inputs=[text_input],
            )

        # ── Tab 2 ──────────────────────────────────────────────────────────────
        with gr.TabItem("Brand Sentiment Monitor"):
            gr.Markdown("Rolling EMA brand sentiment and SPC anomaly alerts.")
            refresh_btn = gr.Button("Refresh", size="sm")
            brand_output = gr.Markdown()
            refresh_btn.click(
                fn=lambda: get_brand_dashboard()[0], outputs=[brand_output]
            )
            demo.load(fn=lambda: get_brand_dashboard()[0], outputs=[brand_output])

        # ── Tab 3 ──────────────────────────────────────────────────────────────
        with gr.TabItem("Topic Explorer"):
            gr.Markdown("Trending topics discovered via online incremental BERTopic.")
            topic_refresh = gr.Button("Refresh Topics", size="sm")
            topic_output = gr.Markdown()
            topic_refresh.click(fn=get_trending_topics, outputs=[topic_output])
            demo.load(fn=get_trending_topics, outputs=[topic_output])

        # ── Tab 4 ──────────────────────────────────────────────────────────────
        with gr.TabItem("Benchmark & Results"):
            gr.Markdown(
                "Ablation study results, throughput benchmark, and active learning comparison."
            )
            bench_refresh = gr.Button("Refresh Results", size="sm")
            bench_output = gr.Markdown(value=load_benchmark_results())
            bench_refresh.click(fn=load_benchmark_results, outputs=[bench_output])

        # ── Tab 5 ──────────────────────────────────────────────────────────────
        with gr.TabItem("Annotation Interface"):
            gr.Markdown(
                "**Active Learning Demo**: Label high-uncertainty examples identified by entropy sampling.\n\n"
                "These are the examples the model is most confused about — your annotations will have "
                "the highest impact on model performance."
            )
            load_btn = gr.Button("Load Next Uncertain Example", variant="primary")
            with gr.Row():
                example_text = gr.Textbox(
                    label="Tweet Text (read-only)", lines=2, interactive=False
                )
                entropy_display = gr.Textbox(
                    label="Uncertainty Score", interactive=False
                )

            label_input = gr.Radio(
                choices=["negative", "neutral", "positive"],
                label="Correct Sentiment Label",
                value="neutral",
            )
            submit_btn = gr.Button("Submit Label", variant="secondary")
            status_text = gr.Markdown()

            load_btn.click(
                fn=lambda: (
                    load_next_example()[0],
                    f"Entropy: {load_next_example()[1]:.4f}",
                ),
                outputs=[example_text, entropy_display],
            )
            submit_btn.click(
                fn=submit_annotation,
                inputs=[example_text, label_input],
                outputs=[status_text, example_text, entropy_display],
            )


if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
