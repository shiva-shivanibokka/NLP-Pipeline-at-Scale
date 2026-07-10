"use client";

import { useState } from "react";

type Entity = {
  text: string;
  entity_type: string;
  score: number;
  canonical_id: string | null;
};

type AnalyzeResult = {
  text: string;
  sentiment: string;
  sentiment_score: number;
  emotion: string;
  emotion_score: number;
  toxicity: string;
  toxicity_score: number;
  uncertainty: number;
  entities: Entity[];
  inference_latency_ms: number;
  error?: string;
};

const SAMPLES = [
  "Just tried the new Apple Vision Pro and honestly it blew me away 🤯",
  "Tesla's customer service has been an absolute nightmare this week.",
  "Not sure how I feel about the Google layoffs, mixed emotions tbh.",
];

const toneClass = (label: string) => {
  const l = label.toLowerCase();
  if (["positive", "joy", "love", "not_toxic"].includes(l)) return "good";
  if (["negative", "anger", "fear", "sadness", "toxic"].includes(l)) return "bad";
  return "neutral";
};

function ScoreCard({
  title,
  label,
  score,
}: {
  title: string;
  label: string;
  score: number;
}) {
  const tone = toneClass(label);
  return (
    <div className="card">
      <div className="card-title">{title}</div>
      <div className={`card-label ${tone}`}>{label.replace("_", " ")}</div>
      <div className="bar">
        <div className={`bar-fill ${tone}`} style={{ width: `${Math.round(score * 100)}%` }} />
      </div>
      <div className="card-score">{(score * 100).toFixed(1)}% confidence</div>
    </div>
  );
}

export default function Home() {
  const [text, setText] = useState("");
  const [result, setResult] = useState<AnalyzeResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function analyze(input?: string) {
    const t = (input ?? text).trim();
    if (!t) return;
    setText(t);
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const r = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
      });
      const data = (await r.json()) as AnalyzeResult;
      if (!r.ok || data.error) throw new Error(data.error || `Request failed (${r.status})`);
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="wrap">
      <header className="head">
        <div className="kicker">multi-task nlp · single forward pass</div>
        <h1>Real-Time Text Analyzer</h1>
        <p className="sub">
          One shared RoBERTa backbone predicts <strong>sentiment</strong>,{" "}
          <strong>emotion</strong>, and <strong>toxicity</strong> together, then extracts
          named entities and normalizes them to canonical brands.
        </p>
      </header>

      <section className="panel">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
          }}
          placeholder="Type or paste a tweet-length message…"
          rows={4}
          maxLength={2000}
        />
        <div className="controls">
          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button key={i} className="chip" onClick={() => analyze(s)} disabled={loading}>
                sample {i + 1}
              </button>
            ))}
          </div>
          <button className="go" onClick={() => analyze()} disabled={loading || !text.trim()}>
            {loading ? "Analyzing…" : "Analyze"}
            <span className="hint">⌘↵</span>
          </button>
        </div>
      </section>

      {error && <div className="error">⚠ {error}</div>}

      {result && (
        <section className="results">
          <div className="grid">
            <ScoreCard title="Sentiment" label={result.sentiment} score={result.sentiment_score} />
            <ScoreCard title="Emotion" label={result.emotion} score={result.emotion_score} />
            <ScoreCard title="Toxicity" label={result.toxicity} score={result.toxicity_score} />
          </div>

          <div className="entities">
            <div className="card-title">Named entities</div>
            {result.entities.length === 0 ? (
              <div className="muted">No entities detected.</div>
            ) : (
              <div className="chips">
                {result.entities.map((e, i) => (
                  <span key={i} className="entity">
                    {e.text}
                    <em>{e.entity_type}</em>
                    {e.canonical_id && <b>{e.canonical_id}</b>}
                  </span>
                ))}
              </div>
            )}
          </div>

          <div className="meta">
            <span>
              uncertainty (entropy) <strong>{result.uncertainty.toFixed(3)}</strong>
            </span>
            <span>
              inference <strong>{result.inference_latency_ms.toFixed(1)} ms</strong>
            </span>
          </div>
        </section>
      )}

      <footer className="foot">
        Backend: FastAPI on a free Hugging Face Space · Frontend: Next.js on Vercel ·{" "}
        <a href="https://github.com/sbokk/NLP-Pipeline-at-Scale">source</a>
      </footer>
    </main>
  );
}
