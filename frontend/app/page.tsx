"use client";

import { useState } from "react";

type Entity = { text: string; entity_type: string; score: number; canonical_id: string | null };
type Result = {
  text: string;
  sentiment: string;
  sentiment_probs: Record<string, number>;
  emotion: string;
  emotion_probs: Record<string, number>;
  toxicity: string;
  toxicity_probs: Record<string, number>;
  uncertainty: number;
  entities: Entity[];
  inference_latency_ms: number;
  error?: string;
};

const SAMPLES = [
  "Just tried the new Apple Vision Pro and honestly it blew me away 🤯",
  "Tesla's customer service has been an absolute nightmare this week.",
  "Not sure how I feel about the Google layoffs — mixed emotions tbh.",
];

// each emotion gets its associative hue (see globals.css)
const EMO_COLOR: Record<string, string> = {
  joy: "var(--joy)",
  love: "var(--love)",
  surprise: "var(--surprise)",
  sadness: "var(--sadness)",
  fear: "var(--fear)",
  anger: "var(--anger)",
};

function pct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

function Tip({ text }: { text: string }) {
  return (
    <span className="tip" tabIndex={0} role="note" aria-label={text}>
      <span className="q" aria-hidden="true">?</span>
      <span className="pop">{text}</span>
    </span>
  );
}

const TIPS = {
  sentiment:
    "3-class sentiment (negative / neutral / positive) from one head of the shared RoBERTa backbone. The marker sits at P(positive) − P(negative); the percentage is the winning class's probability.",
  emotion:
    "6-class emotion from a second head on the same backbone (trained on dair-ai/emotion). Every class probability is shown, sorted high→low — the model always splits its confidence across all six.",
  toxicity:
    "Binary toxicity from a third head (a hate-speech proxy, trained on tweet_eval/hate). The gauge shows the model's P(toxic).",
  entities:
    "Named entities from a separate cased BERT-NER model. It relies on capitalization — write 'McDonald's', not 'mc donalds'. Recognised companies are mapped to a canonical brand id.",
} as const;

export default function Home() {
  const [text, setText] = useState("");
  const [res, setRes] = useState<Result | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function analyze(input?: string) {
    const t = (input ?? text).trim();
    if (!t) return;
    setText(t);
    setLoading(true);
    setErr(null);
    setRes(null);
    try {
      const r = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: t }),
      });
      const data = (await r.json()) as Result;
      if (!r.ok || data.error) throw new Error(data.error || `Request failed (${r.status})`);
      setRes(data);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  // sentiment marker: signed score in [-1, 1] → [0%, 100%]
  const sScore = res ? (res.sentiment_probs.positive ?? 0) - (res.sentiment_probs.negative ?? 0) : 0;
  const markerLeft = ((sScore + 1) / 2) * 100;
  const emotions = res
    ? Object.entries(res.emotion_probs).sort((a, b) => b[1] - a[1])
    : [];
  const toxP = res?.toxicity_probs.toxic ?? 0;
  const toxic = res?.toxicity === "toxic";

  return (
    <main className="wrap">
      <header>
        <span className="eyebrow">Multi-task NLP · Social listening</span>
        <h1>
          Read the <span className="em">signal</span> in every sentence.
        </h1>
        <p className="lede">
          One shared RoBERTa pass scores <b>sentiment</b>, all six <b>emotions</b>, and{" "}
          <b>toxicity</b> at once — then pulls out the <b>brands</b> being talked about. Type
          anything and watch it decode.
        </p>
        <span className="live">
          <span className="dot" /> live · served from Google Cloud Run
        </span>
      </header>

      <div className="explainer">
        <span className="lead">How this works</span>
        <div className="body">
          Type any short message and one <b>RoBERTa</b> forward pass reads it three ways at once —
          a single shared backbone with three heads predicts <b>sentiment</b>, all six{" "}
          <b>emotions</b>, and <b>toxicity</b> simultaneously. A separate <b>NER</b> model then
          pulls out named entities and normalises known companies to a canonical brand id. Every
          panel below has a <b>?</b> that explains what it shows.
          <div className="flow">
            <span>your text</span> <i>→</i> <span>tokenize</span> <i>→</i>{" "}
            <span>RoBERTa · 3 heads</span> <i>→</i> <span>NER + brands</span> <i>→</i>{" "}
            <span>results</span>
          </div>
        </div>
      </div>

      <section className="console">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if ((e.metaKey || e.ctrlKey) && e.key === "Enter") analyze();
          }}
          placeholder="e.g. Amazon's delivery was late again and I'm so done with it…"
          rows={3}
          maxLength={2000}
        />
        <div className="row">
          <div className="samples">
            {SAMPLES.map((s, i) => (
              <button key={i} className="chip" disabled={loading} onClick={() => analyze(s)}>
                sample {i + 1}
              </button>
            ))}
          </div>
          <button className="go" disabled={loading || !text.trim()} onClick={() => analyze()}>
            {loading ? "Reading…" : "Analyze"}
            <span className="kbd">⌘↵</span>
          </button>
        </div>
        <p className="cold-note">
          {loading
            ? "Waking the model if it was idle — the first request can take ~20s."
            : "Free scale-to-zero backend: the first request after idle wakes the model (~20s), then it's fast."}
        </p>
      </section>

      {err && <div className="err">⚠ {err}</div>}

      {res && (
        <div className="results">
          {/* Signature: sentiment spectrum */}
          <section className="card reveal">
            <div className="card-h">
              <span className="t">
                Sentiment spectrum
                <Tip text={TIPS.sentiment} />
              </span>
              <span className="v">3-class</span>
            </div>
            <div className="spectrum-read">
              <span
                className="label"
                style={{
                  color:
                    res.sentiment === "positive"
                      ? "var(--pos)"
                      : res.sentiment === "negative"
                        ? "var(--neg)"
                        : "var(--neu)",
                }}
              >
                {res.sentiment}
              </span>
              <span className="conf">{pct(res.sentiment_probs[res.sentiment] ?? 0)} confident</span>
            </div>
            <div className="spectrum">
              <div className="marker" style={{ left: `${markerLeft}%` }} />
            </div>
            <div className="spectrum-scale">
              <span>negative</span>
              <span>neutral</span>
              <span>positive</span>
            </div>
          </section>

          <div className="two">
            {/* All six emotions */}
            <section className="card reveal">
              <div className="card-h">
                <span className="t">
                  Emotion
                  <Tip text={TIPS.emotion} />
                </span>
                <span className="v">6-class</span>
              </div>
              <div className="bars">
                {emotions.map(([name, p], i) => (
                  <div className={`bar ${i === 0 ? "top" : ""}`} key={name}>
                    <span className="name">{name}</span>
                    <span className="track">
                      <span
                        className="fill"
                        style={{ width: pct(p), background: EMO_COLOR[name] ?? "var(--neu)" }}
                      />
                    </span>
                    <span className="pct">{pct(p)}</span>
                  </div>
                ))}
              </div>
            </section>

            {/* Toxicity + entities */}
            <div style={{ display: "grid", gap: 16 }}>
              <section className="card reveal">
                <div className="card-h">
                  <span className="t">
                    Toxicity
                    <Tip text={TIPS.toxicity} />
                  </span>
                  <span className="v">binary</span>
                </div>
                <div className="gauge">
                  <span className="val" style={{ color: toxic ? "var(--neg)" : "var(--pos)" }}>
                    {toxic ? "Toxic" : "Clean"}
                  </span>
                  <span className="track">
                    <span
                      className="fill"
                      style={{ width: pct(toxP), background: toxic ? "var(--neg)" : "var(--pos)" }}
                    />
                  </span>
                  <span className="scale">{pct(toxP)} toxic probability</span>
                </div>
              </section>

              <section className="card reveal">
                <div className="card-h">
                  <span className="t">
                    Entities
                    <Tip text={TIPS.entities} />
                  </span>
                  <span className="v">NER + brands</span>
                </div>
                {res.entities.length === 0 ? (
                  <span className="empty">
                    No entities found. The NER model is case-sensitive — try proper
                    capitalization (e.g. “Tesla”, “McDonald&apos;s”), not “tesla” or “mc donalds”.
                  </span>
                ) : (
                  <div className="ents">
                    {res.entities.map((e, i) => (
                      <span className="ent" key={i}>
                        {e.text}
                        <span className="ty">{e.entity_type}</span>
                        {e.canonical_id && <span className="brand">{e.canonical_id}</span>}
                      </span>
                    ))}
                  </div>
                )}
              </section>
            </div>
          </div>

          <div className="meta">
            <span>
              predictive entropy <b>{res.uncertainty.toFixed(3)}</b>
            </span>
            <span>
              inference <b>{res.inference_latency_ms.toFixed(0)} ms</b>
            </span>
            <span>
              backbone <b>1× RoBERTa · 3 heads</b>
            </span>
          </div>
        </div>
      )}

      <footer className="foot">
        <span>Multi-task RoBERTa · single forward pass · trained in PyTorch</span>
        <a href="https://github.com/shiva-shivanibokka/NLP-Pipeline-at-Scale">source ↗</a>
      </footer>
    </main>
  );
}
