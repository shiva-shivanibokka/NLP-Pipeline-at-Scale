import { NextResponse } from "next/server";

// Liveness probe for the badge in the header. The badge used to be hardcoded to
// "live · served from Google Cloud Run", which stayed on screen after the Cloud
// Run service stopped existing -- a page that asserts its own backend is up is
// asserting something it has not checked.
//
// Returns { up: boolean, reason?: string } and never throws, so the badge can
// distinguish three states: checking, up, down.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Shorter than /api/analyze's 60s: a cold start is not "up" for badge purposes,
// and the header should settle quickly rather than spin for a minute.
export const maxDuration = 15;

const API = process.env.MODEL_API_URL;

export async function GET() {
  if (!API) {
    return NextResponse.json({ up: false, reason: "MODEL_API_URL not configured" });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 10_000);
  try {
    const r = await fetch(`${API.replace(/\/$/, "")}/health`, {
      signal: controller.signal,
      cache: "no-store",
    });
    if (!r.ok) {
      return NextResponse.json({ up: false, reason: `backend returned ${r.status}` });
    }
    // 2xx is not sufficient. The backend's /health also reports model_trained,
    // and an untrained backbone returns well-formed, meaningless probabilities
    // — a green "reachable" badge over that is the same false assurance in a
    // different place. Only a literal `false` counts against it; an older
    // backend that omits the field reports undefined and is not punished.
    const body = (await r.json().catch(() => ({}))) as { model_trained?: boolean };
    if (body.model_trained === false) {
      return NextResponse.json({
        up: false,
        reason: "backend is serving an untrained model (predictions are meaningless)",
      });
    }
    return NextResponse.json({ up: true });
  } catch (e) {
    const reason = e instanceof Error && e.name === "AbortError"
      ? "backend did not respond within 10s"
      : "backend unreachable";
    return NextResponse.json({ up: false, reason });
  } finally {
    clearTimeout(timer);
  }
}
