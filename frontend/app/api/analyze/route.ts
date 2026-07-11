import { NextRequest, NextResponse } from "next/server";

// Server-side proxy to the model API (FastAPI on Cloud Run). Keeps the backend URL
// server-only (no CORS, not exposed to the browser). Set MODEL_API_URL in the Vercel
// project's environment variables.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";
// Allow the function to wait out a Cloud Run cold start (scale-to-zero loads ~1.5GB of
// models on the first request). 60s is the Vercel Hobby ceiling; raise on Pro if needed.
export const maxDuration = 60;

const API = process.env.MODEL_API_URL;

export async function POST(req: NextRequest) {
  if (!API) {
    return NextResponse.json(
      { error: "MODEL_API_URL is not configured on the server." },
      { status: 500 },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const text = (body as { text?: unknown })?.text;
  if (typeof text !== "string" || !text.trim()) {
    return NextResponse.json({ error: "Field 'text' is required." }, { status: 400 });
  }
  if (text.length > 2000) {
    return NextResponse.json({ error: "Text too long (max 2000 chars)." }, { status: 413 });
  }

  try {
    const upstream = await fetch(`${API.replace(/\/$/, "")}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, include_entities: true, include_topics: false }),
      // A bit under maxDuration so we can return a friendly message instead of a
      // hard function timeout.
      signal: AbortSignal.timeout(57_000),
    });
    const data = await upstream.json().catch(() => ({ error: "Bad upstream response" }));
    return NextResponse.json(data, { status: upstream.status });
  } catch (e) {
    const timedOut =
      e instanceof Error && (e.name === "TimeoutError" || /timeout|abort/i.test(e.message));
    return NextResponse.json(
      {
        error: timedOut
          ? "The model was asleep and is waking up (free scale-to-zero backend). Give it ~20s and click Analyze again — it stays fast once warm."
          : `Model API unreachable: ${e instanceof Error ? e.message : String(e)}`,
      },
      { status: timedOut ? 503 : 502 },
    );
  }
}
