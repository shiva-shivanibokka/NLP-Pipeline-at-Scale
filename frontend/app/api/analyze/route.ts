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
          // Do NOT promise a warm-up here. A timeout is equally consistent with
          // a cold start and with a backend that no longer exists — and on this
          // deployment it is the latter. Telling the user to wait 20s and retry
          // sends them round a loop that cannot terminate. The header badge
          // (via /api/health) reports which case this is.
          ? "No response from the model backend within 60s. If it is merely cold, a retry can succeed; if the backend is gone, it will not. The badge at the top of the page reports which."
          : `Model API unreachable: ${e instanceof Error ? e.message : String(e)}`,
      },
      { status: timedOut ? 503 : 502 },
    );
  }
}
