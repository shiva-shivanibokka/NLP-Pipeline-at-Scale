import { NextRequest, NextResponse } from "next/server";

// Server-side proxy to the model API (a free Hugging Face Space). Keeps the
// backend URL server-only (no CORS, not exposed to the browser). Set
// MODEL_API_URL in the Vercel project's environment variables.
export const runtime = "nodejs";
export const dynamic = "force-dynamic";

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
      // Topics need the streaming pipeline populated, which isn't on the Space.
      body: JSON.stringify({ text, include_entities: true, include_topics: false }),
      signal: AbortSignal.timeout(30_000),
    });
    const data = await upstream.json().catch(() => ({ error: "Bad upstream response" }));
    return NextResponse.json(data, { status: upstream.status });
  } catch (e) {
    return NextResponse.json(
      { error: `Model API unreachable: ${e instanceof Error ? e.message : String(e)}` },
      { status: 502 },
    );
  }
}
