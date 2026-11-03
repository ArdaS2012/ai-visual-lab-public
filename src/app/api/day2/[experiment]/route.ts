export const runtime = "nodejs";
export const dynamic = "force-dynamic";

async function bridge(
  request: Request,
  context: { params: Promise<{ experiment: string }> },
) {
  const { experiment } = await context.params;
  if (!["health", "retrieval", "agent", "evaluation", "evaluation-judge"].includes(experiment))
    return Response.json({ error: "Unknown experiment" }, { status: 404 });
  if ((experiment === "health") !== (request.method === "GET"))
    return Response.json({ error: "Method not allowed" }, { status: 405 });
  let body: string | undefined;
  if (request.method === "POST") {
    // A third-party website must not be able to trigger local CPU experiments.
    const origin = request.headers.get("origin");
    if (request.headers.get("sec-fetch-site") === "cross-site" ||
        (origin !== null && origin !== new URL(request.url).origin))
      return Response.json({ error: "Cross-origin experiment requests are not allowed" }, { status: 403 });
    if (request.headers.get("content-type")?.split(";")[0].trim() !== "application/json")
      return Response.json({ error: "Use application/json" }, { status: 415 });
    const reader = request.body?.getReader();
    const chunks: Uint8Array[] = [];
    let bytes = 0;
    if (reader) {
      while (true) {
        const part = await reader.read();
        if (part.done) break;
        bytes += part.value.byteLength;
        if (bytes > 16384) {
          await reader.cancel();
          return Response.json({ error: "Request body exceeds 16 KiB" }, { status: 413 });
        }
        chunks.push(part.value);
      }
    }
    const merged = new Uint8Array(bytes);
    let offset = 0;
    for (const chunk of chunks) { merged.set(chunk, offset); offset += chunk.byteLength; }
    body = new TextDecoder().decode(merged);
  }
  try {
    const upstream = await fetch(`http://127.0.0.1:8010/${experiment}`, {
      method: request.method,
      headers: { "Content-Type": "application/json" },
      body,
      signal: request.signal,
      cache: "no-store",
    });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        "Content-Type":
          upstream.headers.get("Content-Type") ?? "application/json",
        "Cache-Control": "no-store",
      },
    });
  } catch {
    return Response.json(
      {
        error:
          "Local model runtime unavailable. Start the Visual Lab runtime (see README).",
      },
      { status: 503 },
    );
  }
}
export const GET = bridge;
export const POST = bridge;
