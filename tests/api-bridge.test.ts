import test from "node:test";
import assert from "node:assert/strict";
import { POST } from "../src/app/api/day2/[experiment]/route.ts";

const context = { params: Promise.resolve({ experiment: "retrieval" }) };
const url = "http://127.0.0.1:3001/api/day2/retrieval";

test("foreign webpages cannot trigger local model experiments", async () => {
  const cases: Record<string, string>[] = [
    { origin: "https://example.com", "content-type": "application/json" },
    { "sec-fetch-site": "cross-site", "content-type": "application/json" },
  ];
  for (const headers of cases) {
    const response = await POST(new Request(url, { method: "POST", headers, body: "{}" }), context);
    assert.equal(response.status, 403);
  }
});

test("form posts cannot bypass the JSON API boundary", async () => {
  const response = await POST(new Request(url, { method: "POST", body: "{}" }), context);
  assert.equal(response.status, 415);
});

test("body limit also covers requests without Content-Length", async () => {
  const request = new Request(url, { method: "POST", headers: { "content-type": "application/json" }, body: "x".repeat(16385) });
  assert.equal(request.headers.has("content-length"), false);
  assert.equal((await POST(request, context)).status, 413);
});


test("browser origin follows Host when Next.js uses an internal hostname", async (t) => {
  const upstream = t.mock.method(globalThis, "fetch", async () => new Response("ok"));
  for (const host of ["127.0.0.1:3001", "localhost:3001", "[::1]:3001"]) {
    const response = await POST(new Request("http://localhost:3001/api/day2/retrieval", {
      method: "POST",
      headers: { host, origin: `http://${host}`, "sec-fetch-site": "same-origin", "content-type": "application/json" },
      body: "{}",
    }), context);
    assert.equal(response.status, 200);
  }
  assert.equal(upstream.mock.callCount(), 3);
});

test("forwarded host and same-origin metadata cannot authorize a foreign origin", async () => {
  const response = await POST(new Request(url, {
    method: "POST",
    headers: { host: "127.0.0.1:3001", origin: "https://example.com", "x-forwarded-host": "example.com", "sec-fetch-site": "same-origin", "content-type": "application/json" },
    body: "{}",
  }), context);
  assert.equal(response.status, 403);
});

test("matching host with a different port or protocol is still cross-origin", async () => {
  for (const origin of ["http://127.0.0.1:4001", "https://127.0.0.1:3001", "null"]) {
    const response = await POST(new Request(url, { method: "POST", headers: { host: "127.0.0.1:3001", origin, "content-type": "application/json" }, body: "{}" }), context);
    assert.equal(response.status, 403);
  }
});
