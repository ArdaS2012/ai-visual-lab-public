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
