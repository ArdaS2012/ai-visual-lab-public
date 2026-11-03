import test from "node:test";
import assert from "node:assert/strict";
import {
  scoreAnswer,
  pairedResults,
  parseJudge,
  type BenchmarkCase,
  type ModelAnswer,
} from "../src/lib/evaluation-live.ts";
const row: BenchmarkCase = {
  id: "one",
  label: "One",
  question: "Seats?",
  reference: "18",
  aliases: ["18"],
  relevant: ["cedar"],
};
const answer = (raw: string, variant: "A" | "B" = "A"): ModelAnswer => ({
  raw,
  variant,
  case_id: "one",
  run_id: "run",
  supplied_sources: ["cedar"],
  messages: [],
  model: "test",
  revision: "test",
  prompt_tokens: 1,
  output_tokens: 1,
  seconds: 0,
  truncated: false,
});
test("correctness, JSON contract and supplied citation IDs stay independent", () => {
  assert.deepEqual(
    scoreAnswer(row, answer('{"answer":" 18 ","sources":["absent"]}')),
    {
      answer: " 18 ",
      contentExtraction: "Content read from JSON",
      schema: true,
      exact: true,
      sources: ["absent"],
      unknownSourceIds: ["absent"],
    },
  );
  const extra = scoreAnswer(
    row,
    answer('{"answer":"18","sources":[],"extra":true}'),
  );
  assert.equal(extra.exact, true);
  assert.equal(extra.schema, false);
  const fenced = scoreAnswer(
    row,
    answer('```json\n{"answer":18,"sources":["cedar"]}\n```'),
  );
  assert.equal(fenced.exact, true);
  assert.equal(fenced.schema, false);
  const plain = scoreAnswer(row, answer("18"));
  assert.equal(plain.exact, true);
  assert.equal(plain.schema, false);
  assert.equal(scoreAnswer(row, answer("18 seats")).exact, false);
  assert.equal(
    scoreAnswer(row, answer('{"answer":"13","sources":["cedar"]}')).exact,
    false,
  );
  assert.equal(
    scoreAnswer(row, { ...answer("18"), truncated: true }).exact,
    false,
  );
});
test("paired rates exclude unmatched cases and preserve same comparison denominator", () => {
  const a = answer("18");
  const b = answer("UNKNOWN", "B");
  const unmatched = { ...answer("18"), case_id: "two" };
  const summary = pairedResults(
    [a, b, unmatched],
    [row, { ...row, id: "two" }],
  );
  assert.equal(summary.n, 1);
  assert.deepEqual(summary.variants, [
    { variant: "A", exact: 1, schema: 0 },
    { variant: "B", exact: 0, schema: 0 },
  ]);
  assert.equal(pairedResults([a], [row]).n, 0);
});
test("unknown cases require abstention; malformed judge output cannot count as agreement", () => {
  const unknown = {
    ...row,
    reference: "UNKNOWN",
    aliases: ["UNKNOWN"],
    relevant: [],
  };
  assert.equal(scoreAnswer(unknown, answer("unknown")).exact, true);
  assert.equal(scoreAnswer(unknown, answer("0")).exact, false);
  const judge = (raw: string, truncated = false) => ({
    raw,
    truncated,
    model: "test",
    revision: "test",
    messages: [],
  });
  assert.deepEqual(parseJudge(judge('{"rating":2,"reason":"supported"}')), {
    rating: 2,
    reason: "supported",
  });
  for (const raw of [
    "2",
    '{"rating":true,"reason":"ok"}',
    '{"rating":3,"reason":"ok"}',
    '{"rating":2.1,"reason":"ok"}',
    '{"rating":2,"reason":"ok","extra":0}',
  ])
    assert.equal(parseJudge(judge(raw)), null);
  assert.equal(parseJudge(judge('{"rating":2,"reason":"ok"}', true)), null);
});
