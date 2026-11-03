import test from "node:test";
import assert from "node:assert/strict";
import { isDirectExtension } from "../src/lib/llm-protocol.ts";
import {
  distribution,
  draw,
  mulberry32,
  softmax,
} from "../src/lib/llm-sampling.ts";
const settings = { temperature: 1, topK: 0, topP: 1 };
const logits = [0.64, 0.24, 0.09, 0.03].map(Math.log);
const close = (a: number, b: number) =>
  assert.ok(Math.abs(a - b) < 1e-10, `${a} != ${b}`);

test("KV reuse rejects resets, changed prefixes and nonadjacent sequences", () => {
  assert.equal(isDirectExtension([1, 8, 2], [1, 8, 2, 42]), true);
  for (const next of [[1, 8], [1, 8, 2], [1, 9, 2, 42], [1, 8, 2, 42, 7]])
    assert.equal(isDirectExtension([1, 8, 2], next), false);
  assert.equal(isDirectExtension([], [42]), false);
});

test("nucleus includes crossing candidate, equality closes, p=1 keeps all", () => {
  for (const [p, expected] of [
    [0.8, 2],
    [0.95, 3],
    [1, 4],
    [0.1, 1],
    [0.88, 2],
    [0.64, 1],
  ]) {
    const rows = distribution(logits, { ...settings, topP: p });
    assert.equal(rows.filter((x) => !x.droppedBy).length, expected);
    close(
      rows.reduce((s, x) => s + x.probability, 0),
      1,
    );
    assert.ok(rows.slice(expected).every((x) => x.probability === 0));
  }
  const rows = distribution(logits, { ...settings, topP: 0.8 });
  close(rows[0].probability, 0.64 / 0.88);
  close(rows[1].probability, 0.24 / 0.88);
});
test("top-k then normalized top-p; ties deterministic; greedy bypasses filters", () => {
  const rows = distribution(logits, { ...settings, topK: 2, topP: 0.7 });
  assert.deepEqual(
    rows.filter((x) => !x.droppedBy).map((x) => x.id),
    [0],
  );
  assert.equal(
    distribution(logits, { ...settings, topK: 1 })[0].probability,
    1,
  );
  assert.deepEqual(
    distribution([2, 2, 2], { ...settings, topK: 2 })
      .filter((x) => !x.droppedBy)
      .map((x) => x.id),
    [0, 1],
  );
  assert.deepEqual(
    distribution([1, 4, 2], { temperature: 0, topK: 1, topP: 0.1 }).map(
      (x) => x.probability,
    ),
    [1, 0, 0],
  );
});
test("temperature preserves ranking and changes concentration, stable huge logits", () => {
  const cold = distribution([2, 1, 0, -1], { ...settings, temperature: 0.5 });
  const hot = distribution([2, 1, 0, -1], { ...settings, temperature: 2 });
  assert.deepEqual(
    cold.map((x) => x.id),
    hot.map((x) => x.id),
  );
  assert.ok(cold[0].probability > hot[0].probability);
  softmax([10000, 9999, -10000]).forEach((x) => assert.ok(Number.isFinite(x)));
  for (const p of [{ temperature: -1 }, { topP: 0 }, { topK: 1.5 }])
    assert.throws(() => distribution(logits, { ...settings, ...p }));
});
test("full vocabulary affects mass; seeded draws replay and never sample removed tokens", () => {
  const full = distribution(new Float32Array(49152), { ...settings, topK: 2 });
  close(full[0].base, 1 / 49152);
  close(full[0].probability, 0.5);
  const a = mulberry32(7),
    b = mulberry32(7);
  const samples = Array.from({ length: 10000 }, () => draw(full, a).id);
  assert.deepEqual(
    samples,
    Array.from({ length: 10000 }, () => draw(full, b).id),
  );
  assert.ok(samples.every((id) => id === 0 || id === 1));
  const fraction = samples.filter((id) => id === 0).length / samples.length;
  assert.ok(Math.abs(fraction - 0.5) < 0.025);
});
