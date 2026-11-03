import test from "node:test";
import assert from "node:assert/strict";
import { compareVectors } from "../src/lib/retrieval-geometry.ts";

const close = (a: number, b: number) =>
  assert.ok(Math.abs(a - b) < 1e-10, `${a} ≠ ${b}`);
test("cosine geometry distinguishes matching, perpendicular and opposite directions", () => {
  close(compareVectors([1, 0], [3, 4]).cosine, 0.6);
  close(compareVectors([1, 0], [3, 4]).angle, 53.13010235415598);
  close(compareVectors([1, 0], [0, 2]).angle, 90);
  close(compareVectors([1, 0], [-2, 0]).angle, 180);
  close(compareVectors([1, 0], [9, 0]).angle, 0);
});
test("all coordinates contribute with their signs, regardless of vector scale", () => {
  const q = Array.from({ length: 384 }, (_, i) => (i % 2 ? -1 : 2));
  const d = Array.from({ length: 384 }, (_, i) => (i % 3 ? 3 : -2));
  const result = compareVectors(q, d);
  const expected =
    q.reduce((sum, value, i) => sum + value * d[i], 0) /
    (Math.hypot(...q) * Math.hypot(...d));
  close(result.cosine, expected);
  close(
    result.contributions.reduce((a, b) => a + b, 0),
    expected,
  );
  close(
    compareVectors(
      q.map((v) => v * 7),
      d.map((v) => v * 0.2),
    ).cosine,
    expected,
  );
  assert.ok(result.contributions.some((v) => v < 0));
  assert.ok(result.contributions.some((v) => v > 0));
});
test("invalid vectors cannot produce a plausible similarity graphic", () => {
  for (const [q, d] of [
    [[], []],
    [[1], [1, 2]],
    [[0], [1]],
    [[NaN], [1]],
    [[1], [Infinity]],
  ])
    assert.throws(() => compareVectors(q, d));
});
