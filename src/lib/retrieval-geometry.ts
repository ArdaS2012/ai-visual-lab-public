/** Full-dimensional geometry; the pair drawing must never supply search scores. */
export function compareVectors(
  query: readonly number[],
  passage: readonly number[],
) {
  if (!query.length || query.length !== passage.length)
    throw new Error("Vectors must have the same nonzero dimension.");
  if (![...query, ...passage].every(Number.isFinite))
    throw new Error("Vector coordinates must be finite.");
  const queryNorm = Math.hypot(...query);
  const passageNorm = Math.hypot(...passage);
  if (!queryNorm || !passageNorm)
    throw new Error("Cosine needs nonzero vectors.");
  const denominator = queryNorm * passageNorm;
  const contributions = query.map((q, i) => (q * passage[i]) / denominator);
  const cosine = Math.max(
    -1,
    Math.min(
      1,
      contributions.reduce((a, b) => a + b, 0),
    ),
  );
  return {
    cosine,
    angle: (Math.acos(cosine) * 180) / Math.PI,
    contributions,
    queryNorm,
    passageNorm,
  };
}
