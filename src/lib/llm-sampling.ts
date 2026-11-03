/** Sampling acts on the entire vocabulary, never just the displayed rows. */
export type SamplingSettings = {
  temperature: number;
  topK: number;
  topP: number;
};
export type Candidate = {
  id: number;
  logit: number;
  scaled: number;
  base: number;
  tempered: number;
  afterK: number;
  cumulative: number;
  probability: number;
  droppedBy: "top-k" | "top-p" | "greedy" | null;
};

export function softmax(values: ArrayLike<number>): number[] {
  if (!values.length) throw new Error("Empty logit vector");
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) throw new Error("Non-finite model logit");
    max = Math.max(max, values[i]);
  }
  const exps = Array.from(values, (x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sum);
}

/** Temperature → top-k → renormalize → top-p crossing prefix → renormalize.
 * topK=0 disables k. Equal logits are ordered by token ID for reproducibility.
 * T=0 is a separate greedy policy, not division by zero.
 */
export function distribution(
  logits: ArrayLike<number>,
  settings: SamplingSettings,
): Candidate[] {
  const { temperature, topK, topP } = settings;
  if (
    !Number.isFinite(temperature) ||
    temperature < 0 ||
    !Number.isInteger(topK) ||
    topK < 0 ||
    !(topP > 0 && topP <= 1)
  ) {
    throw new Error("Invalid sampling settings");
  }
  const base = softmax(logits);
  const scaled = Array.from(logits, (x) =>
    temperature === 0 ? x : x / temperature,
  );
  const tempered = temperature === 0 ? base : softmax(scaled);
  const ranked: Candidate[] = Array.from(logits, (logit, id) => ({
    id,
    logit,
    scaled: scaled[id],
    base: base[id],
    tempered: tempered[id],
    afterK: 0,
    cumulative: 0,
    probability: 0,
    droppedBy: null,
  })).sort((a, b) => b.logit - a.logit || a.id - b.id);
  if (temperature === 0) {
    ranked.forEach((row, i) => {
      row.afterK = row.probability = i === 0 ? 1 : 0;
      row.cumulative = 1;
      row.droppedBy = i === 0 ? null : "greedy";
    });
    return ranked;
  }
  const k = topK === 0 ? ranked.length : Math.min(topK, ranked.length);
  const massK = ranked.slice(0, k).reduce((sum, row) => sum + row.tempered, 0);
  let massP = 0;
  let closed = false;
  ranked.forEach((row, i) => {
    row.afterK = i < k ? row.tempered / massK : 0;
    if (i >= k) row.droppedBy = "top-k";
    else if (closed) row.droppedBy = "top-p";
    else {
      massP += row.afterK;
      // Include the crossing candidate. Exact equality closes the prefix.
      if (topP < 1 && massP >= topP - 1e-12) closed = true;
    }
    row.cumulative = massP;
  });
  ranked.forEach((row) => {
    row.probability = row.droppedBy ? 0 : row.afterK / massP;
  });
  return ranked;
}

export function mulberry32(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function draw(rows: Candidate[], random: () => number): Candidate {
  const roll = random();
  let mass = 0;
  let last = rows[0];
  for (const row of rows) {
    if (row.probability <= 0) continue;
    last = row;
    mass += row.probability;
    if (roll < mass) return row;
  }
  return last;
}

export function tokenLabel(piece: string): string {
  return (
    piece.replaceAll(" ", "·").replaceAll("\n", "↵").replaceAll("\t", "⇥") ||
    "∅"
  );
}
