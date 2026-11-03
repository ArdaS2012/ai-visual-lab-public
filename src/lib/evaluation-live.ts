import benchmark from "./data/evaluation-benchmark.json" with { type: "json" };
export { benchmark };
export type BenchmarkCase = (typeof benchmark.cases)[number];
export type EvaluationSettings = {
  evidence: "full" | "missing";
  format: "json" | "text";
};
export type ModelAnswer = {
  run_id: string;
  case_id: string;
  variant: "A" | "B";
  raw: string;
  supplied_sources: string[];
  messages: { role: string; content: string }[];
  model: string;
  revision: string;
  prompt_tokens: number;
  output_tokens: number;
  seconds: number;
  truncated: boolean;
};
export type JudgeOutput = {
  raw: string;
  model: string;
  revision: string;
  messages: { role: string; content: string }[];
  truncated: boolean;
};
export type EvaluationStart = {
  run_id: string;
  settings: EvaluationSettings;
  benchmark: typeof benchmark;
  benchmark_hash: string;
  started_at: string;
  decoding: string;
  max_new_tokens: number;
};
export type EvaluationEvent =
  | { kind: "evaluation_start"; data: EvaluationStart }
  | { kind: "evaluation_answer"; data: ModelAnswer }
  | { kind: "evaluation_judge"; data: JudgeOutput }
  | { kind: "loading" | "terminal" | "error"; data: { reason: string } };
const normalize = (s: string) => s.toLowerCase().trim().replace(/\s+/g, " ");
export function scoreAnswer(row: BenchmarkCase, result: ModelAnswer) {
  let answer = result.raw;
  let schema = false;
  let sources: string[] = [];
  let contentExtraction = "Plain text";
  try {
    const obj = JSON.parse(result.raw);
    schema =
      !!obj &&
      typeof obj === "object" &&
      !Array.isArray(obj) &&
      Object.keys(obj).sort().join(",") === "answer,sources" &&
      typeof obj.answer === "string" &&
      Array.isArray(obj.sources) &&
      obj.sources.every((id: unknown) => typeof id === "string");
  } catch {
    /* Strict JSON schema checks the unmodified response only. */
  }
  // Content extraction is declared independently of the consumer's JSON contract.
  const fenced = result.raw.trim().match(/^```(?:json)?\s*([\s\S]*?)\s*```$/i);
  try {
    const obj = JSON.parse(fenced?.[1] ?? result.raw);
    const content =
      obj && typeof obj === "object" && !Array.isArray(obj) ? obj.answer : obj;
    if (typeof content === "string" || typeof content === "number") {
      answer = String(content);
      contentExtraction = fenced
        ? "Content read from fenced JSON; raw schema still checked"
        : "Content read from JSON";
    }
    if (
      obj &&
      typeof obj === "object" &&
      Array.isArray(obj.sources) &&
      obj.sources.every((id: unknown) => typeof id === "string")
    )
      sources = obj.sources;
  } catch {
    /* No repair or model-based interpretation of malformed outputs. */
  }
  return {
    answer,
    contentExtraction,
    schema: schema && !result.truncated,
    exact:
      !result.truncated &&
      row.aliases.some((ref) => normalize(ref) === normalize(answer)),
    sources,
    unknownSourceIds: sources.filter(
      (id) => !result.supplied_sources.includes(id),
    ),
  };
}
export function pairedResults(answers: ModelAnswer[], cases: BenchmarkCase[]) {
  const ids = cases
    .filter((c) =>
      ["A", "B"].every((v) =>
        answers.some((a) => a.case_id === c.id && a.variant === v),
      ),
    )
    .map((c) => c.id);
  return {
    n: ids.length,
    variants: (["A", "B"] as const).map((variant) => {
      const rows = answers.filter(
        (a) => a.variant === variant && ids.includes(a.case_id),
      );
      const scores = rows.map((a) =>
        scoreAnswer(cases.find((c) => c.id === a.case_id)!, a),
      );
      return {
        variant,
        exact: scores.filter((s) => s.exact).length,
        schema: scores.filter((s) => s.schema).length,
      };
    }),
  };
}
export function parseJudge(value: JudgeOutput) {
  if (value.truncated) return null;
  try {
    const obj = JSON.parse(value.raw);
    return obj &&
      typeof obj === "object" &&
      !Array.isArray(obj) &&
      Object.keys(obj).sort().join(",") === "rating,reason" &&
      Number.isInteger(obj.rating) &&
      obj.rating >= 0 &&
      obj.rating <= 2 &&
      typeof obj.reason === "string"
      ? { rating: obj.rating as number, reason: obj.reason as string }
      : null;
  } catch {
    return null;
  }
}
