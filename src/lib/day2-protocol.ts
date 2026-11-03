export type RetrievalSettings = {
  query: string;
  size: number;
  overlap: number;
  candidates: number;
  selected: number;
};
export type Chunk = {
  id: string;
  source: string;
  start: number;
  end: number;
  text: string;
  vector: number[];
  model_tokens: number;
  cosine: number;
};
export type RankedChunk = Chunk & {
  initial_rank: number;
  final_rank: number;
  logit: number;
  pair_tokens: number;
  selected: boolean;
};
export type RetrievalResult = {
  settings: RetrievalSettings;
  corpus_version: string;
  models: Record<string, string>;
  query_vector: number[];
  query_tokens: number;
  chunks: Chunk[];
  candidates: RankedChunk[];
  seconds: number;
};
export type AgentSettings = {
  task: string;
  mode: string;
  model_calls: number;
  tool_calls: number;
  tokens: number;
  deadline: number;
};
export type AgentEvent = {
  id: string;
  run_id: string;
  sequence: number;
  kind: string;
  origin: string;
  call_id?: string;
  data: Record<string, unknown>;
  budgets: {
    model_calls: number;
    tool_calls: number;
    tokens: number;
    elapsed: number;
  };
};
export type WireEvent = { kind: string; data: Record<string, unknown> };
export function downloadEvidence(name: string, data: unknown) {
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }),
  );
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
