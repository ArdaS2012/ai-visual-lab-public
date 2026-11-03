export const MODEL_ID = "HuggingFaceTB/SmolLM2-360M-Instruct";
export const MODEL_NAME = "SmolLM2-360M-Instruct";
export const MODEL_REVISION = "a10cc1512eabd3dde888204e902eca88bddb4951";
export const VOCABULARY_SIZE = 49152;
export const CONTEXT_LIMIT = 128; // Includes chat formatting; model supports 8,192.
export const EOS_ID = 2; // <|im_end|>: the model's end-of-answer token.

/** Cache reuse is valid only for an exact, one-token continuation. */
export function isDirectExtension(previous: number[], next: number[]): boolean {
  return (
    previous.length > 0 &&
    next.length === previous.length + 1 &&
    previous.every((id, i) => id === next[i])
  );
}
export type ModelSnapshot = {
  ids: number[];
  pieces: string[];
  text: string;
  context: string;
  logits: Float32Array;
  milliseconds: number;
};
export type WorkerRequest =
  | { type: "load"; requestId: number }
  | { type: "predict"; requestId: number; prompt: string }
  | { type: "append"; requestId: number; ids: number[] };
export type WorkerReply =
  | { type: "progress"; message: string; progress?: number }
  | { type: "ready"; requestId: number; vocabulary: string[] }
  | { type: "snapshot"; requestId: number; snapshot: ModelSnapshot }
  | { type: "error"; requestId: number; message: string };
