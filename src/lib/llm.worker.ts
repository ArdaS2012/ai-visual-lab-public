import {
  AutoTokenizer,
  AutoModelForCausalLM,
  Tensor,
  env,
} from "@huggingface/transformers";
import {
  CONTEXT_LIMIT,
  MODEL_ID,
  MODEL_NAME,
  MODEL_REVISION,
  VOCABULARY_SIZE,
  isDirectExtension,
  type WorkerRequest,
  type WorkerReply,
} from "./llm-protocol";

env.allowLocalModels = false;
// No cross-origin isolation or GPU required. Inference stays off the UI thread.
env.backends.onnx.wasm!.numThreads = 1;
env.backends.onnx.wasm!.proxy = false;
let tokenizer:
  | Awaited<ReturnType<typeof AutoTokenizer.from_pretrained>>
  | undefined;
let model:
  | Awaited<ReturnType<typeof AutoModelForCausalLM.from_pretrained>>
  | undefined;
let prompt = "";
let responseStart = 0;
let cachedIds: number[] = [];
let cache: Record<string, Tensor> = {};
function clearCache() {
  Object.values(cache).forEach((tensor) => tensor.dispose());
  cache = {};
  cachedIds = [];
}
const reply = (data: WorkerReply, transfer: Transferable[] = []) =>
  self.postMessage(data, { transfer });

self.onmessage = async (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  try {
    if (request.type === "load") {
      const options = {
        revision: MODEL_REVISION,
        progress_callback: (p: {
          status: string;
          file?: string;
          progress?: number;
        }) => {
          reply({
            type: "progress",
            message:
              p.status === "progress"
                ? `Downloading ${p.file?.split("/").pop()}`
                : `Preparing ${MODEL_NAME}`,
            progress: p.progress,
          });
        },
      };
      tokenizer ??= await AutoTokenizer.from_pretrained(MODEL_ID, options);
      model ??= await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
        ...options,
        dtype: "q8",
        device: "wasm",
      });
      const vocabulary = Array.from({ length: VOCABULARY_SIZE }, (_, id) =>
        tokenizer!.decode([id], { skip_special_tokens: false }),
      );
      reply({ type: "ready", requestId: request.requestId, vocabulary });
      return;
    }
    if (!tokenizer || !model) throw new Error("Load the model first");
    if (request.type === "predict" && !request.prompt.trim())
      throw new Error("Enter a prompt first");
    // Use this instruction model's own template, including its default system
    // message. These are real context tokens, visible in the Context stage.
    const ids =
      request.type === "predict"
        ? (tokenizer.apply_chat_template(
            [{ role: "user", content: request.prompt }],
            {
              tokenize: true,
              return_tensor: false,
              return_dict: false,
              add_generation_prompt: true,
            },
          ) as number[])
        : request.ids;
    if (!ids.length) throw new Error("Enter a prompt first");
    if (ids.length > CONTEXT_LIMIT)
      throw new Error(
        `This lab accepts up to ${CONTEXT_LIMIT} tokens including conversation formatting. Shorten the prompt or reset the continuation.`,
      );
    if (request.type === "predict") {
      prompt = request.prompt;
      responseStart = ids.length;
    }
    const reuseCache =
      request.type === "append" &&
      isDirectExtension(cachedIds, ids) &&
      Object.keys(cache).length > 0;
    if (!reuseCache) clearCache();
    const forwardIds = reuseCache ? ids.slice(-1) : ids;
    const inputs = {
      input_ids: new Tensor("int64", BigInt64Array.from(forwardIds, BigInt), [
        1,
        forwardIds.length,
      ]),
      attention_mask: new Tensor(
        "int64",
        new BigInt64Array(ids.length).fill(BigInt(1)),
        [1, ids.length],
      ),
    };
    const start = performance.now();
    let output: Record<string, unknown> | undefined;
    let logits: Float32Array;
    try {
      const result = (await model({
        ...inputs,
        ...(reuseCache ? { past_key_values: cache } : {}),
      })) as Record<string, unknown>;
      output = result;
      const tensor = result.logits as Tensor;
      const size = tensor.dims[tensor.dims.length - 1];
      const data = tensor.data as Float32Array;
      if (size !== VOCABULARY_SIZE)
        throw new Error("Unexpected model vocabulary size");
      logits = new Float32Array(data.slice(data.length - size));
      const nextCache = model.getPastKeyValues(result, null) as Record<
        string, Tensor
      >;
      clearCache();
      cache = nextCache;
      cachedIds = [...ids];
    } catch (error) {
      clearCache();
      throw error;
    } finally {
      // Keep only the latest sequence's KV cache. Historical snapshots own
      // copied logits, never cache tensors. New prompts/resets invalidate reuse.
      const retained = new Set(Object.values(cache));
      for (const value of Object.values(output ?? {})) {
        if (value instanceof Tensor && !retained.has(value)) value.dispose();
      }
      inputs.input_ids.dispose();
      inputs.attention_mask.dispose();
    }
    const milliseconds = performance.now() - start;
    reply(
      {
        type: "snapshot",
        requestId: request.requestId,
        snapshot: {
          ids,
          pieces: ids.map((id) =>
            tokenizer!.decode([id], { skip_special_tokens: false }),
          ),
          text: `${prompt}\n\n${ids.length > responseStart ? tokenizer.decode(ids.slice(responseStart), { skip_special_tokens: true }) : ""}`,
          context: tokenizer.decode(ids, { skip_special_tokens: false }),
          logits,
          milliseconds,
        },
      },
      [logits.buffer],
    );
  } catch (error) {
    reply({
      type: "error",
      requestId: request.requestId,
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
