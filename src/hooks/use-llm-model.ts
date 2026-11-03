"use client";

import * as React from "react";
import type {
  ModelSnapshot,
  WorkerReply,
  WorkerRequest,
} from "@/lib/llm-protocol";

export function useLlmModel() {
  const worker = React.useRef<Worker | null>(null);
  const serial = React.useRef(0);
  const pending = React.useRef(
    new Map<
      number,
      { resolve: (value: WorkerReply) => void; reject: (error: Error) => void }
    >(),
  );
  const [status, setStatus] = React.useState<"idle" | "loading" | "ready">(
    "idle",
  );
  const [progress, setProgress] = React.useState<{
    message: string;
    progress?: number;
  }>({ message: "" });
  const [vocabulary, setVocabulary] = React.useState<string[]>([]);

  React.useEffect(
    () => () => {
      worker.current?.terminate();
      worker.current = null;
      pending.current.forEach((p) =>
        p.reject(new Error("Model worker closed")),
      );
      pending.current.clear();
    },
    [],
  );

  function getWorker() {
    if (!worker.current) {
      const instance = new Worker(
        new URL("../lib/llm.worker.ts", import.meta.url),
        { type: "module" },
      );
      instance.onmessage = (event: MessageEvent<WorkerReply>) => {
        const data = event.data;
        if (data.type === "progress") {
          setProgress(data);
          return;
        }
        const request = pending.current.get(data.requestId);
        pending.current.delete(data.requestId);
        if (data.type === "error") request?.reject(new Error(data.message));
        else request?.resolve(data);
      };
      instance.onerror = (event) => {
        pending.current.forEach((p) =>
          p.reject(new Error(event.message || "Model worker failed to start")),
        );
        pending.current.clear();
        instance.terminate();
        worker.current = null;
        setStatus("idle");
      };
      worker.current = instance;
    }
    return worker.current;
  }

  function send(
    request:
      | Omit<Extract<WorkerRequest, { type: "load" }>, "requestId">
      | Omit<Extract<WorkerRequest, { type: "predict" }>, "requestId">
      | Omit<Extract<WorkerRequest, { type: "append" }>, "requestId">,
  ) {
    const instance = getWorker();
    const requestId = ++serial.current;
    return new Promise<WorkerReply>((resolve, reject) => {
      pending.current.set(requestId, { resolve, reject });
      instance.postMessage({ ...request, requestId });
    });
  }

  async function load() {
    setStatus("loading");
    try {
      const result = await send({ type: "load" });
      if (result.type !== "ready") throw new Error("Unexpected model response");
      setVocabulary(result.vocabulary);
      setStatus("ready");
    } catch (error) {
      setStatus("idle");
      throw error;
    }
  }

  async function predict(input: string | number[]): Promise<ModelSnapshot> {
    const result = await send(
      typeof input === "string"
        ? { type: "predict", prompt: input }
        : { type: "append", ids: input },
    );
    if (result.type !== "snapshot")
      throw new Error("Unexpected prediction response");
    return result.snapshot;
  }

  function unload() {
    worker.current?.terminate();
    worker.current = null;
    pending.current.forEach((p) => p.reject(new Error("Model unloaded")));
    pending.current.clear();
    setVocabulary([]);
    setProgress({ message: "" });
    setStatus("idle");
  }

  return { status, progress, vocabulary, load, predict, unload };
}
