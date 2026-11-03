"use client";
import { useCallback, useEffect, useRef, useState } from "react";
import type { WireEvent } from "@/lib/day2-protocol";

export function useDay2Runtime<T extends WireEvent>(
  experiment: "retrieval" | "agent" | "evaluation" | "evaluation-judge",
) {
  const [events, setEvents] = useState<T[]>([]);
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState("");
  const [status, setStatus] = useState("");
  const controller = useRef<AbortController | null>(null);
  const serial = useRef(0);
  const cancel = useCallback(() => {
    serial.current++;
    controller.current?.abort();
    controller.current = null;
    setBusy(false);
    setStatus("Cancelled — partial observations retained");
  }, []);
  const clear = useCallback(() => {
    cancel();
    setEvents([]);
    setRecording(false);
    setError("");
    setStatus("");
  }, [cancel]);
  useEffect(
    () => () => {
      serial.current++;
      controller.current?.abort();
    },
    [],
  );
  const run = useCallback(
    async (settings: unknown) => {
      controller.current?.abort();
      const version = ++serial.current;
      const abort = new AbortController();
      controller.current = abort;
      setEvents([]);
      setRecording(false);
      setError("");
      setStatus("Starting local model experiment…");
      setBusy(true);
      const received: T[] = [];
      try {
        const response = await fetch(`/api/day2/${experiment}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(settings),
          signal: abort.signal,
        });
        if (!response.ok) {
          const value = await response.json();
          throw new Error(
            typeof value.error === "string"
              ? value.error
              : typeof value.detail === "string"
                ? value.detail
                : JSON.stringify(value.detail),
          );
        }
        if (!response.body) throw new Error("Missing model response stream.");
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let pending = "";
        const accept = (line: string) => {
          if (!line.trim() || serial.current !== version) return;
          const event = JSON.parse(line) as T;
          if (event.kind === "error")
            throw new Error(String(event.data.reason));
          received.push(event);
          setEvents((old) => [...old, event]);
          setStatus(
            event.kind === "loading"
              ? String(event.data.reason)
              : event.kind === "terminal"
                ? String(event.data.reason)
                : event.kind === "retrieval"
                  ? "Measured run complete"
                  : "Running · " + event.kind,
          );
        };
        while (true) {
          const { value, done } = await reader.read();
          pending += decoder.decode(value, { stream: !done });
          const lines = pending.split("\n");
          pending = lines.pop() ?? "";
          lines.forEach(accept);
          if (done) break;
        }
        accept(pending);
        if (serial.current === version) return received;
      } catch (e) {
        if (serial.current === version && !abort.signal.aborted) {
          setError(e instanceof Error ? e.message : String(e));
          setStatus("Experiment failed");
        }
      } finally {
        if (serial.current === version) {
          setBusy(false);
          controller.current = null;
        }
      }
    },
    [experiment],
  );
  const loadRecording = useCallback(async (path: string) => {
    controller.current?.abort();
    const version = ++serial.current;
    const abort = new AbortController();
    controller.current = abort;
    setBusy(true);
    setError("");
    setEvents([]);
    setRecording(false);
    try {
      const response = await fetch(path, { signal: abort.signal });
      if (!response.ok) throw new Error("Saved trace could not be loaded.");
      const value = (await response.json()) as { events: T[] };
      if (serial.current !== version) return;
      setEvents(value.events);
      setRecording(true);
      setStatus("Saved real trace loaded · no model or tool work");
    } catch (e) {
      if (serial.current === version && !abort.signal.aborted)
        setError(e instanceof Error ? e.message : String(e));
    } finally {
      if (serial.current === version) {
        setBusy(false);
        controller.current = null;
      }
    }
  }, []);
  return {
    events,
    busy,
    error,
    status,
    recording,
    run,
    cancel,
    clear,
    loadRecording,
  };
}
