"use client";
import { useEffect, useState } from "react";
import {
  BrainCircuitIcon,
  CheckIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  DownloadIcon,
  FileSearchIcon,
  PlayIcon,
  RotateCcwIcon,
  ShieldCheckIcon,
  SquareIcon,
  CalculatorIcon,
  PauseIcon,
  PinIcon,
} from "lucide-react";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
  CardAction,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Separator } from "@/components/ui/separator";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  ControlGroup,
  SelectField,
  SliderField,
  InfoTip,
} from "@/components/lab/controls";
import { useDay2Runtime } from "@/hooks/use-day2-runtime";
import { usePrefersReducedMotion } from "@/hooks/use-animation";
import {
  downloadEvidence,
  type AgentEvent,
  type AgentSettings,
} from "@/lib/day2-protocol";
import corpus from "@/lib/data/day2-corpus.json";
import { cn } from "@/lib/utils";
import {
  EvidenceDetails,
  EvidenceInspector,
  JsonEvidence,
  StageTitle,
} from "./day2-evidence";
const DEFAULT: AgentSettings = {
  task: "What percentage of Study A participants completed the study? Cite the source.",
  mode: "live",
  model_calls: 4,
  tool_calls: 3,
  tokens: 6000,
  deadline: 120,
};
const MODES = [
  { value: "live", label: "Live model · search & calculate" },
  { value: "invalid", label: "Controller test · denominator zero" },
  { value: "forbidden", label: "Controller test · forbidden tool" },
  { value: "loop", label: "Controller test · repeated search" },
];
const STAGES = [
  { name: "Model", icon: BrainCircuitIcon, kinds: ["decision", "proposal"] },
  { name: "Guard", icon: ShieldCheckIcon, kinds: ["validated", "rejected"] },
  { name: "Tool", icon: CalculatorIcon, kinds: ["executing"] },
  { name: "Observation", icon: FileSearchIcon, kinds: ["observation"] },
  { name: "Stop", icon: CheckIcon, kinds: ["answer", "terminal"] },
];
const ORIGIN: Record<string, string> = {
  live_model: "Live model",
  scripted_test: "Controller test",
  injected_test: "Injected test",
  saved_replay: "Saved replay",
};
function eventTitle(event: AgentEvent) {
  if (event.kind === "terminal") return String(event.data.status);
  if (event.kind === "proposal") {
    const calls = event.data.calls as { name: string }[] | undefined;
    return calls?.map((c) => c.name).join(", ") || "Tool proposal";
  }
  return String(event.data.name ?? event.kind);
}
export function BoundedResearchAssistant() {
  const runtime = useDay2Runtime<AgentEvent>("agent");
  const reduced = usePrefersReducedMotion();
  const [settings, setSettings] = useState(DEFAULT);
  const [focus, setFocus] = useState<number | null>(null);
  const [inspect, setInspect] = useState<AgentEvent | null>(null);
  const [playing, setPlaying] = useState(false);
  const [baseline, setBaseline] = useState<AgentEvent[] | null>(null);
  const events = runtime.events;
  const shown = focus === null ? events : events.slice(0, focus + 1);
  const current = shown.at(-1);
  const terminal = shown.find((e) => e.kind === "terminal");
  const budgets = current?.budgets;
  const runSettings =
    (events[0]?.data.settings as AgentSettings | undefined) ?? settings;
  const origin =
    current?.origin ??
    (settings.mode === "live" ? "live_model" : "scripted_test");
  const active = STAGES.findIndex((stage) =>
    stage.kinds.includes(current?.kind ?? ""),
  );
  const update = (patch: Partial<AgentSettings>) => {
    runtime.clear();
    setSettings((s) => ({ ...s, ...patch }));
    setFocus(null);
    setInspect(null);
    setPlaying(false);
  };
  useEffect(() => {
    if (!playing) return;
    const timer = setInterval(
      () =>
        setFocus((index) => {
          const next = (index ?? 0) + 1;
          if (next >= events.length) {
            setPlaying(false);
            return events.length - 1;
          }
          return next;
        }),
      reduced ? 1600 : 1000,
    );
    return () => clearInterval(timer);
  }, [playing, events.length, reduced]);
  const reset = () => {
    runtime.clear();
    setSettings(DEFAULT);
    setFocus(null);
    setInspect(null);
    setPlaying(false);
    setBaseline(null);
  };
  const observations = shown.filter((e) => e.kind === "observation");
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid min-w-0 gap-4">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <BrainCircuitIcon className="size-4" />Agents: tool calls and limits
              <InfoTip>
                The local Qwen model selects tools; the application validates,
                executes and stops. This timeline shows actions and evidence,
                not hidden reasoning.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Badge
                variant={origin === "live_model" ? "secondary" : "outline"}
              >
                {runtime.recording ? "Saved real trace" : ORIGIN[origin]}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-6">
            <div className="relative grid grid-cols-5 gap-1 sm:gap-3">
              {STAGES.map((stage, i) => {
                const Icon = stage.icon;
                return (
                  <div
                    key={stage.name}
                    className={cn(
                      "relative grid min-w-0 justify-items-center gap-2 rounded-xl border bg-muted/20 py-5",
                      i === active &&
                        "border-primary/60 bg-primary/10 ring-1 ring-primary/30",
                    )}
                  >
                    <Icon className="size-6 sm:size-8" />
                    <span className="text-[10px] sm:text-xs">{stage.name}</span>
                    <span className="font-mono text-[10px] text-muted-foreground">
                      0{i + 1}
                    </span>
                    {i < 4 && (
                      <ChevronRightIcon className="absolute top-6 -right-2 z-10 size-4 rounded-full bg-background" />
                    )}
                  </div>
                );
              })}
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              {[
                {
                  name: "Model attempts",
                  used: budgets?.model_calls ?? 0,
                  limit: runSettings.model_calls,
                },
                {
                  name: "Tool attempts",
                  used: budgets?.tool_calls ?? 0,
                  limit: runSettings.tool_calls,
                },
                {
                  name: "Charged tokens",
                  used: budgets?.tokens ?? 0,
                  limit: runSettings.tokens,
                },
                {
                  name: "Recorded seconds",
                  used: budgets?.elapsed ?? 0,
                  limit: runSettings.deadline,
                },
              ].map((b) => (
                <div
                  className="grid gap-2 rounded-xl border bg-muted/20 p-3"
                  key={b.name}
                >
                  <span className="text-xs text-muted-foreground">
                    {b.name}
                  </span>
                  <span className="font-mono text-lg tabular-nums">
                    {Number.isInteger(b.used) ? b.used : b.used.toFixed(1)}
                    <span className="text-xs text-muted-foreground">
                      {" "}
                      / {b.limit}
                    </span>
                  </span>
                  <div className="h-1 rounded-full bg-muted">
                    <div
                      className="h-full rounded-full bg-primary"
                      style={{
                        width: `${Math.min(100, (100 * b.used) / b.limit)}%`,
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
            <section className="grid gap-3">
              <StageTitle
                number={1}
                title="Immutable execution trace"
                metric={`${shown.length} / ${events.length} events`}
                info="Each event keeps its original inputs, call ID, origin and budget counters. Click an event for raw proposal, validation, source or arithmetic. A rejected request never starts execution."
              />
              <div className="flex flex-wrap items-center gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  disabled={runtime.busy || !events.length}
                  onClick={() => {
                    setFocus(0);
                    setPlaying(true);
                  }}
                >
                  {" "}
                  <PlayIcon className="size-3" />
                  Replay recorded events
                </Button>
                {playing && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setPlaying(false)}
                  >
                    <PauseIcon className="size-3" />
                    Pause
                  </Button>
                )}
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Previous recorded event"
                  disabled={!events.length || runtime.busy}
                  onClick={() => {
                    setPlaying(false);
                    setFocus(Math.max(0, (focus ?? events.length - 1) - 1));
                  }}
                >
                  <ChevronLeftIcon className="size-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  aria-label="Next recorded event"
                  disabled={!events.length || runtime.busy}
                  onClick={() => {
                    setPlaying(false);
                    setFocus(
                      Math.min(
                        events.length - 1,
                        (focus ?? events.length - 1) + 1,
                      ),
                    );
                  }}
                >
                  <ChevronRightIcon className="size-4" />
                </Button>
                {focus !== null && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => {
                      setPlaying(false);
                      setFocus(null);
                    }}
                  >
                    Show whole run
                  </Button>
                )}
                <Badge variant="outline">
                  {focus !== null
                    ? "Recorded replay · no tools execute"
                    : runtime.recording
                      ? "Saved real trace · no tools execute"
                      : "Current run"}
                </Badge>
              </div>
              <div className="max-h-[32rem] overflow-y-auto rounded-xl border bg-muted/15 p-3">
                {shown.length ? (
                  shown.map((e) => (
                    <button
                      key={e.id}
                      type="button"
                      onClick={() => setInspect(e)}
                      className={cn(
                        "group mb-2 grid w-full grid-cols-[2rem_minmax(0,1fr)_auto] items-center gap-3 rounded-lg border bg-background/75 p-3 text-left transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                        e.kind === "rejected" && "border-destructive/50",
                        e.kind === "terminal" && "border-primary/50",
                      )}
                    >
                      <span className="font-mono text-xs text-muted-foreground">
                        {String(e.sequence).padStart(2, "0")}
                      </span>
                      <div className="grid min-w-0 gap-1">
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge
                            variant={
                              e.kind === "rejected" ? "destructive" : "outline"
                            }
                          >
                            {e.kind}
                          </Badge>
                          <span className="truncate text-sm font-medium">
                            {eventTitle(e)}
                          </span>
                        </div>
                        <span className="truncate text-xs text-muted-foreground">
                          {e.kind === "rejected"
                            ? "Denied before execution"
                            : e.kind === "terminal"
                              ? String(e.data.reason)
                              : e.call_id
                                ? `call ${e.call_id.slice(0, 8)}`
                                : ORIGIN[e.origin]}
                        </span>
                      </div>
                      <ChevronRightIcon className="size-4 text-muted-foreground" />
                    </button>
                  ))
                ) : (
                  <div className="grid min-h-52 place-content-center justify-items-center gap-3 text-muted-foreground">
                    <ShieldCheckIcon className="size-12 opacity-30" />
                    <span className="text-sm">
                      Predict search → calculate → cite
                    </span>
                  </div>
                )}
              </div>
            </section>
            {observations.length > 0 && (
              <section className="grid gap-3">
                <StageTitle
                  number={2}
                  title="Evidence returned by actual tools"
                  info="Search uses distinct query-word overlap on the fixed study corpus. The calculator divides actual arguments. Returned text is evidence; it is not permission to run another tool."
                />
                {observations.map((e) => (
                  <div
                    key={e.id}
                    className="rounded-xl border bg-primary/5 p-4"
                  >
                    <div className="flex items-center justify-between gap-3">
                      <b className="text-sm">{String(e.data.name)}</b>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setInspect(e)}
                      >
                        Inspect observation
                      </Button>
                    </div>
                    {e.data.name === "ratio" ? (
                      <div className="mt-3 flex flex-wrap items-baseline gap-4">
                        <span className="font-mono text-4xl">
                          {Number(
                            (e.data.result as { percent: number }).percent,
                          ).toFixed(1)}
                          %
                        </span>
                        <span className="font-mono text-sm text-muted-foreground">
                          {String(
                            (shown.find(
                              (v) => v.kind === "validated" && v.call_id === e.call_id,
                            )?.data.arguments as { numerator?: number } | undefined)?.numerator ?? "n",
                          )}
                          {" / "}
                          {String(
                            (shown.find(
                              (v) => v.kind === "validated" && v.call_id === e.call_id,
                            )?.data.arguments as { denominator?: number } | undefined)?.denominator ?? "d",
                          )}
                          {" = "}
                          {Number((e.data.result as { proportion: number }).proportion)}
                        </span>
                      </div>
                    ) : (
                      <div className="mt-2 grid gap-2">
                        {(
                          e.data.result as {
                            id: string;
                            text: string;
                            score: number;
                          }[]
                        ).map((r) => (
                          <div key={r.id} className="grid gap-1">
                            <code className="text-xs">
                              {r.id} · lexical score {r.score}
                            </code>
                            <p className="text-sm leading-relaxed">{r.text}</p>
                          </div>
                        ))}
                      </div>
                    )}
                  </div>
                ))}
              </section>
            )}
            {terminal && (
              <section className="grid gap-3 rounded-xl border border-primary/40 bg-primary/5 p-4">
                <div className="flex items-center gap-2">
                  <ShieldCheckIcon className="size-4" />
                  <Badge variant="outline">
                    {String(terminal.data.status)}
                  </Badge>
                </div>
                {typeof terminal.data.answer === "string" && (
                  <p className="text-sm leading-relaxed">
                    {terminal.data.answer}
                  </p>
                )}
                {observations.some((e) => e.data.name === "document_search") && (
                  <div className="flex flex-wrap items-center gap-2 text-xs">
                    <span className="text-muted-foreground">Check against observed sources:</span>
                    {observations.filter((e) => e.data.name === "document_search").flatMap((e) =>
                      (e.data.result as { id: string }[]).map((source) => (
                        <Button key={`${e.id}:${source.id}`} variant="outline" size="sm" onClick={() => setInspect(e)}>
                          <FileSearchIcon className="size-3" />
                          {source.id}
                        </Button>
                      )),
                    )}
                  </div>
                )}
                <span className="text-xs text-muted-foreground">
                  {String(terminal.data.reason)}
                </span>
              </section>
            )}
            <div
              role="status"
              aria-live="polite"
              className="text-xs text-muted-foreground"
            >
              {runtime.status ||
                "Ready · live model or clearly labelled controller test"}
            </div>
            {runtime.error && (
              <Alert variant="destructive">
                <AlertDescription>{runtime.error}</AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>
        {baseline && (
          <Card>
            <CardHeader>
              <CardTitle>Pinned trace comparison</CardTitle>
              <CardAction>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setBaseline(null)}
                >
                  Clear baseline
                </Button>
              </CardAction>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2">
              {[
                { name: "Baseline", trace: baseline },
                { name: "Current", trace: events },
              ].map((t) => (
                <div className="grid gap-2 rounded-xl border p-4" key={t.name}>
                  <b className="text-sm">{t.name}</b>
                  <JsonEvidence
                    value={{
                      settings: t.trace[0]?.data.settings,
                      terminal: t.trace.find((e) => e.kind === "terminal")?.data ?? {
                        status: "In progress",
                        latest_event: t.trace.at(-1)?.kind,
                      },
                      budgets: t.trace.at(-1)?.budgets,
                    }}
                  />
                </div>
              ))}
            </CardContent>
          </Card>
        )}
        <EvidenceDetails title="Corpus, equations & model provenance">
          <p>
            <b>Predict:</b> Who runs the function? <b>Explain:</b> Which event
            proves it ran? What stopped a fifth model request? Could valid JSON
            still be forbidden?
          </p>
          {corpus.studies.map((d) => (
            <div className="rounded-lg border p-3" key={d.id}>
              <code>{d.id}</code>
              <p className="mt-2">{d.text}</p>
            </div>
          ))}
          <p>
            <code>
              aₜ = πθ(hₜ); oₜ = E(aₜ; permissions, budget); hₜ₊₁ = hₜ ⊕ (aₜ,oₜ)
            </code>
            . The model proposes action a from history h; executor E returns
            observation o only after checks. Terminal branches stop without
            another model call. <code>p = n/d</code>: completed n divided by
            enrolled d, 18/24 = 0.75 = 75%. Counts must be finite with 0 ≤ n ≤ d
            and d &gt; 0.
          </p>
          <p>
            Actual local Qwen2.5-1.5B-Instruct, greedy decoding, native tool
            template, LangChain chat-model and StructuredTool interfaces. The
            application owns this serial loop, allowlist and budgets. Tokens
            count the whole formatted prompt plus actual generated IDs at every
            request; before a call the full prompt + 160 output tokens must fit.
            Classroom context cap 4096. No paid API or remote inference. Live
            model errors remain visible.
          </p>
          <p>
            <a
              className="underline"
              href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"
            >
              Qwen model card · Apache-2.0
            </a>{" "}
            ·{" "}
            <a
              className="underline"
              href="https://docs.langchain.com/oss/python/langchain/tools"
            >
              LangChain tools
            </a>
          </p>
          <p>
            Controller tests use deterministic proposals with their own origin
            label, no language model. Search is scoped to these three synthetic
            documents. No tool can send messages, modify files or select a
            filesystem path. Replay displays recorded events; it does not
            execute work.
          </p>
          <p>
            Counters show the selected event&apos;s recorded usage, not an estimated
            live clock. Float32 CPU inference needs several GB of RAM; preload
            once and start the local service using the lab README. The saved
            trace preserves a witnessed run for slower classroom computers.
          </p>
        </EvidenceDetails>
      </div>
      <Card className="order-first xl:order-last">
        <CardHeader>
          <CardTitle>Run controls</CardTitle>
          <CardAction>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Reset agent experiment"
              onClick={reset}
            >
              <RotateCcwIcon className="size-4" />
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-5">
          <ControlGroup label="01 · Task">
            <SelectField
              label="Study question"
              value={settings.task}
              options={["A", "B", "C"].map((study) => ({
                value: `What percentage of Study ${study} participants completed the study? Cite the source.`,
                label: `Study ${study}${study === "C" ? " · missing count" : ""}`,
              }))}
              onValueChange={(task) => update({ task })}
            />
            <Label htmlFor="agent-task">Edit task</Label>
            <Textarea
              rows={3}
              className="resize-none"
              id="agent-task"
              value={settings.task}
              maxLength={300}
              onChange={(e) => update({ task: e.target.value })}
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="02 · Execution mode">
            <SelectField
              label="Scenario"
              value={settings.mode}
              options={MODES}
              onValueChange={(mode) => update({ mode })}
              info="Live mode uses actual model decisions. Controller tests inject deterministic proposals to test enforcement; they are never presented as model outputs."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="03 · Enforced budgets">
            <SliderField
              label="Model-call cap"
              value={settings.model_calls}
              onValueChange={(model_calls) => update({ model_calls })}
              min={1}
              max={6}
              info="Gate checked before requesting model tokens. Loop tests simulate model attempts without charging invented tokens."
            />
            <SliderField
              label="Tool-attempt cap"
              value={settings.tool_calls}
              onValueChange={(tool_calls) => update({ tool_calls })}
              min={1}
              max={8}
              info="Rejected attempts count too. Increase this above the model cap to isolate the model-call gate."
            />
            <SliderField
              label="Token budget"
              value={settings.tokens}
              onValueChange={(tokens) => update({ tokens })}
              min={100}
              max={12000}
              step={100}
              info="Sum of full formatted input + actual output tokens across requests. The next input and maximum output are reserved before a call."
            />
            <SliderField
              label="Deadline · seconds"
              value={settings.deadline}
              onValueChange={(deadline) => update({ deadline })}
              min={1}
              max={180}
              info="Wall time includes loading and inference. Deadline/cancel checks occur between generated tokens and at dispatch boundaries."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="04 · Execute & inspect">
            <Button
              variant="outline"
              disabled={runtime.busy}
              onClick={() => {
                setFocus(null);
                setPlaying(false);
                runtime.loadRecording("/traces/day2-study-a.json");
              }}
            >
              Load saved real trace
            </Button>
            <Button
              disabled={runtime.busy || !settings.task.trim()}
              onClick={() => {
                setFocus(null);
                setPlaying(false);
                runtime.run(settings);
              }}
            >
              <PlayIcon className="size-4" />
              {settings.mode === "live"
                ? "Run live agent"
                : "Run controller test"}
            </Button>
            {runtime.busy && (
              <Button variant="outline" onClick={runtime.cancel}>
                <SquareIcon className="size-4" />
                Cancel run
              </Button>
            )}
            <Button
              variant="outline"
              disabled={runtime.busy || !events.length}
              onClick={() => setBaseline(structuredClone(events))}
            >
              <PinIcon className="size-4" />
              Pin trace
            </Button>
            <Button
              variant="ghost"
              disabled={!events.length}
              onClick={() =>
                downloadEvidence("bounded-agent-trace.json", {
                  events,
                  baseline,
                  client_status: runtime.status,
                })
              }
            >
              <DownloadIcon className="size-4" />
              Export trace
            </Button>
          </ControlGroup>
        </CardContent>
      </Card>
      <EvidenceInspector
        title={
          inspect
            ? `${inspect.sequence} · ${inspect.kind} · ${eventTitle(inspect)}`
            : "Event"
        }
        description="Original event, immutable call linkage, raw evidence and budget counters."
        open={!!inspect}
        onClose={() => setInspect(null)}
      >
        {inspect && (
          <>
            <div className="flex flex-wrap gap-2">
              <Badge>{ORIGIN[inspect.origin]}</Badge>
              <Badge variant="outline">
                {inspect.budgets.elapsed.toFixed(2)}s
              </Badge>
              {inspect.call_id && (
                <code className="text-xs break-all">{inspect.call_id}</code>
              )}
            </div>
            <JsonEvidence value={inspect} />
          </>
        )}
      </EvidenceInspector>
    </div>
  );
}
