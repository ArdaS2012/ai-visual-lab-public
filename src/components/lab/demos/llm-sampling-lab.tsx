"use client";

import * as React from "react";
import { animate } from "animejs";
import {
  ArrowRightIcon,
  ChevronDownIcon,
  DicesIcon,
  DownloadIcon,
  LayersIcon,
  ListOrderedIcon,
  MessagesSquareIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
  ScanSearchIcon,
  StepForwardIcon,
} from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Separator } from "@/components/ui/separator";
import {
  ControlGroup,
  InfoTip,
  SelectField,
  SliderField,
} from "@/components/lab/controls";
import { useLlmModel } from "@/hooks/use-llm-model";
import {
  useIsomorphicLayoutEffect,
  usePrefersReducedMotion,
} from "@/hooks/use-animation";
import {
  CONTEXT_LIMIT,
  EOS_ID,
  MODEL_ID,
  MODEL_NAME,
  VOCABULARY_SIZE,
  type ModelSnapshot,
} from "@/lib/llm-protocol";
import {
  distribution,
  draw,
  mulberry32,
  tokenLabel,
  type Candidate,
  type SamplingSettings,
} from "@/lib/llm-sampling";
import { cn } from "@/lib/utils";

const PROMPTS = [
  { value: "What is the capital of France?", label: "A factual question" },
  {
    value: "Write the opening sentence of a story set in a forest.",
    label: "A story opening",
  },
  {
    value: "Complete this sentence: The scientist opened the door and saw",
    label: "An open-ended scene",
  },
];
const DEFAULTS: SamplingSettings = { temperature: 1, topK: 0, topP: 1 };
type Trace = {
  snapshot: ModelSnapshot;
  settings: SamplingSettings;
  row: Candidate;
  roll: number;
};
type Reference = {
  rows: Candidate[];
  settings: SamplingSettings;
  counts: Record<number, number>;
  total: number;
};
const pct = (value: number) => `${(value * 100).toFixed(2)}%`;
const settingsLabel = (s: SamplingSettings) =>
  `T ${s.temperature.toFixed(2)} · k ${s.topK || "all"} · p ${s.topP.toFixed(2)}`;

function Stage({
  n,
  title,
  metric,
  info,
  children,
}: {
  n: number;
  title: string;
  metric?: string;
  info: string;
  children: React.ReactNode;
}) {
  return (
    <section className="grid min-w-0 content-start gap-3">
      <header className="flex flex-wrap items-center justify-between gap-2 text-sm font-medium">
        <span className="flex items-center gap-2">
          <Badge variant="outline">{n}</Badge>
          {title}
          <InfoTip>{info}</InfoTip>
        </span>
        <span className="font-mono text-xs font-normal text-muted-foreground">
          {metric}
        </span>
      </header>
      {children}
    </section>
  );
}

/** A navigable schematic around real input/output, never invented activations. */
function TransformerComputation({
  snapshot,
  rows,
  piece,
  inspect,
}: {
  snapshot: ModelSnapshot | null;
  rows: Candidate[];
  piece: (id: number) => string;
  inspect: (row: Candidate) => void;
}) {
  const [selected, setSelected] = React.useState(2);
  const panelId = React.useId();
  const steps = [
    { label: "Use context", icon: MessagesSquareIcon },
    { label: "Refine ×32", icon: LayersIcon },
    { label: "Score tokens", icon: ListOrderedIcon },
  ];

  return (
    <div className="grid gap-3 rounded-lg border bg-muted/20 p-3">
      <div className="flex items-center gap-1" aria-label="Explore the Transformer computation">
        {steps.map(({ label, icon: Icon }, index) => (
          <React.Fragment key={label}>
            {index > 0 && <ArrowRightIcon aria-hidden="true" className="size-3 shrink-0 text-muted-foreground" />}
            <button
              type="button"
              aria-pressed={selected === index}
              aria-controls={panelId}
              onClick={() => setSelected(index)}
              className={cn(
                "grid min-w-0 flex-1 justify-items-center gap-2 rounded-md border p-2 text-center text-[11px] font-medium hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                selected === index
                  ? "border-foreground bg-background ring-1 ring-foreground"
                  : "border-transparent text-muted-foreground",
              )}
            >
              <Icon aria-hidden="true" className="size-4" />
              <span>{label}</span>
            </button>
          </React.Fragment>
        ))}
      </div>
      <div id={panelId} role="region" aria-label={steps[selected].label} aria-live="polite" className="grid min-h-36 content-start gap-3 border-t pt-3">
        {selected === 0 && (
          <>
            <div className="flex items-center justify-between gap-2 text-xs font-medium">
              Context in → predict the next position
              <InfoTip>Token IDs select learned embeddings. Causal attention lets each position use itself and earlier positions. The next-token prediction uses the final position after processing. This preview shows the user prompt and reply so far; stage 1 includes the system message and chat formatting. No attention weights are shown.</InfoTip>
            </div>
            <div className="flex items-center gap-2 text-xs" aria-label="Readable context and the next unknown token">
              {snapshot ? (
                <>
                  <span className="min-w-0 flex-1 rounded border bg-background p-2"><span className="line-clamp-3 whitespace-pre-wrap">{snapshot.text.trim()}</span></span>
                  <ArrowRightIcon aria-hidden="true" className="size-3 shrink-0" />
                  <span className="rounded border border-dashed border-foreground px-2 py-1" aria-label="Next token is not selected yet">?</span>
                </>
              ) : <span className="text-muted-foreground">Predict a prompt to inspect its context.</span>}
            </div>
            <span className="text-[11px] text-muted-foreground">Prompt + reply so far · {snapshot?.ids.length ?? 0} tokens with chat formatting</span>
          </>
        )}
        {selected === 1 && (
          <>
            <div className="flex items-center justify-between gap-2 text-xs font-medium">
              Refine the token representations
              <InfoTip>Each of the 32 learned blocks mixes earlier context through causal self-attention, then transforms each position with a feed-forward network (FFN). Normalization and residual connections support both operations. Each block has its own weights; 32 blocks are not 32 generated tokens. This schematic does not show hidden activations or attention strengths.</InfoTip>
            </div>
            <div className="flex items-center justify-center gap-2 rounded-lg border border-dashed p-3 text-center text-xs" aria-label="Schematic: causal attention then feed-forward network, repeated across 32 blocks">
              <span className="rounded border bg-background p-2">Mix context<br /><span className="text-[10px] text-muted-foreground">Causal attention</span></span>
              <ArrowRightIcon aria-hidden="true" className="size-3 shrink-0" />
              <span className="rounded border bg-background p-2">Transform<br /><span className="text-[10px] text-muted-foreground">Feed-forward</span></span>
              <span className="font-mono text-xs">×32</span>
            </div>
            <span className="text-[11px] text-muted-foreground">Architecture schematic · internal activations are not shown</span>
          </>
        )}
        {selected === 2 && (
          <>
            <div className="flex items-center justify-between gap-2 text-xs font-medium">
              Next-token candidates
              <span className="flex items-center gap-1 font-normal text-muted-foreground">Raw score<InfoTip>The output layer maps the final position’s representation to one score (logit) per vocabulary token: z = Wh. h: 960 final features; W: learned output weights; z: vocabulary scores. Higher scores favor a candidate; scores are not percentages. Three highest shown; all 49,152 continue to stage 3. Sampling settings change stage 3, not these raw scores. Click a candidate for its arithmetic.</InfoTip></span>
            </div>
            <div className="grid gap-1">
              {snapshot ? rows.slice(0, 3).map((row) => (
                <button key={row.id} type="button" onClick={() => inspect(row)} aria-label={`Inspect model score for ${piece(row.id)}`} className="flex min-w-0 items-center justify-between gap-2 rounded border bg-background px-2 py-1 text-left font-mono text-xs hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
                  <span className="truncate">{piece(row.id)}</span>
                  <span className="tabular-nums">{row.logit.toFixed(2)}</span>
                </button>
              )) : <span className="text-xs text-muted-foreground">Predict a prompt to see real model scores.</span>}
            </div>
            <span className="flex items-center gap-1 text-[11px] text-muted-foreground">{VOCABULARY_SIZE.toLocaleString("en-US")} scores → probabilities & sampling below<ArrowRightIcon aria-hidden="true" className="size-3 rotate-90" /></span>
          </>
        )}
      </div>
      <Collapsible>
        <CollapsibleTrigger className="group flex w-full items-center justify-between rounded text-left text-[11px] text-muted-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
          Architecture details
          <ChevronDownIcon aria-hidden="true" className="size-3 transition-transform group-data-[state=open]:rotate-180" />
        </CollapsibleTrigger>
        <CollapsibleContent className="pt-3">
          <div className="grid gap-2 text-[11px] text-muted-foreground">
            <span>32 learned blocks · each token represented by 960 numbers</span>
            <div className="grid grid-cols-8 gap-1" aria-label="Layer numbers of the 32 pretrained Transformer blocks">
              {Array.from({ length: 32 }, (_, i) => <span key={i} className="rounded border bg-background py-1 text-center font-mono">{i + 1}</span>)}
            </div>
            <span>Final position: 960 features → 49,152 vocabulary scores</span>
          </div>
        </CollapsibleContent>
      </Collapsible>
    </div>
  );
}

/** The visual route and transport consume the same real snapshot. */
export function LlmSamplingLab() {
  const model = useLlmModel();
  const [prompt, setPrompt] = React.useState<string>(PROMPTS[0].value);
  const [settings, setSettings] = React.useState(DEFAULTS);
  const [seed, setSeed] = React.useState(7);
  const [snapshot, setSnapshot] = React.useState<ModelSnapshot | null>(null);
  const [initial, setInitial] = React.useState<ModelSnapshot | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [playing, setPlaying] = React.useState(false);
  const running = React.useRef(false);
  const operation = React.useRef(false);
  const mounted = React.useRef(true);
  const [error, setError] = React.useState("");
  const [counts, setCounts] = React.useState<Record<number, number>>({});
  const random = React.useRef(mulberry32(7));
  const [lastDraw, setLastDraw] = React.useState<number | null>(null);
  const [history, setHistory] = React.useState<Trace[]>([]);
  const [reference, setReference] = React.useState<Reference | null>(null);
  const [inspection, setInspection] = React.useState<{
    row: Candidate;
    snapshot: ModelSnapshot;
    settings: SamplingSettings;
    roll?: number;
  } | null>(null);
  const tokenRef = React.useRef<HTMLDivElement>(null);
  const reducedMotion = usePrefersReducedMotion();
  React.useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      running.current = false;
    };
  }, []);
  const rows = React.useMemo(
    () => (snapshot ? distribution(snapshot.logits, settings) : []),
    [snapshot, settings],
  );
  const kept = rows.filter((row) => row.probability > 0);
  const total = Object.values(counts).reduce((sum, count) => sum + count, 0);
  const entropy = -kept.reduce(
    (sum, row) => sum + row.probability * Math.log2(row.probability),
    0,
  );
  const display = rows.slice(0, 10);
  const displayedMass = display.reduce((sum, row) => sum + row.probability, 0);
  const boundary =
    settings.temperature > 0 && settings.topP < 1 ? kept.at(-1) : undefined;
  const ended = history.at(-1)?.row.id === EOS_ID;
  const canAdvance =
    !!snapshot &&
    !busy &&
    !ended &&
    history.length < 32 &&
    snapshot.ids.length < CONTEXT_LIMIT;

  useIsomorphicLayoutEffect(() => {
    if (!tokenRef.current || reducedMotion || history.length === 0) return;
    const motion = animate(tokenRef.current, {
      scale: [0.97, 1],
      opacity: [0.65, 1],
      duration: 260,
      ease: "outQuad",
    });
    return () => {
      motion.revert();
    };
  }, [history.length, reducedMotion]);

  function clearDraws(nextSeed = seed) {
    setCounts({});
    setLastDraw(null);
    random.current = mulberry32(nextSeed);
  }
  function changeSettings(patch: Partial<SamplingSettings>) {
    setSettings((current) => ({ ...current, ...patch }));
    clearDraws();
  }
  function changePrompt(value: string) {
    setPrompt(value);
    setSnapshot(null);
    setInitial(null);
    setHistory([]);
    setReference(null);
    setError("");
    clearDraws();
  }
  async function predictPrompt(load = false) {
    if (operation.current) return;
    operation.current = true;
    setBusy(true);
    setError("");
    try {
      if (load) await model.load();
      const next = await model.predict(prompt);
      if (!mounted.current) return;
      setInitial(next);
      setSnapshot(next);
      setHistory([]);
      setReference(null);
      clearDraws();
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      operation.current = false;
      if (mounted.current) setBusy(false);
    }
  }
  function sampleMany(amount: number) {
    if (!rows.length) return;
    const next = { ...counts };
    let id = 0;
    for (let i = 0; i < amount; i++) {
      id = draw(rows, random.current).id;
      next[id] = (next[id] ?? 0) + 1;
    }
    setCounts(next);
    setLastDraw(id);
  }
  async function generate(amount: number) {
    if (!snapshot || operation.current || !canAdvance) return;
    operation.current = true;
    running.current = true;
    setPlaying(amount > 1);
    setBusy(true);
    setError("");
    let current = snapshot;
    try {
      for (let i = 0; i < amount && running.current; i++) {
        if (current.ids.length >= CONTEXT_LIMIT || history.length + i >= 32)
          break;
        const candidates = distribution(current.logits, settings);
        const roll = random.current();
        const row = draw(candidates, () => roll);
        // Finish an in-flight token even after Pause, keeping text and scores aligned.
        const next =
          row.id === EOS_ID
            ? current
            : await model.predict([...current.ids, row.id]);
        if (!mounted.current) return;
        const trace: Trace = {
          snapshot: current,
          row,
          settings: { ...settings },
          roll,
        };
        setHistory((previous) => [...previous, trace]);
        current = next;
        setSnapshot(next);
        setReference(null);
        setCounts({});
        setLastDraw(null);
        if (row.id === EOS_ID) break;
      }
    } catch (e) {
      if (mounted.current) setError(e instanceof Error ? e.message : String(e));
    } finally {
      operation.current = false;
      running.current = false;
      if (mounted.current) {
        setBusy(false);
        setPlaying(false);
      }
    }
  }
  function resetContinuation() {
    // Rebuild the original prefill cache as well as the visible distribution.
    // This keeps seeded replay on the same incremental q8 computation path.
    void predictPrompt();
  }
  const piece = (id: number) =>
    id === EOS_ID ? "[end]" : tokenLabel(model.vocabulary[id] ?? String(id));

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid min-w-0 content-start gap-4">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Next-token sampling
              <InfoTip>
                Real {MODEL_NAME} scores. Every generated token extends the prefix
                and runs the pretrained Transformer again. No prompt biases or
                scripted answers.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Badge variant="secondary">
                {model.status === "ready"
                  ? "Local ONNX · q8"
                  : "Pretrained model"}
              </Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-5">
            {model.status !== "ready" ? (
              <div className="grid justify-items-center gap-5 rounded-lg bg-muted/30 px-4 py-10">
                <div
                  className="flex flex-wrap items-center justify-center gap-3"
                  aria-hidden="true"
                >
                  {[
                    "Text",
                    "Tokens",
                    "Transformer",
                    "Scores",
                    "Next token",
                  ].map((label, i) => (
                    <React.Fragment key={label}>
                      {i > 0 && (
                        <ArrowRightIcon className="size-4 text-muted-foreground" />
                      )}
                      <span className="rounded-lg border bg-background p-3 font-mono text-xs">
                        {label}
                      </span>
                    </React.Fragment>
                  ))}
                </div>
                <Button
                  onClick={() => void predictPrompt(true)}
                  disabled={busy || !prompt.trim()}
                >
                  <DownloadIcon data-icon="inline-start" />
                  {busy ? "Loading model…" : `Load ${MODEL_NAME}`}
                </Button>
                <span
                  className="text-center text-xs text-muted-foreground"
                  role="status"
                >
                  {busy
                    ? `${model.progress.message}${model.progress.progress === undefined ? "" : ` · ${model.progress.progress.toFixed(0)}%`}`
                    : "First load ≈ 365 MB · browser cache · no API key"}
                </span>
              </div>
            ) : (
              <>
                <div className="grid gap-4 md:grid-cols-[minmax(0,1fr)_auto_minmax(0,1fr)]">
                  <Stage
                    n={1}
                    title="Context"
                    metric={snapshot ? `${snapshot.ids.length} tokens` : "—"}
                    info="These are actual tokenizer IDs and decoded pieces, including the model’s system/user/assistant chat formatting. Click a generated piece to inspect the distribution that selected it. · marks a space; ↵ a line break. Single-token Unicode fragments may look incomplete; the full text is decoded together."
                  >
                    <div className="flex min-h-24 flex-wrap content-start gap-1 rounded-lg border bg-muted/20 p-3">
                      {snapshot ? (
                        snapshot.pieces
                          .slice(
                            0,
                            snapshot.ids.length -
                              history.filter((t) => t.row.id !== EOS_ID).length,
                          )
                          .map((part, i) => (
                            <span
                              key={i}
                              title={`Token ID ${snapshot.ids[i]}`}
                              className="rounded border bg-background px-2 py-1 font-mono text-xs"
                            >
                              {tokenLabel(part)}
                            </span>
                          ))
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          Predict a prompt to inspect its tokens.
                        </span>
                      )}
                      {history.map((trace, i) => (
                        <button
                          key={`generated-${i}`}
                          onClick={() => setInspection({ ...trace })}
                          className="rounded border border-primary/40 bg-primary/10 px-2 py-1 font-mono text-xs hover:bg-primary/20 focus-visible:ring-2 focus-visible:ring-ring"
                          title={`Inspect step ${i + 1}`}
                        >
                          {piece(trace.row.id)}
                        </button>
                      ))}
                    </div>
                  </Stage>
                  <ArrowRightIcon className="hidden size-4 self-center text-muted-foreground md:block" />
                  <Stage
                    n={2}
                    title="Compute token scores"
                    metric={snapshot ? `${snapshot.milliseconds.toFixed(0)} ms` : "—"}
                    info="The Transformer uses the context to score every possible next token. Explore the three steps below. Context pieces and output scores are real; the internal-block view is an architecture schematic. A new generated token changes the context and triggers a new prediction."
                  >
                    <TransformerComputation snapshot={snapshot} rows={rows} piece={piece} inspect={(row) => snapshot && setInspection({ row, snapshot, settings })} />
                  </Stage>
                </div>
                <Stage
                  n={3}
                  title="Shape the distribution"
                  metric={
                    snapshot
                      ? `${kept.length.toLocaleString()} retained / ${VOCABULARY_SIZE.toLocaleString("en-US")}`
                      : "—"
                  }
                  info="All 49,152 model logits participate. Bars compare softmax at T=1 with the final sampling probability. Top-k is applied before top-p, with renormalization between filters. Top-p includes the candidate that reaches or crosses the threshold. Click a row for arithmetic."
                >
                  {snapshot ? (
                    <>
                      <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                        <span>Token · raw model logit</span>
                        <span>Outline: T=1 · solid: final probability</span>
                      </div>
                      <div className="grid gap-1">
                        {display.map((row) => (
                          <button
                            key={row.id}
                            onClick={() =>
                              setInspection({ row, snapshot, settings })
                            }
                            className={cn(
                              "group grid grid-cols-[minmax(5rem,0.7fr)_minmax(3rem,1fr)_4.5rem] items-center gap-3 rounded-md px-2 py-2 text-left hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-ring",
                              lastDraw === row.id && "bg-primary/10",
                              row.droppedBy && "opacity-50",
                            )}
                          >
                            <span className="flex min-w-0 flex-wrap items-baseline gap-x-2">
                              <span className="truncate font-mono text-xs">
                                {piece(row.id)}
                              </span>
                              <span className="font-mono text-[10px] text-muted-foreground">
                                {row.logit.toFixed(2)}
                              </span>
                            </span>
                            <span className="relative h-5 rounded bg-muted/60">
                              <span
                                className="absolute inset-y-0 left-0 min-w-px rounded border border-foreground/35"
                                style={{ width: `${row.base * 100}%` }}
                              />
                              <span
                                className="absolute inset-y-1 left-0 rounded bg-primary/75 motion-safe:transition-[width]"
                                style={{ width: `${row.probability * 100}%` }}
                              />
                            </span>
                            <span className="text-right font-mono text-[11px] tabular-nums">
                              {row.droppedBy ?? pct(row.probability)}
                            </span>
                          </button>
                        ))}
                      </div>
                      <div className="flex flex-wrap justify-between gap-2 border-t pt-2 font-mono text-[11px] text-muted-foreground">
                        <span>
                          Other {(VOCABULARY_SIZE - display.length).toLocaleString("en-US")} tokens:{" "}
                          {pct(Math.max(0, 1 - displayedMass))}
                        </span>
                        <span>
                          H = {entropy.toFixed(2)} bits · 2ᴴ ={" "}
                          {Math.pow(2, entropy).toFixed(1)} choices
                          <InfoTip>
                            H = −Σ qᵢ log₂ qᵢ over retained candidates; 2ᴴ is
                            perplexity, an effective choice count. Both describe
                            this distribution, not model quality.
                          </InfoTip>
                        </span>
                      </div>
                      {boundary && (
                        <div className="flex flex-wrap items-center gap-2 rounded-lg bg-muted/40 p-3 text-xs">
                          <span className="flex items-center gap-1">
                            Top-p boundary
                            <InfoTip>
                              Accumulated mass after top-k, before the final
                              normalization. The last retained token crosses or
                              exactly reaches p. Click to inspect this candidate
                              even when it is outside the ten chart rows.
                            </InfoTip>
                          </span>
                          <button
                            className="rounded border bg-background px-2 py-1 font-mono hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring"
                            onClick={() =>
                              setInspection({
                                row: boundary,
                                snapshot,
                                settings,
                              })
                            }
                          >
                            {piece(boundary.id)} ·{" "}
                            {pct(boundary.cumulative - boundary.afterK)} →{" "}
                            {pct(boundary.cumulative)}
                          </button>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="rounded-lg border border-dashed p-6 text-center text-sm text-muted-foreground">
                      Predict a prompt to reveal real model scores.
                    </div>
                  )}
                </Stage>
              </>
            )}
            {error && (
              <Alert variant="destructive">
                <AlertDescription>
                  {error} Retry with “
                  {model.status === "ready"
                    ? "Predict prompt"
                    : `Load ${MODEL_NAME}`}
                  ”.
                </AlertDescription>
              </Alert>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Generate a continuation
              <InfoTip>
                Step selects one token and recomputes logits. Run repeats up to
                32 new tokens. Pause finishes the current inference. Inspect any
                generated token to see its actual prior context, settings and
                probability.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Badge variant="outline">{history.length} / 32</Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div
              ref={tokenRef}
              className="min-h-24 whitespace-pre-wrap break-words rounded-lg bg-muted/30 p-4 font-mono text-sm leading-7"
              aria-live="polite"
            >
              <span className="text-muted-foreground">
                {initial?.text ?? prompt}
              </span>
              <span>
                {snapshot && initial
                  ? snapshot.text.slice(initial.text.length)
                  : ""}
              </span>
              {ended && (
                <Badge variant="outline" className="ml-2">
                  End of answer
                </Badge>
              )}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                disabled={!canAdvance}
                onClick={() => void generate(1)}
              >
                <StepForwardIcon data-icon="inline-start" />
                Step token
              </Button>
              {playing ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    running.current = false;
                  }}
                >
                  <PauseIcon data-icon="inline-start" />
                  Pause
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="outline"
                  disabled={!canAdvance}
                  onClick={() => void generate(32 - history.length)}
                >
                  <PlayIcon data-icon="inline-start" />
                  Run 32 tokens
                </Button>
              )}
              <Button
                size="sm"
                variant="ghost"
                disabled={busy || !history.length}
                onClick={resetContinuation}
              >
                <RotateCcwIcon data-icon="inline-start" />
                Reset continuation
              </Button>
              <span className="text-xs text-muted-foreground" role="status">
                {busy
                  ? "Running the model…"
                  : history.length >= 32
                    ? "32-token lab limit reached"
                    : snapshot && snapshot.ids.length >= CONTEXT_LIMIT
                      ? "Context budget reached"
                      : history.length
                        ? "Click a context token to inspect its step"
                        : ""}
              </span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Hold context. Repeat the draw.
              <InfoTip>
                These independent samples reuse the current model logits. They
                are a frequency experiment, not generated text. Settings changes
                clear counts and rewind the seed; individual draws advance one
                random stream.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Badge variant="outline">{total.toLocaleString()} draws</Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex flex-wrap gap-2">
              <Button
                size="sm"
                variant="outline"
                disabled={!snapshot || busy}
                onClick={() => sampleMany(1)}
              >
                <DicesIcon data-icon="inline-start" />
                Draw once
              </Button>
              <Button
                size="sm"
                disabled={!snapshot || busy}
                onClick={() => sampleMany(1000)}
              >
                Draw 1,000
              </Button>
              <Button
                size="sm"
                variant="outline"
                disabled={!snapshot || busy}
                onClick={() =>
                  setReference({
                    rows,
                    settings: { ...settings },
                    counts: { ...counts },
                    total,
                  })
                }
              >
                Pin comparison
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={busy || !total}
                onClick={() => clearDraws()}
              >
                Clear draws
              </Button>
            </div>
            {snapshot && (
              <div className="grid gap-2">
                <div className="flex flex-wrap items-center justify-between gap-2 text-[11px] text-muted-foreground">
                  <span>Outline: expected · solid: observed</span>
                  <span>{settingsLabel(settings)}</span>
                </div>
                {display.slice(0, 5).map((row) => (
                  <div
                    key={row.id}
                    className="grid grid-cols-[5rem_minmax(0,1fr)_5.5rem] items-center gap-3 text-xs"
                  >
                    <span className="truncate font-mono">{piece(row.id)}</span>
                    <div className="relative h-5 rounded bg-muted/50">
                      <span
                        className="absolute inset-y-0 left-0 min-w-px rounded border border-foreground/40"
                        style={{ width: `${row.probability * 100}%` }}
                      />
                      <span
                        className="absolute inset-y-1 left-0 rounded bg-primary/75"
                        style={{
                          width: `${total ? ((counts[row.id] ?? 0) / total) * 100 : 0}%`,
                        }}
                      />
                    </div>
                    <span className="text-right font-mono text-[11px]">
                      {counts[row.id] ?? 0} / {total}
                    </span>
                  </div>
                ))}
                {lastDraw !== null && (
                  <span className="text-xs text-muted-foreground">
                    Last draw:{" "}
                    <span className="font-mono text-foreground">
                      {piece(lastDraw)}
                    </span>
                  </span>
                )}
              </div>
            )}
            {reference && (
              <div className="grid gap-3 rounded-lg border bg-muted/20 p-3">
                <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
                  <span>
                    Pinned · {settingsLabel(reference.settings)} ·{" "}
                    {reference.total.toLocaleString()} draws
                  </span>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={() => setReference(null)}
                  >
                    Unpin
                  </Button>
                </div>
                <div className="grid grid-cols-[minmax(0,1fr)_4.5rem_4.5rem] gap-2 font-mono text-[11px]">
                  <span>Same prefix · probability</span>
                  <span className="text-right">Pinned</span>
                  <span className="text-right">Current</span>
                  {display.slice(0, 5).map((row) => (
                    <React.Fragment key={row.id}>
                      <span>{piece(row.id)}</span>
                      <span className="text-right">
                        {pct(
                          reference.rows.find((x) => x.id === row.id)
                            ?.probability ?? 0,
                        )}
                      </span>
                      <span className="text-right">{pct(row.probability)}</span>
                    </React.Fragment>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>

        <Collapsible>
          <Card>
            <CardHeader>
              <CollapsibleTrigger asChild>
                <button className="group flex items-center justify-between gap-2 text-left">
                  <CardTitle className="flex items-center gap-2">
                    <ScanSearchIcon className="size-4" />
                    How this experiment works
                  </CardTitle>
                  <ChevronDownIcon className="size-4 group-data-[state=open]:rotate-180" />
                </button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="grid gap-4 text-sm text-muted-foreground">
                <p>
                  <a
                    href={`https://huggingface.co/${MODEL_ID}`}
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    {MODEL_ID}
                  </a>{" "}
                  · 360M parameters · 32 causal Transformer blocks · 49,152 tokens
                  · q8 ONNX. An instruction-tuned English model; it can still
                  repeat itself or produce inaccurate text. Its own chat template
                  adds system/user/assistant tokens to the context. Model details:{" "}
                  <a
                    href={`https://huggingface.co/${MODEL_ID}`}
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    original model card
                  </a>
                  .
                </p>
                <p>
                  Predictions run in a browser worker using{" "}
                  <a
                    href="https://huggingface.co/docs/transformers.js/v3.8.1/index"
                    target="_blank"
                    rel="noreferrer"
                    className="underline"
                  >
                    Transformers.js
                  </a>{" "}
                  and ONNX Runtime. The first load fetches public weights and
                  runtime files. Prompt text stays in this browser. The browser
                  caches model files where available; rehearse loading on the
                  classroom device. Inference uses one CPU thread; weights remain in memory while this page is open. No offline availability is assumed.
                </p>
                <div className="rounded-lg bg-muted p-3 font-mono text-xs leading-7">
                  pᵢ(T) = exp((zᵢ − max z) / T) / Σⱼ exp((zⱼ − max z) / T)
                  <br />K = top-k IDs; rᵢ = pᵢ / Σⱼ∈K pⱼ
                  <br />S = smallest ranked prefix with Σᵢ∈S rᵢ ≥ top-p
                  <br />
                  qᵢ = rᵢ / Σⱼ∈S rⱼ for i ∈ S; otherwise 0<br />T = 0: argmax z
                  · top-k = all: no k filter
                </div>
                <p>
                  z is the model’s raw logit vector; T is positive temperature;
                  p is the tempered distribution; r is normalized after top-k; S
                  is the nucleus; q is the final sampling distribution. Positive
                  temperature preserves ranking. Sampling changes token
                  selection, not model weights. Frequencies from finite runs
                  need not equal probabilities.
                </p>
                <p>
                  Try T=0.5 then T=2 with k=all, p=1. Pin one distribution to
                  compare. Then isolate k=2, and finally p=0.8 with k=all. Reset
                  continuation to return to the same prefix. In generation, each
                  new token changes the next distribution. The lab caps context
                  at 128 tokens including chat formatting (model capacity: 8,192); generation
                  caps at 32 new tokens. It reuses a KV cache only for an exact one-token continuation;
                  new prompts and reset branches recompute the context. Unload
                  model releases the worker and its memory.
                </p>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Experiment controls</CardTitle>
          <CardAction>
            <Button
              size="icon-sm"
              variant="ghost"
              aria-label="Reset all"
              disabled={busy}
              onClick={() => {
                setSettings(DEFAULTS);
                setSeed(7);
                changePrompt(PROMPTS[0].value);
                random.current = mulberry32(7);
              }}
            >
              <RotateCcwIcon className="size-4" />
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-5">
          <ControlGroup label="Input">
            <SelectField
              label="Prompt starter"
              value={
                PROMPTS.some((p) => p.value === prompt) ? prompt : "custom"
              }
              options={[
                ...PROMPTS,
                { value: "custom", label: "Your own prompt" },
              ]}
              disabled={busy}
              onValueChange={(value) =>
                changePrompt(value === "custom" ? "" : value)
              }
              info="Ask a question or give a short instruction. The model’s chat template adds the system message and starts an assistant reply; inspect these tokens in Context."
            />
            <label className="grid gap-2 text-sm" htmlFor="llm-prompt">
              Prompt
              <textarea
                id="llm-prompt"
                value={prompt}
                disabled={busy}
                onChange={(e) => changePrompt(e.target.value)}
                maxLength={2000}
                rows={3}
                className="w-full resize-y rounded-md border bg-background p-2 font-mono text-xs focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
              />
            </label>
            {model.status === "ready" && (
              <div className="flex flex-wrap gap-2">
                <Button
                  size="sm"
                  disabled={busy || !prompt.trim()}
                  onClick={() => void predictPrompt()}
                >
                  Predict prompt
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy}
                  title="Release model memory; cached downloads stay available."
                  onClick={() => {
                    changePrompt(prompt);
                    model.unload();
                  }}
                >
                  Unload model
                </Button>
              </div>
            )}
          </ControlGroup>
          <Separator />
          <ControlGroup label="Sampling policy">
            <SliderField
              label="Temperature"
              value={settings.temperature}
              min={0}
              max={2}
              step={0.05}
              format={(v) => (v === 0 ? "0 · greedy" : v.toFixed(2))}
              disabled={busy}
              onValueChange={(temperature) => changeSettings({ temperature })}
              info="Divide logits by T before softmax. Lower T concentrates probabilities; higher T spreads them. T=0 picks the highest score deterministically."
            />
            <SelectField
              label="Top-k"
              value={String(settings.topK)}
              onValueChange={(value) => changeSettings({ topK: Number(value) })}
              options={[0, 1, 2, 5, 10, 50, 100].map((k) => ({
                value: String(k),
                label: k === 0 ? "All 49,152 tokens" : String(k),
              }))}
              disabled={busy || settings.temperature === 0}
              info="Keep the k highest-score candidates, then renormalize. All disables the filter."
            />
            <SliderField
              label="Top-p"
              value={settings.topP}
              min={0.05}
              max={1}
              step={0.05}
              format={(v) => v.toFixed(2)}
              disabled={busy || settings.temperature === 0}
              onValueChange={(topP) => changeSettings({ topP })}
              info="After top-k, keep the smallest probability-ranked prefix whose mass reaches or exceeds p, including the crossing candidate. Renormalize again."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="Reproducibility">
            <label className="grid gap-2 text-sm" htmlFor="llm-seed">
              Seed
              <Input
                id="llm-seed"
                type="number"
                min={0}
                max={4294967295}
                value={seed}
                disabled={busy}
                onChange={(e) => {
                  const next = Math.max(
                    0,
                    Math.min(
                      4294967295,
                      Math.trunc(Number(e.target.value) || 0),
                    ),
                  );
                  setSeed(next);
                  clearDraws(next);
                }}
              />
            </label>
            <div className="flex items-center justify-between text-xs text-muted-foreground">
              <span>Seeded random stream</span>
              <InfoTip>
                Each draw advances Mulberry32. Clear draws, seed changes,
                settings changes and reset continuation rewind it. Exact replay
                also requires the same prefix, settings and model/backend.
                Generation shares this stream with repeated draws.
              </InfoTip>
            </div>
          </ControlGroup>
        </CardContent>
      </Card>

      <Dialog
        open={!!inspection}
        onOpenChange={(open) => {
          if (!open) setInspection(null);
        }}
      >
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-2xl">
          {inspection && (
            <>
              <DialogTitle>Inside token {piece(inspection.row.id)}</DialogTitle>
              <DialogDescription>
                Actual model output · {settingsLabel(inspection.settings)}
              </DialogDescription>
              <div className="break-words rounded-lg bg-muted/40 p-3 font-mono text-xs">
                {inspection.snapshot.context}
              </div>
              <div className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-3">
                {[
                  ["Token ID", inspection.row.id],
                  ["Raw model logit z", inspection.row.logit.toFixed(5)],
                  [
                    "Scaled logit z/T",
                    inspection.settings.temperature === 0
                      ? "greedy"
                      : inspection.row.scaled.toFixed(5),
                  ],
                  ["Softmax at T=1", pct(inspection.row.base)],
                  [
                    "After temperature",
                    inspection.settings.temperature === 0
                      ? "greedy"
                      : pct(inspection.row.tempered),
                  ],
                  ["After top-k", pct(inspection.row.afterK)],
                  ["Final probability q", pct(inspection.row.probability)],
                  ["Retention", inspection.row.droppedBy ?? "kept"],
                  ["Random draw u", inspection.roll?.toFixed(6) ?? "—"],
                ].map(([label, value]) => (
                  <div key={label} className="grid gap-1 rounded border p-3">
                    <span className="text-muted-foreground">{label}</span>
                    <span className="font-mono">{value}</span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-muted-foreground">
                The top-p crossing token stays in the nucleus. Removed tokens
                have zero final probability. Probabilities use all 49,152
                candidates, including those outside the visible chart.
              </p>
              <div className="grid gap-1 font-mono text-xs">
                {distribution(inspection.snapshot.logits, inspection.settings)
                  .slice(0, 8)
                  .map((row) => (
                    <div
                      key={row.id}
                      className={cn(
                        "flex items-center justify-between rounded px-2 py-1",
                        row.id === inspection.row.id && "bg-primary/10",
                      )}
                    >
                      <span>{piece(row.id)}</span>
                      <span>{row.droppedBy ?? pct(row.probability)}</span>
                    </div>
                  ))}
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
