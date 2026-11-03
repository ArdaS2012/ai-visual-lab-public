"use client";
import { useRef, useState } from "react";
import {
  ArrowRightIcon,
  DownloadIcon,
  PinIcon,
  PlayIcon,
  RotateCcwIcon,
  SearchIcon,
  SquareIcon,
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
  ChoiceField,
  InfoTip,
} from "@/components/lab/controls";
import { useDay2Runtime } from "@/hooks/use-day2-runtime";
import {
  downloadEvidence,
  type RetrievalResult,
  type RetrievalSettings,
  type WireEvent,
} from "@/lib/day2-protocol";
import corpus from "@/lib/data/day2-corpus.json";
import { cn } from "@/lib/utils";
import { RetrievalSimilarity } from "./retrieval-similarity";
import {
  StageTitle,
  EvidenceDetails,
  EvidenceInspector,
  JsonEvidence,
  evidenceGridKey,
} from "./day2-evidence";

const QUESTIONS = [
  {
    value: "When does registration for the Cedar research workshop close?",
    label: "Registration deadline",
  },
  { value: "How late can I sign up for Cedar?", label: "Paraphrase" },
  {
    value: "Who may register for Cedar after the deadline?",
    label: "Boundary / exception",
  },
  {
    value: "How many people fit in the Cedar workshop room?",
    label: "Missing evidence",
  },
];
const DEFAULT: RetrievalSettings = {
  query: QUESTIONS[0].value,
  size: 80,
  overlap: 20,
  candidates: 5,
  selected: 2,
};
const shortSource = (id: string) =>
  ({
    "cedar-workshop": "Cedar W",
    "cedar-library": "Cedar L",
    "birch-workshop": "Birch W",
  })[id] ?? id;
const sourceIndex = (id: string) =>
  corpus.retrieval.findIndex((d) => d.id === id);
const MARKS = [
  "bg-chart-1/15 border-chart-1/50",
  "bg-chart-2/15 border-chart-2/50",
  "bg-chart-3/15 border-chart-3/50",
];

export function RetrievalReranking() {
  const runtime = useDay2Runtime<WireEvent>("retrieval");
  const [settings, setSettings] = useState(DEFAULT);
  const [prediction, setPrediction] = useState("cedar-workshop");
  const [baseline, setBaseline] = useState<RetrievalResult | null>(null);
  const [source, setSource] = useState("cedar-workshop");
  const [inspect, setInspect] = useState<string | null>(null);
  const [focus, setFocus] = useState<string | null>(null);
  const [stage, setStage] = useState("sources");
  const stageNav = useRef<HTMLElement>(null);
  const showStage = (next: string) => {
    setStage(next);
    stageNav.current?.scrollIntoView({ block: "start", behavior: "instant" });
    stageNav.current
      ?.querySelector<HTMLButtonElement>(`button[data-stage="${next}"]`)
      ?.focus({ preventScroll: true });
  };
  const [support, setSupport] = useState<Record<string, string>>({});
  const result = runtime.events.find((e) => e.kind === "retrieval")?.data as
    | RetrievalResult
    | undefined;
  const update = (patch: Partial<RetrievalSettings>) => {
    runtime.clear();
    setStage("sources");
    setSettings((s) => {
      const next = { ...s, ...patch };
      if (next.overlap >= next.size) next.overlap = next.size - 5;
      return next;
    });
    setFocus(null);
    setInspect(null);
    setSupport({});
  };
  const doc = corpus.retrieval.find((d) => d.id === source)!;
  const chunks = result?.chunks.filter((c) => c.source === source) ?? [];
  const inspected = result?.chunks.find((c) => c.id === inspect);
  const ranked = result?.candidates.find((c) => c.id === inspect);
  const selected = result?.candidates.filter((c) => c.selected) ?? [];
  const focused =
    result?.chunks.find((c) => c.id === focus) ??
    result?.candidates.find((c) => c.initial_rank === 1);
  const sourceTitle = (id: string) =>
    corpus.retrieval.find((d) => d.id === id)?.title ?? id;
  const selectPassage = (id: string) => {
    setFocus(id);
    const chunk = result?.chunks.find((c) => c.id === id);
    if (chunk) setSource(chunk.source);
  };
  const reset = () => {
    runtime.clear();
    setSettings(DEFAULT);
    setSource("cedar-workshop");
    setPrediction("cedar-workshop");
    setBaseline(null);
    setSupport({});
    setInspect(null);
    setFocus(null);
    setStage("sources");
  };
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid min-w-0 gap-4">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <SearchIcon className="size-4" />
              RAG: retrieval and reranking
              <InfoTip>
                Follow source text → vector matches → reranking → chosen
                passages. This demo returns evidence for an answer model; it
                does not generate an answer.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Badge variant="secondary">Real models · local CPU</Badge>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-6">
            <nav
              ref={stageNav}
              aria-label="Retrieval experiment stages"
              className="grid scroll-mt-20 grid-cols-2 gap-2 sm:grid-cols-4"
            >
              {[
                ["sources", "Read sources"],
                ["matches", "Find matches"],
                ["rerank", "Choose passages"],
                ["output", "Final output"],
              ].map(([key, title], i) => (
                <Button
                  key={key}
                  data-stage={key}
                  variant={stage === key ? "default" : "outline"}
                  aria-pressed={stage === key}
                  disabled={key !== "sources" && !result}
                  onClick={() => showStage(key)}
                  className="h-auto justify-start gap-2 whitespace-normal px-3 py-3 text-left text-xs"
                >
                  <span className="font-mono">0{i + 1}</span>
                  <span>{title}</span>
                </Button>
              ))}
            </nav>
            {result && (
              <div className="flex flex-wrap items-center justify-between gap-2 rounded-lg border bg-muted/20 px-3 py-2 text-xs">
                <span>
                  {result.chunks.length} passages → {result.candidates.length}{" "}
                  retrieved → <b>{selected.length} chosen</b>
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() =>
                    showStage(stage === "output" ? "matches" : "output")
                  }
                >
                  {stage === "output"
                    ? "Compare vector matches"
                    : "Read chosen passages"}
                  <ArrowRightIcon className="size-3" />
                </Button>
              </div>
            )}
            {stage === "sources" && (
              <section className="grid gap-3">
                <StageTitle
                  number={1}
                  title="Source → chunk boundaries"
                  metric={
                    result
                      ? `${result.chunks.length} chunks · ${settings.size} words`
                      : "Fictional classroom corpus"
                  }
                  info="Fixed whitespace-word windows. Click a chunk to inspect its exact [start,end) source span. Overlap repeats words; it does not create new evidence."
                />
                <div className="flex flex-wrap gap-2">
                  {corpus.retrieval.map((d) => (
                    <Button
                      size="sm"
                      variant={source === d.id ? "default" : "outline"}
                      key={d.id}
                      onClick={() => {
                        setSource(d.id);
                        setFocus(
                          result?.chunks.find((c) => c.source === d.id)?.id ??
                            null,
                        );
                      }}
                    >
                      {d.title}
                      <Badge variant="outline" className="text-current">
                        {d.text.split(/\s+/).length}w
                      </Badge>
                    </Button>
                  ))}
                </div>
                <div className="flex flex-wrap gap-2">
                  {chunks.map((c) => (
                    <Button
                      key={c.id}
                      size="sm"
                      variant={focused?.id === c.id ? "default" : "outline"}
                      onClick={() => selectPassage(c.id)}
                      aria-pressed={focused?.id === c.id}
                    >
                      Words [{c.start},{c.end})
                    </Button>
                  ))}
                </div>
                <div
                  className="rounded-xl border bg-muted/20 p-4 text-sm leading-8"
                  aria-label="Readable original source"
                >
                  {doc.text.split(/\s+/).map((word, i) => {
                    const highlighted =
                      focused?.source === source &&
                      i >= focused.start &&
                      i < focused.end;
                    return (
                      <span
                        key={i}
                        className={cn(
                          highlighted &&
                            "rounded-sm bg-primary/15 font-semibold",
                        )}
                      >
                        {word}{" "}
                      </span>
                    );
                  })}
                </div>
                {result && (
                  <Button
                    variant="outline"
                    onClick={() => showStage("matches")}
                  >
                    Find matching passages
                    <ArrowRightIcon className="size-4" />
                  </Button>
                )}
                <EvidenceDetails title="Word positions and overlap details">
                  <div className="rounded-xl border bg-muted/25 p-4">
                    <div
                      className="grid grid-cols-[repeat(24,minmax(0,1fr))] gap-1"
                      aria-label="Document word positions; use arrow keys"
                      onKeyDown={(e) => evidenceGridKey(e, 24)}
                    >
                      {doc.text.split(/\s+/).map((word, i) => {
                        const hits = chunks.filter(
                          (c) => i >= c.start && i < c.end,
                        );
                        return (
                          <button
                            key={i}
                            type="button"
                            disabled={!hits.length}
                            tabIndex={i === 0 ? 0 : -1}
                            onClick={() => setInspect(hits[0]?.id ?? null)}
                            aria-label={`Word ${i}: ${word}${hits[0] ? `, inspect ${hits[0].id}` : ""}`}
                            title={`${i}: ${word}`}
                            className={cn(
                              "h-5 min-w-0 rounded-sm border focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                              hits.length
                                ? MARKS[sourceIndex(source)]
                                : "bg-muted border-border",
                              hits.length > 1 &&
                                "border-dashed border-foreground/50",
                            )}
                          />
                        );
                      })}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {chunks.map((c) => (
                        <Button
                          variant="outline"
                          size="sm"
                          key={c.id}
                          onClick={() => setInspect(c.id)}
                          className={cn(
                            "font-mono text-xs",
                            MARKS[sourceIndex(c.source)],
                          )}
                        >
                          [{c.start}, {c.end})
                        </Button>
                      ))}
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">
                      {result
                        ? "One cell = one source word · dashed = overlap"
                        : "Run search to reveal measured chunk boundaries"}
                    </p>
                  </div>
                </EvidenceDetails>
              </section>
            )}
            {stage === "matches" && (
              <section className="grid gap-3">
                <StageTitle
                  number={2}
                  title="Question → matching passages"
                  metric={
                    result
                      ? `(384,) · ${result.query_tokens} model tokens${result.query_tokens > 256 ? " · TRUNCATED" : ""}`
                      : "(384,)"
                  }
                  info="Compare each passage vector with the question vector. All 384 coordinates determine cosine similarity. Larger cosine means a smaller angle, not a greater probability of truth. Only the top-k matches reach the reranker."
                />
                {result && focused ? (
                  <>
                    <RetrievalSimilarity
                      result={result}
                      passage={focused}
                      sourceTitle={sourceTitle}
                      onSelect={(c) => selectPassage(c.id)}
                    />
                    <Button
                      variant="outline"
                      onClick={() => showStage("rerank")}
                    >
                      See reranking of {result.candidates.length} retrieved
                      passages
                      <ArrowRightIcon className="size-4" />
                    </Button>
                  </>
                ) : (
                  <div className="rounded-xl border border-dashed p-6 text-sm text-muted-foreground">
                    Run search to compare real question and passage vectors.
                  </div>
                )}
              </section>
            )}
            {stage === "rerank" && (
              <section className="grid gap-3">
                <StageTitle
                  number={3}
                  title="Retrieved → chosen passages"
                  metric={
                    result
                      ? `${result.candidates.length} → ${selected.length}`
                      : "top-k → top-n"
                  }
                  info="Cosine retrieves candidates. The cross-encoder reads each query/passage pair and assigns a raw logit. Reranking cannot recover a passage outside this candidate set. Click a row for exact text, vectors and arithmetic."
                />
                {result ? (
                  <>
                    <div className="rounded-lg border bg-muted/20 p-3 text-sm">
                      Reranker reads the question with each of the{" "}
                      {result.candidates.length} retrieved passages → keeps{" "}
                      {selected.length}.
                    </div>
                    <div className="grid gap-2">
                      {result.candidates.map((c) => (
                        <button
                          key={c.id}
                          onClick={() => setInspect(c.id)}
                          className={cn(
                            "grid gap-2 rounded-xl border p-3 text-left transition-colors hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring sm:grid-cols-[3rem_minmax(0,1fr)]",
                            c.selected && "border-primary/50 bg-primary/5",
                          )}
                        >
                          <div className="flex items-center gap-1 font-mono text-sm">
                            <b>#{c.final_rank}</b>
                          </div>
                          <div className="min-w-0">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge variant="outline">
                                {corpus.retrieval[sourceIndex(c.source)].title}
                              </Badge>
                              <span className="font-mono text-xs text-muted-foreground">
                                [{c.start},{c.end})
                              </span>
                              <Badge
                                variant={c.selected ? "default" : "outline"}
                              >
                                {c.selected
                                  ? "Chosen for context"
                                  : "Not chosen"}
                              </Badge>
                            </div>
                            <p className="mt-2 line-clamp-2 text-sm leading-relaxed">
                              {c.text}
                            </p>
                            <p className="mt-2 text-xs text-muted-foreground">
                              Search rank {c.initial_rank} → reranker rank{" "}
                              {c.final_rank} · click to inspect source
                            </p>
                          </div>
                        </button>
                      ))}
                    </div>
                    <EvidenceDetails title="Rank changes and raw model scores">
                      <RankFlow result={result} />
                      {result.candidates.map((c) => (
                        <div
                          key={c.id}
                          className="flex flex-wrap justify-between gap-2 font-mono text-xs"
                        >
                          <span>{c.id}</span>
                          <span>
                            cosine {c.cosine.toFixed(4)} · reranker logit{" "}
                            {c.logit.toFixed(3)}
                          </span>
                        </div>
                      ))}
                    </EvidenceDetails>
                    <p className="text-xs text-muted-foreground">
                      Cosine and raw logits use different scales · neither is a
                      truth probability
                    </p>
                    <Button
                      variant="outline"
                      onClick={() => showStage("output")}
                    >
                      Read the {selected.length} chosen passages
                      <ArrowRightIcon className="size-4" />
                    </Button>
                  </>
                ) : (
                  <div className="flex min-h-44 items-center justify-center rounded-xl border border-dashed bg-muted/20 text-sm text-muted-foreground">
                    Predict a source, then run the experiment
                  </div>
                )}
              </section>
            )}
            {stage === "output" && selected.length > 0 && (
              <section className="grid gap-3">
                <StageTitle
                  number={4}
                  title="Final output: chosen source passages"
                  metric={`${selected.length} selected passages`}
                  info="Read the original words. Manually judge support independently of rank. For the room-capacity question, none of the three original documents provides a capacity."
                />
                <div className="rounded-xl border bg-primary/5 p-4">
                  <div className="mb-2 flex flex-wrap items-center gap-2">
                    <Badge>Context ready for an answer model</Badge>
                    <InfoTip>
                      These are the actual passages chosen by the reranker, in
                      final rank order. A complete RAG application would give
                      this context and the question to an answer model. No
                      answer model is called here. A selected passage can still
                      fail to answer the question; check its words.
                    </InfoTip>
                  </div>
                  <p className="text-sm font-medium">
                    {result?.settings.query}
                  </p>
                  <p className="mt-2 text-xs text-muted-foreground">
                    This demo returns passages; answer generation comes next.
                  </p>
                </div>
                {selected.map((c, i) => (
                  <div
                    key={c.id}
                    className="grid gap-3 rounded-xl border bg-muted/20 p-4"
                  >
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div className="grid gap-1">
                        <b className="text-sm">
                          {i + 1}. {sourceTitle(c.source)}
                        </b>
                        <code className="text-xs break-all">
                          {c.id} · words [{c.start},{c.end})
                        </code>
                      </div>
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => setInspect(c.id)}
                      >
                        Inspect source
                      </Button>
                    </div>
                    <p className="text-sm leading-relaxed">{c.text}</p>
                    <span className="text-xs text-muted-foreground">
                      Search rank {c.initial_rank} → reranker rank{" "}
                      {c.final_rank} → included in context
                    </span>
                    <ChoiceField
                      label="Your support judgment"
                      value={support[c.id] ?? "unchecked"}
                      options={[
                        { value: "supports", label: "Supports" },
                        { value: "partial", label: "Partial" },
                        { value: "absent", label: "Not stated" },
                      ]}
                      onChange={(value) =>
                        setSupport((s) => ({ ...s, [c.id]: value }))
                      }
                    />
                  </div>
                ))}
                <EvidenceDetails title="View assembled context preview">
                  <pre className="whitespace-pre-wrap break-words rounded-lg border p-3 font-mono text-xs">
                    {selected.map((c) => `[${c.id}]\n${c.text}`).join("\n\n")}
                  </pre>
                </EvidenceDetails>
              </section>
            )}
            <div
              role="status"
              aria-live="polite"
              className="flex flex-wrap items-center justify-between gap-2 text-xs text-muted-foreground"
            >
              <span>
                {runtime.status || "Ready · predict → run → inspect → explain"}
              </span>
              {result && (
                <span className="font-mono">
                  {result.seconds.toFixed(2)}s · prediction: {prediction}
                </span>
              )}
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
              <CardTitle>One-change comparison</CardTitle>
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
            <CardContent className="grid gap-3 sm:grid-cols-2">
              <Comparison label="Pinned baseline" result={baseline} />
              {result ? (
                <Comparison label="Current run" result={result} />
              ) : (
                <div className="rounded-lg border border-dashed p-4 text-sm text-muted-foreground">
                  Change one setting and run again
                </div>
              )}
              {result && (
                <p className="text-xs text-muted-foreground sm:col-span-2">
                  {comparisonText(baseline, result)} · changed IDs include
                  changed word boundaries.
                </p>
              )}
            </CardContent>
          </Card>
        )}
        <EvidenceDetails title="Equations, model provenance & learning prompts">
          <p>
            <b>Predict:</b> Which source contains the registration deadline?{" "}
            <b>Explain:</b> Did smaller chunks preserve the exception? Did
            reranking change the order? What cannot be answered?
          </p>
          <p>
            <code>
              e = normalize(mean_mask(H)); cos(q,d) = (q·d)/(‖q‖ ‖d‖); s(q,d) =
              cross_encoder(q,d)
            </code>
            . H is the token × 384 hidden-state matrix; q,d are 384D vectors.
            All scores are unitless. For (1,0) and (3,4), cosine = 3/5 = 0.6.
            Reranker logits are uncalibrated relevance scores.
          </p>
          <p>
            Exact full-dimensional cosine search over this tiny corpus; the
            student notebook uses Chroma. Embedding inputs truncate at 256 model
            tokens, pairs at 512. Word counts are separate from model tokens.
            Query and corpus are processed on the local CPU; downloads
            happen only during explicit preload.
          </p>
          <p>
            <a
              className="underline"
              href="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
            >
              MiniLM model card
            </a>{" "}
            ·{" "}
            <a
              className="underline"
              href="https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1"
            >
              Mixedbread reranker card
            </a>{" "}
            · Apache-2.0
          </p>
          {result && (
            <JsonEvidence
              value={{
                models: result.models,
                corpus_version: result.corpus_version,
                settings: result.settings,
              }}
            />
          )}
        </EvidenceDetails>
      </div>
      <Card className="order-first xl:order-last">
        <CardHeader>
          <CardTitle>Experiment</CardTitle>
          <CardAction>
            <Button
              variant="ghost"
              size="icon"
              aria-label="Reset retrieval experiment"
              onClick={reset}
            >
              <RotateCcwIcon className="size-4" />
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-5">
          <ControlGroup label="01 · Question & prediction">
            <SelectField
              label="Question"
              value={settings.query}
              options={QUESTIONS}
              onValueChange={(query) => update({ query })}
            />
            <Label htmlFor="retrieval-query">Edit question</Label>
            <Textarea
              rows={3}
              className="resize-none"
              id="retrieval-query"
              value={settings.query}
              maxLength={300}
              onChange={(e) => update({ query: e.target.value })}
            />
            <SelectField
              label="Predict the source"
              value={prediction}
              onValueChange={setPrediction}
              disabled={runtime.busy || !!result}
              options={corpus.retrieval.map((d) => ({
                value: d.id,
                label: d.title,
              }))}
              info="Make a prediction before measuring. This records a learner hypothesis, not a reference answer."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="02 · Chunking">
            <ChoiceField
              label="Chunk size · words"
              value={String(settings.size)}
              options={[
                { value: "40", label: "40 words" },
                { value: "80", label: "80 words" },
              ]}
              onChange={(size) => update({ size: Number(size) })}
            />
            <SliderField
              label="Overlap · words"
              value={settings.overlap}
              onValueChange={(overlap) => update({ overlap })}
              min={0}
              max={Math.min(40, settings.size - 1)}
              step={5}
              info="Stride = size − overlap. Hold overlap fixed when comparing chunk sizes."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="03 · Candidate selection">
            <SliderField
              label="Retrieve top-k"
              value={settings.candidates}
              onValueChange={(candidates) => update({ candidates })}
              min={1}
              max={8}
              info="Only these retrieved candidates reach the reranker."
            />
            <SliderField
              label="Select top-n"
              value={settings.selected}
              onValueChange={(selected) => update({ selected })}
              min={1}
              max={3}
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="04 · Run & compare">
            <Button
              onClick={() => {
                setFocus(null);
                setInspect(null);
                setSupport({});
                showStage("matches");
                void runtime.run(settings);
              }}
              disabled={runtime.busy || !settings.query.trim()}
            >
              <PlayIcon className="size-4" />
              Run search & rerank
            </Button>
            {runtime.busy && (
              <Button variant="outline" onClick={runtime.cancel}>
                <SquareIcon className="size-4" />
                Cancel
              </Button>
            )}
            <Button
              variant="outline"
              disabled={!result || runtime.busy}
              onClick={() => {
                if (result) setBaseline(structuredClone(result));
              }}
            >
              <PinIcon className="size-4" />
              Pin baseline
            </Button>
            <Button
              variant="ghost"
              disabled={!result}
              onClick={() =>
                downloadEvidence("retrieval-evidence.json", {
                  result,
                  baseline,
                  prediction,
                  support,
                })
              }
            >
              <DownloadIcon className="size-4" />
              Export evidence
            </Button>
          </ControlGroup>
        </CardContent>
      </Card>
      <EvidenceInspector
        title={inspected?.id ?? "Source evidence"}
        description="Exact source words, full-dimensional model evidence and original run settings."
        open={!!inspected}
        onClose={() => setInspect(null)}
      >
        {inspected && result && (
          <>
            <div className="rounded-lg border bg-muted/20 p-4 text-sm leading-7">
              {corpus.retrieval
                .find((d) => d.id === inspected.source)!
                .text.split(/\s+/)
                .map((word, i) => (
                  <span
                    key={i}
                    className={cn(
                      i >= inspected.start &&
                        i < inspected.end &&
                        "rounded-sm bg-primary/15 font-medium",
                    )}
                  >
                    {word}{" "}
                  </span>
                ))}
            </div>
            <div className="grid grid-cols-2 gap-3">
              <Badge variant="outline">
                words [{inspected.start},{inspected.end})
              </Badge>
              <Badge variant="outline">
                {inspected.model_tokens} model tokens{" "}
                {inspected.model_tokens > 256 ? "· TRUNCATED" : ""}
              </Badge>
            </div>
            <p className="font-mono text-sm">
              q·d = {inspected.cosine.toFixed(6)} · ‖q‖ ={" "}
              {Math.hypot(...result.query_vector).toFixed(6)} · ‖d‖ ={" "}
              {Math.hypot(...inspected.vector).toFixed(6)}
            </p>
            {ranked && (
              <p className="text-sm">
                Reranker pair: {ranked.pair_tokens} model tokens{" "}
                {ranked.pair_tokens > 512 ? "· TRUNCATED" : ""} · raw logit{" "}
                {ranked.logit.toFixed(6)}
              </p>
            )}
            <JsonEvidence
              value={{
                query: result.settings.query,
                query_vector: result.query_vector,
                chunk: inspected,
                ranked,
                settings: result.settings,
              }}
            />
          </>
        )}
      </EvidenceInspector>
    </div>
  );
}
function RankFlow({ result }: { result: RetrievalResult }) {
  const rows = result.candidates,
    n = rows.length,
    h = Math.max(110, n * 34);
  return (
    <div className="rounded-xl border bg-muted/20 p-3">
      <div className="mb-2 flex justify-between font-mono text-xs text-muted-foreground">
        <span>Cosine order</span>
        <span>Cross-encoder order</span>
      </div>
      <svg
        viewBox={`0 0 500 ${h}`}
        role="img"
        aria-label="Candidate rank changes; full passage rows below"
        className="max-h-64 w-full"
      >
        {rows.map((r) => {
          const y1 = ((r.initial_rank - 0.5) * h) / n,
            y2 = ((r.final_rank - 0.5) * h) / n;
          return (
            <g key={r.id}>
              <path
                d={`M 95 ${y1} C 220 ${y1} 280 ${y2} 405 ${y2}`}
                fill="none"
                className={
                  r.selected ? "stroke-chart-1" : "stroke-muted-foreground/35"
                }
                strokeWidth={r.selected ? 3 : 1.5}
              />
              <circle cx="95" cy={y1} r="4" className="fill-foreground" />
              <circle cx="405" cy={y2} r="4" className="fill-foreground" />
              <text x="4" y={y1 + 4} fontSize="11" className="fill-foreground">
                {r.initial_rank} · {shortSource(r.source)}
              </text>
              <text
                x="420"
                y={y2 + 4}
                fontSize="11"
                className="fill-foreground"
              >
                {r.final_rank} · {shortSource(r.source)}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
function Comparison({
  label,
  result,
}: {
  label: string;
  result: RetrievalResult;
}) {
  return (
    <div className="grid gap-2 rounded-xl border bg-muted/20 p-4">
      <b className="text-sm">{label}</b>
      <span className="font-mono text-xs">
        size {result.settings.size} · overlap {result.settings.overlap} · k{" "}
        {result.settings.candidates} · n {result.settings.selected}
      </span>
      <span className="text-xs">{result.settings.query}</span>
      {result.candidates
        .filter((r) => r.selected)
        .map((r) => (
          <div key={r.id} className="grid gap-1 border-t pt-2">
            <div className="flex items-center justify-between gap-2">
              <b className="text-xs">
                {r.final_rank}.{" "}
                {corpus.retrieval.find((d) => d.id === r.source)?.title}
              </b>
              <InfoTip label={`Raw baseline score for ${r.id}`}>
                Reranker logit {r.logit.toFixed(3)} · cosine{" "}
                {r.cosine.toFixed(4)}. Separate relevance scales, not truth
                probabilities.
              </InfoTip>
            </div>
            <code className="text-xs break-all">{r.id}</code>
            <p className="line-clamp-2 text-xs leading-relaxed text-muted-foreground">
              {r.text}
            </p>
          </div>
        ))}
    </div>
  );
}
function comparisonText(a: RetrievalResult, b: RetrievalResult) {
  const changes = (
    Object.keys(a.settings) as (keyof RetrievalSettings)[]
  ).filter((k) => a.settings[k] !== b.settings[k]);
  return changes.length === 1
    ? `Controlled comparison: changed only ${changes[0]}`
    : changes.length === 0
      ? "Repeated run: same settings"
      : `Multiple changes: ${changes.join(", ")} — isolate one for a causal comparison`;
}
