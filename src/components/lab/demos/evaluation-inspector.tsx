"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import {
  PlayIcon,
  SquareIcon,
  RotateCcwIcon,
  DownloadIcon,
} from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { InfoTip, SelectField } from "@/components/lab/controls";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import { useDay2Runtime } from "@/hooks/use-day2-runtime";
import { downloadEvidence } from "@/lib/day2-protocol";
import {
  benchmark,
  scoreAnswer,
  pairedResults,
  parseJudge,
  type EvaluationEvent,
  type EvaluationSettings,
} from "@/lib/evaluation-live";
import { EvaluationFixtures } from "./evaluation-fixtures";

const ratio = (n: number, d: number) =>
  d ? `${n}/${d} · ${Math.round((n / d) * 100)}%` : "—";

export function EvaluationInspector() {
  const run = useDay2Runtime<EvaluationEvent>("evaluation");
  const judge = useDay2Runtime<EvaluationEvent>("evaluation-judge");
  const [settings, setSettings] = useState<EvaluationSettings>({
    evidence: "full",
    format: "json",
  });
  const [selection, setSelection] = useState({
    id: "seats",
    variant: "B" as "A" | "B",
  });
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [humans, setHumans] = useState<Record<string, number>>({});
  const [judgments, setJudgments] = useState<Record<string, EvaluationEvent[]>>(
    {},
  );
  const start = run.events.find((e) => e.kind === "evaluation_start")?.data;
  const dataset = start?.benchmark ?? benchmark;
  const answers = run.events
    .filter((e) => e.kind === "evaluation_answer")
    .map((e) => e.data);
  const pair = pairedResults(answers, dataset.cases);
  const row = dataset.cases.find((c) => c.id === selection.id)!;
  const result = answers.find(
    (a) => a.case_id === row.id && a.variant === selection.variant,
  );
  const score = result ? scoreAnswer(row, result) : null;
  const key = `${start?.run_id}/${selection.variant}/${selection.id}`;
  const human = humans[key];
  const judgeEvents = judgments[key] ?? [];
  const judgment = judgeEvents.find((e) => e.kind === "evaluation_judge")?.data;
  const parsed = judgment ? parseJudge(judgment) : null;
  const complete = run.events.some((e) => e.kind === "terminal");
  const changed =
    start && JSON.stringify(settings) !== JSON.stringify(start.settings);
  const busy = run.busy || judge.busy;
  const compared = Object.entries(judgments).flatMap(([id, events]) => {
    const data = events.find((e) => e.kind === "evaluation_judge")?.data;
    const p = data ? parseJudge(data) : null;
    return p && humans[id] !== undefined
      ? [{ agree: p.rating === humans[id] }]
      : [];
  });
  const evidence =
    (start?.settings ?? settings).evidence === "full"
      ? "All 3 sources"
      : "Cedar removed";
  async function judgeSelected() {
    if (!result || human === undefined) return;
    // The key is captured before inference; navigation cannot reassign this judgment.
    const events = await judge.run({ case_id: row.id, raw: result.raw });
    if (events) setJudgments((old) => ({ ...old, [key]: events }));
  }
  function reset() {
    run.clear();
    judge.clear();
    setHumans({});
    setJudgments({});
    setSettings({ evidence: "full", format: "json" });
    setSelection({ id: "seats", variant: "B" });
    setInspectorOpen(false);
  }
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="order-2 grid min-w-0 gap-4 xl:order-1">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Does source evidence improve the answers?
              <InfoTip label="About the live experiment">
                Same local Qwen model, questions, instructions and greedy
                decoding. A receives no sources; B receives the selected
                catalogue evidence. References are frozen and never sent to the
                answering model. This isolates supplied context, not retrieval
                quality.
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid min-w-0 grid-cols-1 gap-4">
            <div className="grid min-w-0 gap-3 sm:grid-cols-2">
              {pair.variants.map((s) => (
                <section
                  key={s.variant}
                  className="grid gap-3 rounded-xl border bg-muted/30 p-4"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <span className="text-sm font-medium">
                      {s.variant} ·{" "}
                      {s.variant === "A" ? "No source evidence" : evidence}
                    </span>
                    <Badge variant="outline">
                      {answers.filter((a) => a.variant === s.variant).length}/4
                      generated
                    </Badge>
                  </div>
                  <div className="font-mono text-3xl tabular-nums">
                    {ratio(s.exact, pair.n)}
                    <span className="ml-2 font-sans text-xs text-muted-foreground">
                      correct
                    </span>
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    JSON schema{" "}
                    <span className="ml-auto font-mono">
                      {ratio(s.schema, pair.n)}
                    </span>
                    <InfoTip>
                      Always checks exactly answer:string and
                      sources:list[string], even when you ask for plain text. A
                      correct plain-text answer can fail this integration
                      contract.
                    </InfoTip>
                  </div>
                </section>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              <Badge variant={run.recording ? "secondary" : "outline"}>
                {run.recording
                  ? "Recorded real inference"
                  : start
                    ? "Live local inference"
                    : "Ready for real inference"}
              </Badge>
              <span>
                {pair.n}/4 paired cases
                {complete ? " · complete" : " · partial / not run"}
              </span>
              {pair.n > 0 && (
                <span className="font-mono">
                  B − A:{" "}
                  {Math.round(
                    ((pair.variants[1].exact - pair.variants[0].exact) /
                      pair.n) *
                      100,
                  )}{" "}
                  pp
                </span>
              )}
              <InfoTip>
                Rates and differences use only completed A/B pairs. Four visible
                development cases demonstrate evaluation mechanics; they are not
                a held-out benchmark or proof of general reliability.
              </InfoTip>
            </div>
            <div className="grid min-w-0 grid-cols-1 gap-2">
              {dataset.cases.map((c) => (
                <section key={c.id} className="rounded-xl border p-3">
                  <div className="mb-2 flex items-center gap-2 text-xs text-muted-foreground">
                    <Badge variant="outline">{c.label}</Badge>
                    <span className="truncate">{c.question}</span>
                    <InfoTip>{c.question}</InfoTip>
                  </div>
                  <div className="grid gap-2 sm:grid-cols-2">
                    {(["A", "B"] as const).map((variant) => {
                      const a = answers.find(
                        (v) => v.case_id === c.id && v.variant === variant,
                      );
                      const s = a ? scoreAnswer(c, a) : null;
                      const selected =
                        selection.id === c.id && selection.variant === variant;
                      return (
                        <button
                          key={variant}
                          type="button"
                          aria-label={`Inspect ${variant}: ${c.label}`}
                          aria-pressed={selected}
                          onClick={() => {
                            setSelection({ id: c.id, variant });
                            setInspectorOpen(true);
                          }}
                          className={`flex min-w-0 items-center gap-3 rounded-lg border p-3 text-left text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${selected ? "bg-primary/10 ring-2 ring-primary" : "hover:bg-muted/50"}`}
                        >
                          <span className="font-mono text-muted-foreground">
                            {variant}
                          </span>
                          <span className="min-w-0 flex-1 break-words font-mono">
                            {s ? s.answer : "Awaiting run"}
                          </span>
                          <Badge variant={s?.exact ? "default" : "outline"}>
                            {s ? (s.exact ? "Pass" : "Fail") : "—"}
                          </Badge>
                        </button>
                      );
                    })}
                  </div>
                </section>
              ))}
            </div>
          </CardContent>
        </Card>
        <Dialog open={inspectorOpen} onOpenChange={setInspectorOpen}>
          <DialogContent className="max-h-[90dvh] overflow-y-auto p-1 sm:max-w-4xl">
            <DialogHeader className="sr-only">
              <DialogTitle>Answer evidence inspector</DialogTitle>
              <DialogDescription>
                Compare the source, reference, raw answer and human/model
                judgment.
              </DialogDescription>
            </DialogHeader>
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  Inspect {selection.variant} · {row.label}
                  <InfoTip>
                    Click any A/B answer above. Check the original source,
                    frozen reference and raw response before trusting an
                    aggregate or a model judge.
                  </InfoTip>
                </CardTitle>
              </CardHeader>
              <CardContent className="grid min-w-0 grid-cols-1 gap-4">
                <div className="text-sm font-medium">{row.question}</div>
                <div className="grid gap-4 md:grid-cols-2">
                  <section className="grid content-start gap-2">
                    <div className="text-xs text-muted-foreground">
                      Original catalogue · evaluation evidence
                    </div>
                    {dataset.sources.map((source) => (
                      <div
                        key={source.id}
                        className={`rounded-lg border p-3 text-sm ${row.relevant.includes(source.id) ? "border-primary bg-primary/5" : "bg-muted/30"}`}
                      >
                        <span className="font-mono text-xs">[{source.id}]</span>{" "}
                        {source.text}
                        {result && (
                          <div className="mt-2 text-xs font-medium text-muted-foreground">
                            {result.supplied_sources.includes(source.id)
                              ? "Supplied to this answer"
                              : "Withheld from this answer"}
                          </div>
                        )}
                      </div>
                    ))}
                  </section>
                  <section className="grid content-start gap-3">
                    <div className="rounded-lg bg-muted/50 p-3">
                      <div className="mb-1 text-xs text-muted-foreground">
                        Frozen reference
                      </div>
                      <div className="font-mono">{row.reference}</div>
                    </div>
                    <div>
                      <div className="mb-2 text-xs text-muted-foreground">
                        Actual raw model response
                      </div>
                      <pre className="max-h-64 overflow-auto rounded-lg bg-muted/50 p-3 text-xs whitespace-pre-wrap break-words">
                        {result?.raw ??
                          "Run the comparison to generate this answer."}
                      </pre>
                    </div>
                    {score && (
                      <>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant={score.exact ? "default" : "outline"}>
                            Answer: {score.exact ? "Pass" : "Fail"}
                          </Badge>
                          <Badge variant={score.schema ? "default" : "outline"}>
                            JSON: {score.schema ? "Pass" : "Fail"}
                          </Badge>
                          {result?.truncated && (
                            <Badge variant="destructive">
                              Output truncated
                            </Badge>
                          )}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          Scored content:{" "}
                          <span className="font-mono">{score.answer}</span> ·{" "}
                          {score.contentExtraction}
                        </div>
                        <div className="text-xs text-muted-foreground">
                          {score.exact
                            ? "Matches a frozen accepted answer."
                            : "Does not exactly match an accepted answer; check meaning against the source."}
                        </div>
                        {score.unknownSourceIds.length > 0 && (
                          <Alert>
                            <AlertDescription>
                              Cites IDs absent from this answer’s context:{" "}
                              {score.unknownSourceIds.join(", ")}
                            </AlertDescription>
                          </Alert>
                        )}
                      </>
                    )}
                    {result && (
                      <div className="font-mono text-xs text-muted-foreground">
                        {result.seconds}s · {result.prompt_tokens} input /{" "}
                        {result.output_tokens} output tokens
                      </div>
                    )}
                  </section>
                </div>
                {result && (
                  <section className="grid gap-3 rounded-xl border p-3">
                    <div className="flex items-center gap-2 text-sm font-medium">
                      Human first, model judge second
                      <InfoTip>
                        0 wrong/unsupported; 1 partial/incomplete; 2 correct,
                        complete and supported by the original catalogue. The
                        judge uses the same model as the answerer, so its errors
                        are not independent. Agreement is a calibration signal,
                        not a truth guarantee.
                      </InfoTip>
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="mr-1 text-xs text-muted-foreground">
                        Your source-based rating
                      </span>
                      {[0, 1, 2].map((n) => (
                        <Button
                          key={n}
                          size="sm"
                          variant={human === n ? "default" : "outline"}
                          aria-label={`Rate answer ${n}`}
                          aria-pressed={human === n}
                          disabled={judge.busy}
                          onClick={() =>
                            setHumans((old) => ({ ...old, [key]: n }))
                          }
                        >
                          {n}
                        </Button>
                      ))}
                      <Button
                        size="sm"
                        disabled={busy || human === undefined}
                        onClick={judgeSelected}
                      >
                        Run real judge
                      </Button>
                    </div>
                    {human === undefined && (
                      <div className="text-xs text-muted-foreground">
                        Choose your rating before seeing the judge.
                      </div>
                    )}
                    {judgment && (
                      <>
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant="outline">
                            {parsed
                              ? `Model: ${parsed.rating}`
                              : "Invalid judge JSON"}
                          </Badge>
                          {parsed && human !== undefined && (
                            <Badge variant="secondary">
                              {parsed.rating === human ? "Agree" : "Disagree"}{" "}
                              with your rating
                            </Badge>
                          )}
                          <span className="text-xs text-muted-foreground">
                            {parsed?.reason}
                          </span>
                        </div>
                        <details className="text-xs">
                          <summary className="cursor-pointer">
                            Raw judge output
                          </summary>
                          <pre className="mt-2 whitespace-pre-wrap break-words">
                            {judgment.raw}
                          </pre>
                        </details>
                      </>
                    )}
                    {compared.length > 0 && (
                      <div className="text-xs text-muted-foreground">
                        Agreement on rated answers:{" "}
                        {ratio(
                          compared.filter((x) => x.agree).length,
                          compared.length,
                        )}
                      </div>
                    )}
                    {judge.busy && (
                      <div
                        role="status"
                        className="text-xs text-muted-foreground"
                      >
                        {judge.status}
                      </div>
                    )}
                    {judge.error && (
                      <Alert variant="destructive">
                        <AlertDescription>{judge.error}</AlertDescription>
                      </Alert>
                    )}
                  </section>
                )}
                {result && (
                  <details className="text-xs">
                    <summary className="cursor-pointer">
                      Actual prompts, model and run provenance
                    </summary>
                    <pre className="mt-3 max-h-80 overflow-auto whitespace-pre-wrap break-words">
                      {JSON.stringify(
                        { start, answer: result, judge: judgment },
                        null,
                        2,
                      )}
                    </pre>
                  </details>
                )}
              </CardContent>
            </Card>
          </DialogContent>
        </Dialog>
        <Collapsible>
          <Card>
            <CardHeader>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="justify-start">
                  Scoring rules, formulas and limits
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="grid gap-3 text-sm text-muted-foreground">
                <p>
                  Exact match = accepted answers / completed paired cases.
                  Extract the answer field (string or number) from JSON,
                  optionally inside one markdown fence; otherwise use the plain
                  text. Then normalize case and whitespace only. Raw JSON schema
                  is checked separately without removing fences. Fixed aliases
                  are shown in exported benchmark data; extra words can fail
                  even when meaning is right. JSON validity = outputs matching
                  exactly answer:string, sources:list[string] / paired cases.
                  Neither verifies citation support.
                </p>
                <p>
                  Δ = score(B) − score(A), in percentage points. Both use the
                  same completed case IDs. The unknown-fee case passes only on
                  UNKNOWN; inspect it separately because abstention on every
                  question can look safe while failing useful answers.
                </p>
                <p>
                  This is supplied-context QA, with known source passages. It
                  does not run a retriever, measure Recall@k or establish a
                  public benchmark score. The four fictional, visible
                  development questions are too small for a release decision.
                  Repeat on fresh, reserved tasks. Greedy decoding controls
                  sampling; hardware/runtime changes can still change outputs.
                </p>
                <p>
                  Judge agreement = equal human/model ratings / valid rated
                  pairs. Invalid judge JSON is excluded, never counted as
                  agreement. This same-model judge can reinforce the answerer’s
                  mistakes.
                </p>
                <p>
                  Runs use the cached{" "}
                  <a
                    className="underline"
                    href="https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct"
                  >
                    official Qwen model
                  </a>{" "}
                  through{" "}
                  <a
                    className="underline"
                    href="https://huggingface.co/docs/transformers/en/main_classes/text_generation"
                  >
                    Transformers generation
                  </a>
                  . No remote inference. The saved run button loads witnessed
                  outputs without generating new answers.
                </p>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
        <Collapsible>
          <Card>
            <CardHeader>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="justify-start">
                  Earlier metric calculator · authored examples
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent>
                <EvaluationFixtures />
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
      <Card className="order-1 xl:order-2 xl:sticky xl:top-4">
        <CardHeader>
          <CardTitle>Experiment</CardTitle>
        </CardHeader>
        <CardContent className="grid min-w-0 grid-cols-1 gap-4">
          <SelectField
            label="Evidence for B"
            value={settings.evidence}
            onValueChange={(v) =>
              setSettings((s) => ({
                ...s,
                evidence: v as EvaluationSettings["evidence"],
              }))
            }
            options={[
              { value: "full", label: "All 3 sources" },
              { value: "missing", label: "Remove Cedar source" },
            ]}
            info="A always gets no sources. Remove Cedar for B to test what happens when essential evidence is missing. Answers change only after Run."
          />
          <SelectField
            label="Requested output"
            value={settings.format}
            onValueChange={(v) =>
              setSettings((s) => ({
                ...s,
                format: v as EvaluationSettings["format"],
              }))
            }
            options={[
              { value: "json", label: "JSON answer + sources" },
              { value: "text", label: "Plain text answer" },
            ]}
            info="Applies to both A and B. The JSON integration contract is still measured, so plain text can be factually right yet unusable by the consumer."
          />
          <Button
            disabled={busy}
            onClick={() => {
              judge.clear();
              setHumans({});
              setJudgments({});
              run.run(settings);
            }}
          >
            <PlayIcon className="size-4" />
            Run A/B · 4 questions
          </Button>
          {busy && (
            <Button
              variant="outline"
              onClick={() => {
                run.cancel();
                judge.cancel();
              }}
            >
              <SquareIcon className="size-4" />
              Cancel inference
            </Button>
          )}
          <div
            role="status"
            aria-live="polite"
            className="text-xs text-muted-foreground"
          >
            {run.status || "Predict which questions will change."}
          </div>
          {changed && (
            <Badge variant="outline" className="whitespace-normal">
              Settings changed · results use previous settings
            </Badge>
          )}
          {run.error && (
            <Alert variant="destructive">
              <AlertDescription>{run.error}</AlertDescription>
            </Alert>
          )}
          <Button
            variant="outline"
            disabled={busy}
            onClick={() => {
              judge.clear();
              setHumans({});
              setJudgments({});
              run.loadRecording("/traces/evaluation-workshops.json");
            }}
          >
            Load recorded real run
          </Button>
          <Button
            variant="outline"
            disabled={!answers.length || busy}
            onClick={() =>
              downloadEvidence("evaluation-evidence.json", {
                kind: run.recording
                  ? "recorded-real-inference"
                  : "live-real-inference",
                events: run.events,
                humanRatings: humans,
                judgments,
                pairedSummary: pair,
              })
            }
          >
            <DownloadIcon className="size-4" />
            Export evidence
          </Button>
          <Button variant="ghost" onClick={reset}>
            <RotateCcwIcon className="size-4" />
            Reset experiment
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
