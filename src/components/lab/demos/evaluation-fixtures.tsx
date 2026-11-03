"use client";
import { useState } from "react";
import {
  EVALUATION_CASES,
  evaluateCase,
  summarizeEvaluation,
} from "@/lib/evaluation";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { InfoTip, SliderField } from "@/components/lab/controls";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";

const percent = (n: number | null) =>
  n === null ? "N/A" : `${(100 * n).toFixed(1)}%`;
export function EvaluationFixtures() {
  const [configuration, setConfiguration] = useState("baseline");
  const [k, setK] = useState(2);
  const [selected, setSelected] = useState("t3");
  const [ratings, setRatings] = useState<Record<string, number>>({});
  const rows = EVALUATION_CASES[configuration],
    summary = summarizeEvaluation(rows, k);
  const row = rows.find((r) => r.id === selected)!,
    score = evaluateCase(row, k);
  const human = ratings[`${configuration}/${selected}`] ?? row.human;
  const agreement =
    rows.filter(
      (r) => (ratings[`${configuration}/${r.id}`] ?? r.human) === r.judge,
    ).length / rows.length;
  function reset() {
    setConfiguration("baseline");
    setK(2);
    setSelected("t3");
    setRatings({});
  }
  const exportHref =
    "data:application/json;charset=utf-8," +
    encodeURIComponent(
      JSON.stringify(
        {
          kind: "authored-evaluation-fixtures",
          configuration,
          k,
          rows,
          summary,
          humanRatings: ratings,
          agreement,
        },
        null,
        2,
      ),
    );
  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid gap-4">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              Evaluation: answer quality and retrieval
              <InfoTip label="About evaluation fixtures">
                Authored outputs and illustrative judge ratings; no model runs
                here. Changing k rescores fixed rankings and keeps answers
                unchanged. Actual model judging is in the Block 4 marimo
                notebook.
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-5">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              {[
                ["Exact match", percent(summary.exact)],
                ["Schema validity", percent(summary.schema)],
                ["Recall@" + k, percent(summary.recall)],
                ["Agreement", percent(agreement)],
              ].map(([name, value]) => (
                <div key={name} className="rounded-lg bg-muted/50 p-3">
                  <div className="text-xs text-muted-foreground">{name}</div>
                  <div className="font-mono text-xl tabular-nums">{value}</div>
                </div>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">N = {summary.n}</Badge>
              <Badge variant="outline">Recall M = {summary.recallN}</Badge>
              <Badge variant="secondary">Authored examples</Badge>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr>
                    <th className="p-2">Case</th>
                    <th className="p-2">Match</th>
                    <th className="p-2">Schema</th>
                    <th className="p-2">Recall</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const s = evaluateCase(r, k);
                    return (
                      <tr
                        key={r.id}
                        className={
                          selected === r.id
                            ? "bg-primary/10 ring-1 ring-primary/40"
                            : ""
                        }
                      >
                        <td className="p-2">
                          <Button
                            variant="ghost"
                            size="sm"
                            aria-label={`Inspect ${r.id}`}
                            aria-pressed={selected === r.id}
                            onClick={() => setSelected(r.id)}
                          >
                            {r.id} · {r.question}
                          </Button>
                        </td>
                        <td className="p-2 font-mono">
                          {s.exact ? "Pass" : "Fail"}
                        </td>
                        <td className="p-2 font-mono">
                          {s.schema ? "Pass" : "Fail"}
                        </td>
                        <td className="p-2 font-mono">{percent(s.recall)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {selected} · Evidence inspector
              <InfoTip label="About source support">
                Schema validity checks the declared output shape. Exact match
                uses lowercase and whitespace normalization for these ASCII
                fixtures. Neither checks whether each citation supports its
                claim.
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <div className="text-xs text-muted-foreground">
                  Original source
                </div>
                <div className="mt-2 rounded-lg bg-muted/50 p-3">
                  {row.source}
                </div>
              </div>
              <div>
                <div className="text-xs text-muted-foreground">
                  Raw candidate
                </div>
                <pre className="mt-2 overflow-auto rounded-lg bg-muted/50 p-3 text-xs whitespace-pre-wrap break-all">
                  {row.raw}
                </pre>
              </div>
            </div>
            <div className="text-sm">
              Reference: <span className="font-mono">{row.reference}</span> ·
              Parsed answer: <span className="font-mono">{score.answer}</span>
            </div>
            <div className="flex flex-wrap gap-2">
              {row.ranked.map((id, i) => (
                <Badge key={i} variant={i < k ? "default" : "outline"}>
                  {i + 1}: {id}{" "}
                  {row.relevant.includes(id) ? "✓ relevant" : "× distractor"}{" "}
                  {i < k ? "· selected" : ""}
                </Badge>
              ))}
            </div>
            <div className="font-mono text-sm">
              Recall:{" "}
              {row.relevant.length
                ? `${new Set(row.ranked.slice(0, k).filter((id) => row.relevant.includes(id))).size} / ${new Set(row.relevant).size}`
                : "undefined · no relevant chunks"}
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-sm">
                Illustrative judge: {row.judge} · Human:
              </span>
              {[0, 1, 2].map((n) => (
                <Button
                  key={n}
                  size="sm"
                  variant={human === n ? "default" : "outline"}
                  aria-label={`Human rating ${n}`}
                  aria-pressed={human === n}
                  onClick={() =>
                    setRatings({
                      ...ratings,
                      [`${configuration}/${selected}`]: n,
                    })
                  }
                >
                  {n}
                </Button>
              ))}
              <Badge variant="outline">
                {human === row.judge ? "Agree" : "Disagree"}
              </Badge>
            </div>
          </CardContent>
        </Card>
        <Collapsible>
          <Card>
            <CardHeader>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" className="justify-start">
                  Formulas, rubric and limits
                </Button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="grid gap-3 text-sm text-muted-foreground">
                <p>
                  Exact match = matching normalized answers / N. Schema validity
                  = outputs fitting exactly answer:string and
                  sources:list[string] / N.
                </p>
                <p>
                  Recall@k = |relevant ∩ first k IDs| / |relevant|. Average over
                  M answerable cases only. Here N=4, M=3; absent answers require
                  separate abstention checks.
                </p>
                <p>
                  Agreement = equal judge/human ratings / valid paired ratings.
                  Illustrative rubric: 0 wrong/unsupported; 1 partial; 2
                  correct, complete, supported. Blind identity and check
                  disagreements against source evidence.
                </p>
                <p>
                  Four authored cases cannot establish population quality.
                  Higher recall here does not generate a new answer. The changed
                  configuration intentionally sacrifices one schema pass.
                </p>
                <p>
                  Sources:{" "}
                  <a
                    className="underline"
                    href="https://arxiv.org/abs/2009.03300"
                  >
                    MMLU
                  </a>{" "}
                  ·{" "}
                  <a
                    className="underline"
                    href="https://arxiv.org/abs/2306.05685"
                  >
                    LLM-as-judge
                  </a>
                </p>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
      <Card>
        <CardHeader>
          <CardTitle>Experiment</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-5">
          <div className="grid gap-2">
            <div className="text-xs text-muted-foreground">Configuration</div>
            {["baseline", "changed"].map((c) => (
              <Button
                key={c}
                variant={configuration === c ? "default" : "outline"}
                aria-pressed={configuration === c}
                onClick={() => setConfiguration(c)}
              >
                {c}
              </Button>
            ))}
          </div>
          <SliderField
            label="Retrieved k"
            value={k}
            min={1}
            max={3}
            step={1}
            onValueChange={setK}
            info="Only the metric's cutoff changes. Saved candidate answers remain fixed."
          />
          <Button variant="outline" onClick={reset}>
            Reset experiment
          </Button>
          <Button asChild>
            <a href={exportHref} download="evaluation-lab.json">
              Export evidence
            </a>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
