"use client";

import { useId, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { InfoTip } from "@/components/lab/controls";
import type { Chunk, RetrievalResult } from "@/lib/day2-protocol";
import { compareVectors } from "@/lib/retrieval-geometry";
import { heatColor } from "@/lib/heatmap";
import { cn } from "@/lib/utils";
import { EvidenceDetails, evidenceGridKey } from "./day2-evidence";

export function RetrievalSimilarity({
  result,
  passage,
  onSelect,
  sourceTitle,
}: {
  result: RetrievalResult;
  passage: Chunk;
  onSelect: (chunk: Chunk) => void;
  sourceTitle: (source: string) => string;
}) {
  const rows = [...result.chunks].sort(
    (a, b) => b.cosine - a.cosine || a.id.localeCompare(b.id),
  );
  const geometry = compareVectors(result.query_vector, passage.vector);
  return (
    <div className="grid items-start gap-4 lg:grid-cols-2">
      <div className="rounded-xl border bg-muted/20 p-4 lg:col-span-2">
        <div className="mb-2 flex items-center gap-2 text-xs font-medium text-muted-foreground">
          Question → {result.query_vector.length} numbers
          <InfoTip>
            Both the question and each passage become a vector. Search compares
            all coordinates, not shared keywords. Higher cosine means a smaller
            angle to the question; it does not prove factual support.
          </InfoTip>
        </div>
        <p className="text-base font-medium">{result.settings.query}</p>
      </div>
      <div className="overflow-hidden rounded-xl border">
        <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/30 px-4 py-3 text-xs">
          <b>Passages ordered by similarity to the question</b>
          <span className="text-muted-foreground">
            Click a passage to compare vectors
          </span>
        </div>
        <div
          className="max-h-[26rem] overflow-y-auto"
          aria-label="Passage similarity ranking"
        >
          {rows.map((chunk, i) => {
            const candidate = result.candidates.find((c) => c.id === chunk.id);
            const active = chunk.id === passage.id;
            const position = (Math.max(-1, Math.min(1, chunk.cosine)) + 1) * 50;
            return (
              <button
                type="button"
                key={chunk.id}
                aria-pressed={active}
                aria-label={`Compare ${sourceTitle(chunk.source)} words ${chunk.start} to ${chunk.end}, similarity ${chunk.cosine.toFixed(4)}`}
                onClick={() => onSelect(chunk)}
                className={cn(
                  "grid w-full gap-2 border-b p-3 text-left last:border-b-0 hover:bg-muted/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-ring",
                  active && "bg-primary/5 border-l-4 border-l-primary",
                )}
              >
                <div className="min-w-0">
                  <div className="flex flex-wrap items-center gap-2 text-sm font-medium">
                    <span className="font-mono text-xs text-muted-foreground">
                      #{i + 1}
                    </span>
                    {sourceTitle(chunk.source)}
                    <span className="font-mono text-xs text-muted-foreground">
                      [{chunk.start},{chunk.end})
                    </span>
                    <Badge variant={candidate ? "secondary" : "outline"}>
                      {candidate ? "Retrieved" : "Outside top-k"}
                    </Badge>
                    {active && <Badge variant="outline">Comparing</Badge>}
                  </div>
                  <p className="mt-1 line-clamp-1 text-xs leading-relaxed text-muted-foreground">
                    {chunk.text}
                  </p>
                </div>
                <div className="grid content-center gap-1.5">
                  <span className="text-right font-mono text-xs">
                    Cosine {chunk.cosine.toFixed(4)}
                  </span>
                  <div
                    className="relative h-2 rounded-full bg-muted"
                    aria-hidden="true"
                  >
                    <span className="absolute inset-y-0 left-1/2 border-l border-foreground/30" />
                    <span
                      className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-background bg-primary"
                      style={{ left: `${position}%` }}
                    />
                  </div>
                  <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
                    <span>−1</span>
                    <span>0</span>
                    <span>+1</span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
      <div
        className="grid gap-4 rounded-xl border bg-muted/20 p-4"
        aria-label="Selected vector comparison"
      >
        <div>
          <div className="flex items-center gap-2 text-sm font-medium">
            Angle to the question
            <InfoTip>
              This is the plane spanned by this pair of real vectors. Their
              angle is computed in all {result.query_vector.length} dimensions.
              Each pair has its own plane: this is not a shared 2D map of
              passages. Smaller angle means higher cosine similarity.
            </InfoTip>
          </div>
          <VectorAngle cosine={geometry.cosine} angle={geometry.angle} />
          <div className="flex justify-between gap-2 font-mono text-xs">
            <span>Angle {geometry.angle.toFixed(1)}°</span>
            <span>Cosine {passage.cosine.toFixed(4)}</span>
          </div>
        </div>
        <div className="grid content-start gap-2">
          <span className="text-xs text-muted-foreground">Comparing with</span>
          <b className="text-sm">
            {sourceTitle(passage.source)} · words [{passage.start},{passage.end}
            )
          </b>
          <p className="text-sm leading-relaxed">{passage.text}</p>
          <span className="text-xs text-muted-foreground">
            Smaller angle → higher similarity → earlier search rank
          </span>
        </div>
      </div>
      <div className="lg:col-span-2">
        <VectorCoordinates
          key={passage.id}
          query={result.query_vector}
          passage={passage}
        />
      </div>
    </div>
  );
}

function VectorAngle({ cosine, angle }: { cosine: number; angle: number }) {
  const marker = useId().replace(/:/g, "");
  const sine = Math.sqrt(Math.max(0, 1 - cosine * cosine));
  const x = 150 + 105 * cosine,
    y = 145 - 105 * sine;
  const arcX = 150 + 35 * cosine,
    arcY = 145 - 35 * sine;
  return (
    <svg
      viewBox="0 0 300 190"
      role="img"
      aria-label={`Question and passage unit vectors, angle ${angle.toFixed(1)} degrees`}
      className="mx-auto max-h-52 w-full"
    >
      <defs>
        <marker
          id={marker}
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="5"
          markerHeight="5"
          orient="auto-start-reverse"
        >
          <path d="M 0 0 L 10 5 L 0 10 z" className="fill-foreground" />
        </marker>
      </defs>
      <path
        d="M 45 145 A 105 105 0 0 1 255 145"
        fill="none"
        className="stroke-border"
        strokeDasharray="3 4"
      />
      <line
        x1="150"
        y1="145"
        x2="255"
        y2="145"
        className="stroke-foreground"
        strokeWidth="2.5"
        markerEnd={`url(#${marker})`}
      />
      <line
        x1="150"
        y1="145"
        x2={x}
        y2={y}
        className="stroke-foreground"
        strokeWidth="2.5"
        strokeDasharray="6 3"
        markerEnd={`url(#${marker})`}
      />
      {angle > 0.01 && (
        <path
          d={`M 185 145 A 35 35 0 0 0 ${arcX} ${arcY}`}
          fill="none"
          className="stroke-muted-foreground"
        />
      )}
      <circle cx="150" cy="145" r="3" className="fill-foreground" />
      <text
        x="255"
        y="168"
        textAnchor="end"
        className="fill-foreground"
        fontSize="12"
      >
        Question q
      </text>
      <text
        x={x}
        y={y - 10}
        textAnchor="middle"
        className="fill-foreground"
        fontSize="12"
      >
        Passage d
      </text>
      <text x="12" y="185" className="fill-muted-foreground" fontSize="10">
        Pair plane · both vectors have length 1
      </text>
    </svg>
  );
}

function VectorCoordinates({
  query,
  passage,
}: {
  query: number[];
  passage: Chunk;
}) {
  const [dimension, setDimension] = useState(0);
  const comparison = compareVectors(query, passage.vector);
  const bound = Math.max(
    ...query.map(Math.abs),
    ...passage.vector.map(Math.abs),
  );
  return (
    <EvidenceDetails title="How the vectors produce this score">
      <div className="flex items-center gap-2 text-xs">
        Same column = same coordinate · common colour scale
        <InfoTip>
          Red is positive; blue is negative. A matching sign gives a positive
          product; opposite signs give a negative product. A coordinate has no
          individual named meaning. All coordinates contribute, including those
          not clicked.
        </InfoTip>
      </div>
      {[
        { label: "Question q", vector: query },
        { label: "Passage d", vector: passage.vector },
      ].map(({ label, vector }) => (
        <div key={label} className="grid gap-2">
          <b className="text-xs">
            {label} · {vector.length} coordinates
          </b>
          <div
            className="grid grid-cols-[repeat(32,minmax(0,1fr))] gap-0.5"
            aria-label={`${label} vector coordinates`}
            onKeyDown={(e) => evidenceGridKey(e, 32)}
          >
            {vector.map((value, i) => (
              <button
                type="button"
                key={i}
                tabIndex={i === dimension ? 0 : -1}
                aria-label={`${label} coordinate ${i}: ${value.toFixed(6)}`}
                aria-pressed={dimension === i}
                title={`${label}[${i}] = ${value.toFixed(6)}`}
                onClick={() => setDimension(i)}
                className={cn(
                  "h-3 rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  dimension === i &&
                    "ring-2 ring-foreground ring-offset-1 ring-offset-background",
                )}
                style={{ backgroundColor: heatColor(value, bound) }}
              />
            ))}
          </div>
        </div>
      ))}
      <div
        className="grid gap-2 rounded-lg border p-3 font-mono text-xs"
        aria-live="polite"
      >
        <span>
          Coordinate {dimension}: q = {query[dimension].toFixed(6)} · d ={" "}
          {passage.vector[dimension].toFixed(6)}
        </span>
        <span>
          Contribution qᵢdᵢ / (‖q‖‖d‖) ={" "}
          {comparison.contributions[dimension].toFixed(8)}
        </span>
        <b>
          Sum of all {query.length} contributions ={" "}
          {comparison.cosine.toFixed(6)}
        </b>
      </div>
      <div className="flex flex-wrap gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() =>
            setDimension(
              comparison.contributions.indexOf(
                Math.max(...comparison.contributions),
              ),
            )
          }
        >
          Largest positive contribution
        </Button>
        <Button
          size="sm"
          variant="outline"
          onClick={() =>
            setDimension(
              comparison.contributions.indexOf(
                Math.min(...comparison.contributions),
              ),
            )
          }
        >
          Smallest contribution
        </Button>
      </div>
      <p className="text-xs">
        cos(q,d) = Σ qᵢdᵢ / (‖q‖‖d‖). q is the question vector; d is this
        passage vector. Lengths are approximately 1 after normalization. Angle =
        arccos(cosine). Original inputs: {passage.model_tokens} passage model
        tokens; the encoder uses at most 256.
        {passage.model_tokens > 256
          ? " This passage was truncated by the encoder."
          : ""}
      </p>
    </EvidenceDetails>
  );
}
