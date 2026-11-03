"use client"

import * as React from "react"
import { animate } from "animejs"
import {
  ArrowRightIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
  ScissorsIcon,
  SparklesIcon,
  XIcon,
} from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { HeatCanvas } from "@/components/lab/heat-canvas"
import { useIsomorphicLayoutEffect } from "@/hooks/use-animation"
import type { ExplanationController } from "@/hooks/use-explanation"
import {
  EXPLANATION_STEPS,
  type DenseContributions,
  type ExplanationStage,
  type PoolSample,
  type ReluTransition,
} from "@/lib/explanation"
import { formatValue } from "@/lib/heatmap"
import { cn } from "@/lib/utils"

/** Heatmap hue convention: positive red, negative blue. */
const POSITIVE = "hsl(8 72% 46%)"
const NEGATIVE = "hsl(220 72% 46%)"

/**
 * The floating annotation near the focused layer. One sentence, a mono detail
 * line, and optional stage-specific visual — never a text panel.
 */
export function ExplanationCard({
  title,
  caption,
  detail,
  reducedMotion,
  children,
  className,
}: {
  title: string
  caption: string
  detail?: string
  reducedMotion: boolean
  children?: React.ReactNode
  className?: string
}) {
  const ref = React.useRef<HTMLDivElement>(null)

  useIsomorphicLayoutEffect(() => {
    const node = ref.current
    if (!node || reducedMotion) return
    const animation = animate(node, {
      opacity: [0, 1],
      y: [8, 0],
      scale: [0.98, 1],
      duration: 260,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [reducedMotion])

  return (
    <div
      ref={ref}
      data-guide-card=""
      className={cn(
        "pointer-events-auto grid gap-1.5 rounded-lg bg-card/95 p-2.5 ring-1 ring-foreground/15 backdrop-blur-sm",
        className
      )}
    >
      <span className="flex items-baseline gap-2">
        <span className="text-xs font-semibold">{title}</span>
        {detail ? (
          <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
            {detail}
          </span>
        ) : null}
      </span>
      <p className="text-xs leading-snug text-muted-foreground">{caption}</p>
      {children}
    </div>
  )
}

/**
 * ReLU beat: the strongest map before and after the gate, crossfading so the
 * negative responses visibly disappear. Both canvases are the real stages.
 */
export function ReluInset({
  transition,
  durationMs,
  playing,
  reducedMotion,
}: {
  transition: ReluTransition
  durationMs: number
  playing: boolean
  reducedMotion: boolean
}) {
  const preRef = React.useRef<HTMLSpanElement>(null)
  const postRef = React.useRef<HTMLSpanElement>(null)
  const animations = React.useRef<ReturnType<typeof animate>[]>([])
  /**
   * A beat opened by hand (stage dot, Previous/Next while paused) may not have
   * a running clock, so its visual still plays through once; pause/resume only
   * freezes beats that were entered mid-flight.
   */
  const [pausedAtMount] = React.useState(() => !playing)

  const preMap = transition.pre.maps[transition.channel]
  const postMap = transition.post.maps[transition.channel]

  useIsomorphicLayoutEffect(() => {
    const pre = preRef.current
    const post = postRef.current
    if (!pre || !post || reducedMotion || !preMap || !postMap) return
    const duration = Math.max(500, durationMs * 0.7)
    const created = [
      animate(pre, { opacity: [1, 0.25], duration, ease: "inOut(2)" }),
      animate(post, { opacity: [0, 1], duration, ease: "out(3)" }),
    ]
    animations.current = created
    return () => {
      animations.current = []
      for (const animation of created) animation.revert()
    }
  }, [transition, durationMs, reducedMotion, preMap, postMap])

  React.useEffect(() => {
    if (pausedAtMount) return
    for (const animation of animations.current) {
      if (playing) animation.play()
      else animation.pause()
    }
  }, [playing, pausedAtMount])

  return (
    <div data-relu-inset="" className="grid gap-1.5">
      <div className="flex items-center gap-2">
        <span
          ref={preRef}
          data-relu-pre=""
          className="relative block size-20 overflow-hidden rounded-[3px] ring-1 ring-foreground/10"
          style={{ opacity: reducedMotion ? 0.3 : 1 }}
        >
          {preMap ? (
            <HeatCanvas map={preMap} bound={transition.pre.stats.bound} />
          ) : null}
        </span>
        <ArrowRightIcon className="size-3.5 shrink-0 text-muted-foreground" />
        <span
          ref={postRef}
          data-relu-post=""
          className="relative block size-20 overflow-hidden rounded-[3px] ring-1 ring-foreground/10"
          style={{ opacity: reducedMotion ? 1 : 0 }}
        >
          {postMap ? (
            <HeatCanvas map={postMap} bound={transition.post.stats.bound} />
          ) : null}
        </span>
      </div>
      <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
        {transition.pre.name} → {transition.post.name} ·{" "}
        {Math.round(transition.zeroFraction * 100)}% of map{" "}
        {transition.channel + 1} zeroed
      </span>
    </div>
  )
}

const POOL_FRAME = 112

/**
 * Pooling beat: a real window glides to the region that won the max, then the
 * map crossfades to the pooled result — same activation, lower resolution.
 */
export function PoolInset({
  sample,
  durationMs,
  playing,
  reducedMotion,
}: {
  sample: PoolSample
  durationMs: number
  playing: boolean
  reducedMotion: boolean
}) {
  const windowRef = React.useRef<HTMLSpanElement>(null)
  const inputRef = React.useRef<HTMLSpanElement>(null)
  const outputRef = React.useRef<HTMLSpanElement>(null)
  const maxRef = React.useRef<HTMLSpanElement>(null)
  const animations = React.useRef<ReturnType<typeof animate>[]>([])
  /** See ReluInset: a hand-opened beat plays its visual once. */
  const [pausedAtMount] = React.useState(() => !playing)

  const inputMap = sample.input.maps[sample.channel]
  const outputMap = sample.output.maps[sample.channel]
  const cell = POOL_FRAME / Math.max(1, sample.input.rows)
  const windowSize = sample.window * cell

  useIsomorphicLayoutEffect(() => {
    const windowNode = windowRef.current
    const inputNode = inputRef.current
    const outputNode = outputRef.current
    const maxNode = maxRef.current
    if (!windowNode || !inputNode || !outputNode || !maxNode || reducedMotion) return

    const created = [
      animate(windowNode, {
        x: sample.col * cell,
        y: sample.row * cell,
        duration: durationMs * 0.42,
        ease: "out(4)",
      }),
      animate(outputNode, {
        opacity: [0, 1],
        duration: durationMs * 0.3,
        delay: durationMs * 0.52,
        ease: "out(3)",
      }),
      animate(inputNode, {
        opacity: [1, 0.3],
        duration: durationMs * 0.3,
        delay: durationMs * 0.52,
        ease: "out(3)",
      }),
      animate(maxNode, {
        opacity: [0, 1],
        duration: 220,
        delay: durationMs * 0.4,
        ease: "out(3)",
      }),
    ]
    animations.current = created
    return () => {
      animations.current = []
      for (const animation of created) animation.revert()
    }
  }, [sample, durationMs, reducedMotion, cell])

  React.useEffect(() => {
    if (pausedAtMount) return
    for (const animation of animations.current) {
      if (playing) animation.play()
      else animation.pause()
    }
  }, [playing, pausedAtMount])

  const maxLabel = (
    <span
      ref={maxRef}
      data-pool-max=""
      className="font-mono text-[10px] tabular-nums"
      style={{ color: POSITIVE, opacity: reducedMotion ? 1 : 0 }}
    >
      max {formatValue(sample.max)}
    </span>
  )

  return (
    <div data-pool-inset="" className="grid gap-1.5">
      <div className="flex items-center gap-2">
        <span
          ref={inputRef}
          data-pool-input=""
          className="relative block shrink-0 overflow-hidden rounded-[3px] ring-1 ring-foreground/10"
          style={{ width: POOL_FRAME, height: POOL_FRAME, opacity: reducedMotion ? 0.3 : 1 }}
        >
          {inputMap ? (
            <HeatCanvas map={inputMap} bound={sample.input.stats.bound} />
          ) : null}
          <span
            ref={windowRef}
            data-pool-window=""
            aria-hidden
            className="pointer-events-none absolute top-0 left-0 rounded-[2px] bg-primary/20 ring-2 ring-primary"
            style={{
              width: windowSize,
              height: windowSize,
              transform: reducedMotion
                ? `translate(${sample.col * cell}px, ${sample.row * cell}px)`
                : "translate(0px, 0px)",
            }}
          />
          <span
            ref={outputRef}
            data-pool-output=""
            className="absolute inset-0"
            style={{ opacity: reducedMotion ? 1 : 0 }}
          >
            {outputMap ? (
              <HeatCanvas map={outputMap} bound={sample.output.stats.bound} />
            ) : null}
            <span
              data-pool-winner=""
              aria-hidden
              className="absolute rounded-[2px] ring-2 ring-primary"
              style={{
                left: `${(sample.outputCol / sample.output.cols) * 100}%`,
                top: `${(sample.outputRow / sample.output.rows) * 100}%`,
                width: `${100 / sample.output.cols}%`,
                height: `${100 / sample.output.rows}%`,
              }}
            />
          </span>
        </span>
        <span className="grid content-center gap-1">
          <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
            {sample.input.rows}×{sample.input.cols} → {sample.output.rows}×
            {sample.output.cols}
          </span>
          {maxLabel}
        </span>
      </div>
      <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
        {sample.window}×{sample.window} window · map {sample.channel + 1}
      </span>
    </div>
  )
}

/** The features pushing the predicted class score up (red) and down (blue). */
export function DenseContributionChips({
  contributions,
}: {
  contributions: DenseContributions
}) {
  const chips = [
    ...contributions.positive.map((entry) => ({ ...entry, positive: true })),
    ...contributions.negative.map((entry) => ({ ...entry, positive: false })),
  ]
  return (
    <span className="flex flex-wrap gap-1">
      {chips.map((entry) => (
        <span
          key={entry.index}
          data-contribution={entry.index}
          title={`feature ${entry.index + 1} of 256 · pooled ${entry.row},${entry.col} of map ${
            entry.channel + 1
          }`}
          className="rounded bg-muted/60 px-1.5 py-0.5 font-mono text-[10px] tabular-nums ring-1 ring-foreground/10"
          style={{ color: entry.positive ? POSITIVE : NEGATIVE }}
        >
          f{entry.index + 1} {entry.positive ? "+" : ""}
          {formatValue(entry.value)}
        </span>
      ))}
    </span>
  )
}

export function ExplanationCta({ onTry }: { onTry: () => void }) {
  return (
    <span className="flex flex-wrap items-center gap-2 pt-0.5">
      <span className="text-[11px] text-muted-foreground">
        What happens if parts of the network are changed?
      </span>
      <Button size="xs" variant="secondary" onClick={onTry}>
        <ScissorsIcon data-icon="inline-start" />
        Try Network Surgery
      </Button>
    </span>
  )
}

export type BeamLine = {
  x1: number
  y1: number
  x2: number
  y2: number
}

/** Dashed connections from the previous layer's strongest map to this one's. */
export function GuideBeams({
  lines,
  width,
  height,
  visible,
}: {
  lines: readonly BeamLine[]
  width: number
  height: number
  visible: boolean
}) {
  return (
    <svg
      aria-hidden
      data-guide-beams=""
      viewBox={`0 0 ${width} ${height}`}
      className={cn(
        "pointer-events-none absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none",
        visible ? "opacity-100" : "opacity-0"
      )}
    >
      {lines.map((line, index) => (
        <line
          key={index}
          x1={line.x1}
          y1={line.y1}
          x2={line.x2}
          y2={line.y2}
          stroke={POSITIVE}
          strokeOpacity={0.5}
          strokeWidth={1.5}
          strokeLinecap="round"
          strokeDasharray="5 4"
          className="motion-safe:animate-[guide-dash_1.1s_linear_infinite]"
        />
      ))}
    </svg>
  )
}

/**
 * The guided-mode transport: stage dots, count, play/pause, step, restart and
 * exit, with a per-frame progress bar fed by the controller subscription.
 */
export function ExplanationControls({
  controller,
  onExit,
}: {
  controller: ExplanationController<ExplanationStage>
  onExit: () => void
}) {
  const { subscribe, index, total, state, step } = controller
  const barRef = React.useRef<HTMLSpanElement>(null)

  useIsomorphicLayoutEffect(() => {
    return subscribe((progress) => {
      const node = barRef.current
      if (node) node.style.transform = `scaleX(${Math.max(0, Math.min(1, progress))})`
    })
  }, [subscribe])

  const atEnd = index >= total - 1

  return (
    <section
      aria-label="Guided explanation"
      data-explanation-controls=""
      className="grid gap-2 rounded-lg bg-muted/40 p-2.5 ring-1 ring-foreground/10"
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <span className="flex items-center gap-1.5 text-xs font-medium">
          <SparklesIcon className="size-3.5 text-primary" />
          Explaining prediction
        </span>
        <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
          {index + 1} / {total}
        </span>
        <ol className="flex flex-wrap items-center gap-1.5" aria-label="Stages">
          {EXPLANATION_STEPS.map((entry, entryIndex) => (
            <li key={entry.id} className="flex items-center">
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() => controller.select(entryIndex)}
                    aria-label={`Go to ${entry.title}`}
                    aria-current={entryIndex === index ? "step" : undefined}
                    className={cn(
                      "size-2.5 rounded-full transition-colors",
                      entryIndex < index
                        ? "bg-primary"
                        : entryIndex === index
                          ? "bg-primary ring-2 ring-primary/30"
                          : "bg-muted-foreground/25 hover:bg-muted-foreground/50"
                    )}
                  />
                </TooltipTrigger>
                <TooltipContent side="top" className="text-xs">
                  {entry.title}
                </TooltipContent>
              </Tooltip>
            </li>
          ))}
        </ol>
        <span className="min-w-0 truncate text-xs text-muted-foreground">
          {step?.title}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Previous stage"
            disabled={index === 0}
            onClick={controller.previous}
          >
            <ChevronLeftIcon />
          </Button>
          {state.playing ? (
            <Button size="sm" variant="outline" onClick={controller.pause}>
              <PauseIcon data-icon="inline-start" />
              Pause
            </Button>
          ) : (
            <Button
              size="sm"
              variant="outline"
              onClick={controller.resume}
              disabled={atEnd}
            >
              <PlayIcon data-icon="inline-start" />
              Resume
            </Button>
          )}
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Next stage"
            disabled={atEnd}
            onClick={controller.next}
          >
            <ChevronRightIcon />
          </Button>
          <Button
            size="icon-sm"
            variant="ghost"
            aria-label="Restart explanation"
            onClick={controller.restart}
          >
            <RotateCcwIcon />
          </Button>
          <Button size="sm" variant="ghost" onClick={onExit}>
            <XIcon data-icon="inline-start" />
            Exit
          </Button>
        </div>
      </div>
      <span
        aria-hidden
        className="h-0.5 w-full overflow-hidden rounded-full bg-foreground/10"
      >
        <span
          ref={barRef}
          className="block h-full w-full origin-left rounded-full bg-primary"
          style={{ transform: "scaleX(0)" }}
        />
      </span>
    </section>
  )
}
