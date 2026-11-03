"use client"

import * as React from "react"
import { PauseIcon, PlayIcon, RotateCcwIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { SelectField, SliderField } from "@/components/lab/controls"
import { cn } from "@/lib/utils"

const SCHEDULES = [
  { value: "linear", label: "Linear" },
  { value: "cosine", label: "Cosine" },
  { value: "sigmoid", label: "Sigmoid" },
] as const

type ScheduleId = (typeof SCHEDULES)[number]["value"]

function noiseLevel(progress: number, schedule: ScheduleId) {
  switch (schedule) {
    case "cosine":
      return Math.cos((progress * Math.PI) / 2)
    case "sigmoid":
      return 1 / (1 + Math.exp((progress - 0.5) * 12))
    default:
      return 1 - progress
  }
}

function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

type Blob = { cx: number; cy: number; r: number; hue: number; opacity: number }

function buildBlobs(seed: number): Blob[] {
  const random = mulberry32(seed * 2654435761)
  return Array.from({ length: 7 }, () => ({
    cx: 18 + random() * 64,
    cy: 18 + random() * 64,
    r: 10 + random() * 22,
    hue: Math.round((seed * 37 + random() * 120) % 360),
    opacity: 0.45 + random() * 0.45,
  }))
}

function DiffusionFrame({
  seed,
  progress,
  guidance,
  className,
  withNoise = true,
}: {
  seed: number
  progress: number
  guidance: number
  className?: string
  withNoise?: boolean
}) {
  const uid = React.useId().replace(/:/g, "")
  const noiseId = `noise-${uid}`
  const backdropId = `backdrop-${uid}`
  const nodes = React.useMemo(() => buildBlobs(seed), [seed])

  const noise = noiseLevel(progress, "linear")
  const blur = noise * 16
  const baseFrequency = 0.04 + noise * 0.55
  const hue = (seed * 37) % 360

  return (
    <svg
      viewBox="0 0 100 100"
      className={className}
      role="img"
      aria-label={`Generated sample after ${Math.round(progress * 100)} percent of the denoising steps`}
    >
      <defs>
        <linearGradient id={backdropId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor={`hsl(${hue} 70% 22%)`} />
          <stop offset="100%" stopColor={`hsl(${(hue + 60) % 360} 65% 12%)`} />
        </linearGradient>
        <filter id={noiseId}>
          <feTurbulence
            type="fractalNoise"
            baseFrequency={baseFrequency}
            numOctaves={3}
            stitchTiles="stitch"
          />
          <feColorMatrix type="saturate" values="0" />
        </filter>
      </defs>
      <rect width="100" height="100" fill={`url(#${backdropId})`} />
      <g
        style={{
          filter: `blur(${blur}px) saturate(${Math.min(1 + guidance / 12, 2.2)}) contrast(${Math.min(1 + guidance / 25, 1.6)})`,
        }}
      >
        {nodes.map((node, index) => (
          <circle
            key={index}
            cx={node.cx}
            cy={node.cy}
            r={node.r * (0.6 + progress * 0.5)}
            fill={`hsl(${node.hue} 85% 62%)`}
            opacity={node.opacity * (0.25 + progress * 0.75)}
          />
        ))}
        <circle
          cx="50"
          cy="50"
          r={30 + progress * 6}
          fill="none"
          stroke={`hsl(${(hue + 180) % 360} 90% 75%)`}
          strokeWidth={0.8}
          opacity={0.35 * progress}
        />
      </g>
      {withNoise ? (
        <rect
          width="100"
          height="100"
          filter={`url(#${noiseId})`}
          opacity={noise * 0.85}
        />
      ) : null}
    </svg>
  )
}

export function GenaiDiffusionStudio() {
  const [seed, setSeed] = React.useState(12)
  const [steps, setSteps] = React.useState(24)
  const [step, setStep] = React.useState(24)
  const [guidance, setGuidance] = React.useState(7.5)
  const [schedule, setSchedule] = React.useState<ScheduleId>("cosine")
  const [playing, setPlaying] = React.useState(false)

  React.useEffect(() => {
    if (!playing || step >= steps) return
    const id = window.setTimeout(() => {
      setStep((current) => Math.min(current + 1, steps))
    }, 140)
    return () => window.clearTimeout(id)
  }, [playing, step, steps])

  const isPlaying = playing && step < steps
  const progress = steps === 0 ? 0 : step / steps
  const noise = noiseLevel(progress, schedule)
  const strip = [0, 0.25, 0.5, 0.75, 1]

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <div className="grid content-start gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Denoising trajectory</CardTitle>
            <CardDescription>
              Pure noise is gradually pushed back into structure. The overlay is
              an SVG turbulence field — as the steps advance its frequency and
              opacity drop while the shapes sharpen.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="mx-auto w-full max-w-md overflow-hidden rounded-xl ring-1 ring-foreground/10">
              <DiffusionFrame
                seed={seed}
                progress={progress}
                guidance={guidance}
                className="block h-auto w-full"
              />
            </div>
            <div className="flex flex-wrap items-center gap-2">
              <Button
                size="sm"
                onClick={() => {
                  if (step >= steps) {
                    setStep(0)
                    setPlaying(true)
                    return
                  }
                  setPlaying((current) => !current)
                }}
              >
                {isPlaying ? (
                  <PauseIcon data-icon="inline-start" />
                ) : (
                  <PlayIcon data-icon="inline-start" />
                )}
                {isPlaying ? "Pause" : "Run schedule"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={() => {
                  setPlaying(false)
                  setStep(0)
                }}
              >
                <RotateCcwIcon data-icon="inline-start" />
                Back to noise
              </Button>
              <span className="ml-auto font-mono text-xs text-muted-foreground tabular-nums">
                step {step}/{steps} · σ≈{noise.toFixed(2)}
              </span>
            </div>
            <div className="grid gap-2">
              <SliderField
                label="Scrub the trajectory"
                value={step}
                onValueChange={(value) => {
                  setPlaying(false)
                  setStep(value)
                }}
                min={0}
                max={steps}
                format={(value) => `${value}/${steps}`}
              />
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Step previews</CardTitle>
            <CardDescription>
              The same latent sampled at 0%, 25%, 50%, 75% and 100% of the
              schedule.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-2">
              {strip.map((value) => (
                <div key={value} className="grid gap-1.5">
                  <div
                    className={cn(
                      "overflow-hidden rounded-lg ring-1",
                      Math.abs(progress - value) < 0.04
                        ? "ring-2 ring-primary"
                        : "ring-foreground/10"
                    )}
                  >
                    <DiffusionFrame
                      seed={seed}
                      progress={value}
                      guidance={guidance}
                      withNoise={false}
                      className="block h-auto w-full"
                    />
                  </div>
                  <span className="text-center text-[11px] text-muted-foreground tabular-nums">
                    {Math.round(value * 100)}%
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid content-start gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Sampler settings</CardTitle>
            <CardDescription>
              Steps, guidance and the noise schedule shape the same latent in
              very different ways.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-5">
            <SliderField
              label="Seed"
              value={seed}
              onValueChange={setSeed}
              min={0}
              max={99}
              format={(value) => `#${value}`}
              hint="Changing the seed reshuffles the latent entirely."
            />
            <SliderField
              label="Steps"
              value={steps}
              onValueChange={(value) => {
                setSteps(value)
                setStep((current) => Math.min(current, value))
              }}
              min={4}
              max={40}
              format={(value) => `${value} steps`}
            />
            <SliderField
              label="Guidance scale"
              value={guidance}
              onValueChange={setGuidance}
              min={1}
              max={15}
              step={0.5}
              format={(value) => value.toFixed(1)}
              hint="Higher guidance boosts adherence — and saturation."
            />
            <SelectField
              label="Noise schedule"
              value={schedule}
              onValueChange={(value) => setSchedule(value as ScheduleId)}
              options={SCHEDULES.map(({ value, label }) => ({ value, label }))}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Noise schedule</CardTitle>
            <CardDescription>
              How much noise is left at each step of the {steps}-step run.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <div className="flex h-24 items-end gap-[2px]">
              {Array.from({ length: steps }, (_, index) => {
                const value = noiseLevel((index + 1) / steps, schedule)
                return (
                  <div
                    key={index}
                    className={cn(
                      "flex-1 rounded-sm",
                      index < step ? "bg-primary" : "bg-muted"
                    )}
                    style={{ height: `${Math.max(value * 100, 3)}%` }}
                  />
                )
              })}
            </div>
            <Separator />
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Remaining noise</span>
              <span className="font-mono tabular-nums">
                {(noise * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">Steps completed</span>
              <span className="font-mono tabular-nums">
                {step}/{steps}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
