"use client"

import * as React from "react"
import { CopyIcon, RotateCcwIcon } from "lucide-react"
import { toast } from "sonner"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import {
  SelectField,
  SliderField,
  SwitchField,
} from "@/components/lab/controls"

const DEFAULTS = {
  blur: 0,
  brightness: 100,
  contrast: 100,
  saturate: 100,
  hue: 0,
  grayscale: 0,
}

type FilterState = typeof DEFAULTS

const PRESETS: {
  value: string
  label: string
  filters: FilterState
}[] = [
  { value: "identity", label: "Identity (no-op)", filters: DEFAULTS },
  {
    value: "soft-blur",
    label: "Gaussian blur",
    filters: { ...DEFAULTS, blur: 6, brightness: 104 },
  },
  {
    value: "sharpen",
    label: "Sharpen / local contrast",
    filters: { ...DEFAULTS, contrast: 160, saturate: 115 },
  },
  {
    value: "edge",
    label: "Edge emphasis",
    filters: { ...DEFAULTS, contrast: 200, grayscale: 100, brightness: 90 },
  },
  {
    value: "warm",
    label: "Warm grade",
    filters: { ...DEFAULTS, hue: -18, saturate: 140, brightness: 106 },
  },
  {
    value: "mono",
    label: "Monochrome",
    filters: { ...DEFAULTS, grayscale: 100, contrast: 120 },
  },
]

function SampleScene({ className }: { className?: string }) {
  const uid = React.useId().replace(/:/g, "")
  const sky = `sky-${uid}`
  const water = `water-${uid}`

  return (
    <svg
      viewBox="0 0 320 200"
      className={className}
      role="img"
      aria-label="Sample landscape used to preview filters"
    >
      <defs>
        <linearGradient id={sky} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#38bdf8" />
          <stop offset="100%" stopColor="#fef3c7" />
        </linearGradient>
        <linearGradient id={water} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#0ea5e9" />
          <stop offset="100%" stopColor="#0f172a" />
        </linearGradient>
      </defs>
      <rect width="320" height="200" fill={`url(#${sky})`} />
      <circle cx="248" cy="52" r="26" fill="#fde047" />
      <polygon points="0,140 70,70 140,140" fill="#334155" />
      <polygon points="90,140 175,50 260,140" fill="#1e293b" />
      <polygon points="180,140 260,80 320,140" fill="#475569" />
      <rect y="140" width="320" height="60" fill={`url(#${water})`} />
      <rect y="140" width="320" height="2" fill="#e2e8f0" opacity="0.6" />
      <g fill="#f8fafc" opacity="0.75">
        <circle cx="196" cy="154" r="3" />
        <circle cx="228" cy="166" r="2" />
        <circle cx="120" cy="172" r="2.5" />
        <circle cx="60" cy="160" r="2" />
      </g>
    </svg>
  )
}

function toFilterString(state: FilterState) {
  return [
    `blur(${state.blur}px)`,
    `brightness(${state.brightness}%)`,
    `contrast(${state.contrast}%)`,
    `saturate(${state.saturate}%)`,
    `hue-rotate(${state.hue}deg)`,
    `grayscale(${state.grayscale}%)`,
  ].join(" ")
}

const clamp = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

export function CvFilterStudio() {
  const [filters, setFilters] = React.useState<FilterState>(DEFAULTS)
  const [preset, setPreset] = React.useState("identity")
  const [compare, setCompare] = React.useState(false)

  const set = <K extends keyof FilterState>(key: K) => (value: FilterState[K]) =>
    setFilters((current) => ({ ...current, [key]: value }))

  const applyPreset = (value: string) => {
    setPreset(value)
    const match = PRESETS.find((item) => item.value === value)
    if (match) setFilters(match.filters)
  }

  const filterString = toFilterString(filters)

  const metrics = [
    {
      label: "Sharpness",
      value: clamp(100 - (filters.blur / 12) * 100, 0, 100),
      hint: "falls as blur grows",
    },
    {
      label: "Colourfulness",
      value: clamp(
        filters.saturate / 2 - (filters.grayscale / 100) * 50,
        0,
        100
      ),
      hint: "saturation minus greyscale",
    },
    {
      label: "Contrast",
      value: clamp(filters.contrast / 2, 0, 100),
      hint: "spread between dark and light",
    },
  ]

  const copyFilter = async () => {
    try {
      await navigator.clipboard.writeText(`filter: ${filterString};`)
      toast.success("Filter chain copied", {
        description: filterString,
      })
    } catch {
      toast.error("Clipboard unavailable", {
        description: "Copy the filter chain manually from the readout.",
      })
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_20rem]">
      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle>Live preview</CardTitle>
          <CardDescription>
            The same pixel grid, pushed through a CSS filter chain. Every slider
            maps to a stage a convolutional network would learn.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4">
          <div
            className={
              compare ? "grid gap-3 sm:grid-cols-2" : "grid gap-3"
            }
          >
            {compare ? (
              <figure className="grid gap-1.5">
                <div className="overflow-hidden rounded-lg ring-1 ring-foreground/10">
                  <SampleScene className="block h-auto w-full" />
                </div>
                <figcaption className="text-xs text-muted-foreground">
                  Original
                </figcaption>
              </figure>
            ) : null}
            <figure className="grid gap-1.5">
              <div
                className="overflow-hidden rounded-lg ring-1 ring-foreground/10"
                style={{ filter: filterString }}
              >
                <SampleScene className="block h-auto w-full" />
              </div>
              <figcaption className="text-xs text-muted-foreground">
                Filtered
              </figcaption>
            </figure>
          </div>
          <div className="grid gap-1.5">
            <span className="text-xs text-muted-foreground">
              Equivalent filter chain
            </span>
            <code className="block overflow-x-auto rounded-lg bg-muted px-3 py-2 font-mono text-xs">
              filter: {filterString};
            </code>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={copyFilter}>
              <CopyIcon data-icon="inline-start" />
              Copy CSS
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                setFilters(DEFAULTS)
                setPreset("identity")
              }}
            >
              <RotateCcwIcon data-icon="inline-start" />
              Reset
            </Button>
          </div>
        </CardContent>
      </Card>

      <div className="grid content-start gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Controls</CardTitle>
            <CardDescription>
              Start from a preset, then detune it.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-5">
            <SelectField
              label="Preset"
              value={preset}
              onValueChange={applyPreset}
              options={PRESETS.map(({ value, label }) => ({ value, label }))}
              hint="Presets are just slider positions — nothing is hidden."
            />
            <SliderField
              label="Blur"
              value={filters.blur}
              onValueChange={set("blur")}
              min={0}
              max={12}
              step={0.5}
              format={(value) => `${value}px`}
            />
            <SliderField
              label="Brightness"
              value={filters.brightness}
              onValueChange={set("brightness")}
              min={50}
              max={150}
              format={(value) => `${value}%`}
            />
            <SliderField
              label="Contrast"
              value={filters.contrast}
              onValueChange={set("contrast")}
              min={50}
              max={200}
              format={(value) => `${value}%`}
            />
            <SliderField
              label="Saturation"
              value={filters.saturate}
              onValueChange={set("saturate")}
              min={0}
              max={200}
              format={(value) => `${value}%`}
            />
            <SliderField
              label="Hue rotation"
              value={filters.hue}
              onValueChange={set("hue")}
              min={-180}
              max={180}
              format={(value) => `${value}°`}
            />
            <SliderField
              label="Greyscale"
              value={filters.grayscale}
              onValueChange={set("grayscale")}
              min={0}
              max={100}
              format={(value) => `${value}%`}
            />
            <SwitchField
              label="Side-by-side compare"
              hint="Keep the original next to the filtered image."
              checked={compare}
              onCheckedChange={setCompare}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Readout</CardTitle>
            <CardDescription>
              Normalised summaries of the current chain.
            </CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4">
            {metrics.map((metric) => (
              <div key={metric.label} className="grid gap-1.5">
                <div className="flex items-center justify-between gap-3 text-xs">
                  <span className="text-muted-foreground">
                    {metric.label}
                  </span>
                  <span className="font-mono tabular-nums">
                    {Math.round(metric.value)}
                  </span>
                </div>
                <Progress value={metric.value} />
                <span className="text-[11px] text-muted-foreground">
                  {metric.hint}
                </span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
