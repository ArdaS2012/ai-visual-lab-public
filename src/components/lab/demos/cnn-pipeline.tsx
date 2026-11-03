"use client"

import * as React from "react"
import Image from "next/image"
import { animate } from "animejs"
import {
  ArrowRightIcon,
  ChevronDownIcon,
  DicesIcon,
  LightbulbIcon,
  PauseIcon,
  PlayIcon,
  RotateCcwIcon,
  SparklesIcon,
  StepForwardIcon,
} from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import { Separator } from "@/components/ui/separator"
import {
  ChoiceField,
  ControlGroup,
  InfoTip,
  SelectField,
  SliderField,
  SwitchField,
} from "@/components/lab/controls"
import {
  useIsomorphicLayoutEffect,
  usePrefersReducedMotion,
} from "@/hooks/use-animation"
import {
  buildNetwork,
  forwardPass,
  layerNames,
  receptiveFieldSizes,
  traceReceptiveField,
  totalParams,
  type FeatureMap,
  type Filter,
  type RfRegion,
  type StageResult,
  type Tensor,
} from "@/lib/cnn"
import {
  CNN_IMAGES,
  DEFAULT_CNN_IMAGE_ID,
  findCnnImage,
  imageToGrid,
  loadImageElement,
} from "@/lib/cnn-images"
import {
  buildScene,
  CHANNEL_COLOR,
  CHANNEL_LABEL,
  toChannelFirst,
} from "@/lib/cnn-scenes"
import {
  formatValue,
  heatRgb,
  heatTextColor,
  weightStyle,
} from "@/lib/heatmap"
import { cn } from "@/lib/utils"

const CHANNEL_OPTIONS = [
  { value: "grayscale", label: "Grayscale" },
  { value: "rgb", label: "RGB" },
] as const

type InputSource = "synthetic" | "photo"

const SOURCE_OPTIONS: { value: InputSource; label: string }[] = [
  { value: "synthetic", label: "Synthetic" },
  { value: "photo", label: "Photo" },
]

const SCENE_OPTIONS = [
  { value: "gradient", label: "Gradient" },
  { value: "edge", label: "Hard edge" },
  { value: "diagonal", label: "Diagonal" },
  { value: "cross", label: "Cross" },
  { value: "noise", label: "Noise" },
  { value: "smiley", label: "Smiley" },
]

const PHOTO_OPTIONS = CNN_IMAGES.map((image) => ({
  value: image.id,
  label: image.label,
}))

const DEPTH_OPTIONS = [
  { value: "1", label: "1 block" },
  { value: "2", label: "2 blocks" },
  { value: "3", label: "3 blocks" },
]

const FILTER_OPTIONS = [
  { value: "4", label: "4" },
  { value: "8", label: "8" },
  { value: "12", label: "12" },
]

const KERNEL_SIZE_OPTIONS = [
  { value: "3", label: "3×3" },
  { value: "5", label: "5×5" },
]

const PADDING_OPTIONS = [
  { value: "same", label: "Same" },
  { value: "valid", label: "Valid" },
]

const POOL_MODE_OPTIONS = [
  { value: "max", label: "Max" },
  { value: "avg", label: "Average" },
]

const POOL_WINDOW_OPTIONS = [
  { value: "2", label: "2×2" },
  { value: "3", label: "3×3" },
]

const FILTER_MODE_OPTIONS = [
  { value: "interpretable", label: "Interpretable" },
  { value: "random", label: "Random" },
]

const MIN_SIZE = 8
const MAX_SIZE = 32
const SIZE_STEP = 4

/** Above this many columns the inspector hides per-cell numbers. */
const NUMBER_CELL_LIMIT = 12

const DEFAULTS = {
  channelMode: "grayscale",
  scene: "gradient",
  size: 24,
  depth: 2,
  filters: 8,
  kernelSize: 3,
  padding: "same" as "same" | "valid",
  poolWindow: 2,
  poolMode: "max" as "max" | "avg",
  filterMode: "interpretable" as "interpretable" | "random",
  seed: 7,
  speed: 700,
  showNumbers: true,
  colorWeights: true,
}

type Cell = { i: number; j: number }

function clamp01(value: number) {
  return Math.min(1, Math.max(0, value))
}

/**
 * One channel rendered as pixels — 1 device pixel per cell, CSS scales it.
 * `image` mode paints the input volume (grey or RGB), `heat` mode paints the
 * diverging feature-map colours.
 */
const MapCanvas = React.memo(function MapCanvas({
  tensor,
  channel = 0,
  mode,
  bound = 1,
}: {
  tensor: Tensor
  channel?: number
  mode: "image" | "heat"
  bound?: number
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const rows = tensor[channel]?.length ?? 0
  const cols = tensor[channel]?.[0]?.length ?? 0

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || cols === 0) return
    const context = canvas.getContext("2d")
    if (!context) return
    const image = context.createImageData(cols, rows)
    const data = image.data

    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        const offset = (i * cols + j) * 4
        if (mode === "image") {
          if (tensor.length === 1) {
            const level = Math.round(clamp01(tensor[0][i][j]) * 255)
            data[offset] = level
            data[offset + 1] = level
            data[offset + 2] = level
          } else {
            data[offset] = Math.round(clamp01(tensor[0][i][j]) * 255)
            data[offset + 1] = Math.round(clamp01(tensor[1][i][j]) * 255)
            data[offset + 2] = Math.round(clamp01(tensor[2][i][j]) * 255)
          }
        } else {
          const [r, g, b] = heatRgb(tensor[channel]?.[i]?.[j] ?? 0, bound)
          data[offset] = r
          data[offset + 1] = g
          data[offset + 2] = b
        }
        data[offset + 3] = 255
      }
    }

    context.putImageData(image, 0, 0)
  }, [tensor, channel, mode, bound, rows, cols])

  if (rows === 0 || cols === 0) {
    return <span className="block size-full bg-muted/50" />
  }

  return (
    <canvas
      ref={canvasRef}
      width={cols}
      height={rows}
      aria-hidden
      className="block size-full"
      style={{ imageRendering: "pixelated" }}
    />
  )
})

/** Exact values printed over a canvas; only used on small maps. */
function NumberOverlay({ map, bound }: { map: FeatureMap; bound: number }) {
  const cols = map[0]?.length ?? 0
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute inset-0 grid"
      style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
    >
      {map.map((row, i) =>
        row.map((value, j) => (
          <span
            key={`${i}-${j}`}
            className="flex min-w-0 items-center justify-center font-mono text-[9px] leading-none tabular-nums"
            style={{ color: heatTextColor(value, bound) }}
          >
            {formatValue(value)}
          </span>
        ))
      )}
    </span>
  )
}

/** The window an earlier stage contributed, drawn over a map thumbnail. */
function RegionOverlay({
  region,
  rows,
  cols,
}: {
  region: RfRegion
  rows: number
  cols: number
}) {
  return (
    <span
      aria-hidden
      className="pointer-events-none absolute rounded-[2px] bg-primary/15 ring-2 ring-primary"
      style={{
        left: `${(region.col / cols) * 100}%`,
        top: `${(region.row / rows) * 100}%`,
        width: `${(region.cols / cols) * 100}%`,
        height: `${(region.rows / rows) * 100}%`,
      }}
    />
  )
}

function StageCard({
  index,
  name,
  stage,
  revealed,
  selected,
  region,
  onSelect,
}: {
  index: number
  name: string
  stage: StageResult
  revealed: boolean
  selected: boolean
  region: RfRegion | null
  onSelect: (index: number) => void
}) {
  const isInput = stage.spec.kind === "input"

  return (
    <button
      type="button"
      data-stage-card={index}
      onClick={() => onSelect(index)}
      className={cn(
        "grid w-[6.5rem] shrink-0 content-start gap-1.5 rounded-lg bg-muted/30 p-1.5 text-left ring-1 ring-foreground/10 transition-colors hover:bg-muted/60 hover:ring-foreground/25 focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
        selected && "bg-muted/50 ring-2 ring-primary",
        !revealed && "opacity-30"
      )}
    >
      <span className="flex min-w-0 items-baseline justify-between gap-1">
        <span className="truncate text-[11px] font-medium">{name}</span>
        <span className="shrink-0 font-mono text-[9px] text-muted-foreground tabular-nums">
          {stage.rows}×{stage.cols}×{stage.channels}
        </span>
      </span>
      {isInput ? (
        <span className="relative block size-[4.5rem] overflow-hidden rounded-[3px] ring-1 ring-foreground/10">
          <MapCanvas tensor={stage.output} mode="image" />
          {region ? (
            <RegionOverlay region={region} rows={stage.rows} cols={stage.cols} />
          ) : null}
        </span>
      ) : (
        <span className="grid h-[4.5rem] grid-cols-4 content-start gap-[2px] overflow-hidden">
          {stage.output.map((_, channel) => (
            <span
              key={channel}
              className="relative size-5 shrink-0 overflow-hidden rounded-[3px] ring-1 ring-foreground/10"
            >
              <MapCanvas
                tensor={stage.output}
                channel={channel}
                mode="heat"
                bound={stage.stats.bound}
              />
              {region && (region.channel === "all" || region.channel === channel) ? (
                <RegionOverlay
                  region={region}
                  rows={stage.rows}
                  cols={stage.cols}
                />
              ) : null}
            </span>
          ))}
        </span>
      )}
    </button>
  )
}

function FilterPanel({
  filter,
  index,
  inChannels,
  kernelSize,
  colorWeights,
}: {
  filter: Filter
  index: number
  inChannels: number
  kernelSize: number
  colorWeights: boolean
}) {
  const scale = React.useMemo(() => {
    let max = 0
    for (const kernel of filter.weights) {
      for (const row of kernel) {
        for (const value of row) max = Math.max(max, Math.abs(value))
      }
    }
    return max || 1
  }, [filter])

  return (
    <div className="grid content-start gap-2">
      <span className="flex items-center gap-1.5 text-xs font-medium">
        {filter.label ?? `Filter ${index + 1}`}
        <InfoTip>
          The weights that produce this output map: one {kernelSize}×{kernelSize}
          {" "}grid per input channel. Every position of the incoming volume is
          dotted with them, then the bias is added.
        </InfoTip>
      </span>
      <div className="flex flex-wrap gap-3">
        {filter.weights.map((kernel, channel) => (
          <div key={channel} className="grid gap-1">
            {inChannels > 1 ? (
              <span
                className="text-[10px] font-medium"
                style={{ color: CHANNEL_COLOR[channel] }}
              >
                {CHANNEL_LABEL[channel]}
              </span>
            ) : null}
            <div
              className={cn(
                "grid gap-[2px]",
                kernelSize >= 5 ? "w-24" : "w-20"
              )}
              style={{
                gridTemplateColumns: `repeat(${kernelSize}, minmax(0, 1fr))`,
              }}
            >
              {kernel.map((row, m) =>
                row.map((value, n) => (
                  <span
                    key={`${m}-${n}`}
                    title={`w[${m}][${n}] = ${value}`}
                    style={weightStyle(value, colorWeights, scale)}
                    className={cn(
                      "flex aspect-square items-center justify-center rounded-[3px] bg-muted/60 font-mono text-[9px] tabular-nums ring-1 ring-foreground/10",
                      value === 0 && "text-muted-foreground"
                    )}
                  >
                    {formatValue(value)}
                  </span>
                ))
              )}
            </div>
          </div>
        ))}
      </div>
      <p className="font-mono text-[10px] text-muted-foreground tabular-nums">
        bias {formatValue(filter.bias)}
      </p>
    </div>
  )
}

function stageTip(stage: StageResult): React.ReactNode {
  const spec = stage.spec
  if (spec.kind === "input") {
    return "The tensor the network starts from: H×W×C values in 0…1. Every stage to the right recomputes when the input or a network control changes."
  }
  if (spec.kind === "conv") {
    return (
      <>
        Each filter slides over all input channels at once: one dot product per
        position, plus its bias — and one output map per filter. Weights are
        shared across positions, which is why a conv layer is so cheap.
      </>
    )
  }
  if (spec.kind === "relu") {
    return "Pointwise gate: max(0, x). Every negative value is zeroed, and this is the only non-linearity in the pipeline — without it, stacked convolutions would collapse into one linear filter."
  }
  return `${spec.mode === "max" ? "Max" : "Average"} pooling with a ${spec.window}×${spec.window} window and stride ${spec.stride}: each window collapses to one value, so the map shrinks and small shifts stop changing it.`
}

export function CnnPipeline() {
  const [channelMode, setChannelMode] = React.useState(DEFAULTS.channelMode)
  const channels = channelMode === "rgb" ? 3 : 1
  const [source, setSource] = React.useState<InputSource>("synthetic")
  const [scene, setScene] = React.useState(DEFAULTS.scene)
  const [imageId, setImageId] = React.useState(DEFAULT_CNN_IMAGE_ID)
  const [size, setSize] = React.useState(DEFAULTS.size)
  const [depth, setDepth] = React.useState(DEFAULTS.depth)
  const [filters, setFilters] = React.useState(DEFAULTS.filters)
  const [kernelSize, setKernelSize] = React.useState(DEFAULTS.kernelSize)
  const [padding, setPadding] = React.useState(DEFAULTS.padding)
  const [poolWindow, setPoolWindow] = React.useState(DEFAULTS.poolWindow)
  const [poolMode, setPoolMode] = React.useState(DEFAULTS.poolMode)
  const [filterMode, setFilterMode] = React.useState(DEFAULTS.filterMode)
  const [seed, setSeed] = React.useState(DEFAULTS.seed)

  const [input, setInput] = React.useState<number[][][]>(() =>
    buildScene(DEFAULTS.scene, DEFAULTS.size, 1)
  )
  const [imageState, setImageState] = React.useState<
    "idle" | "loading" | "ready" | "error"
  >("idle")

  const [selected, setSelected] = React.useState(0)
  const [selectedChannel, setSelectedChannel] = React.useState(0)
  const [revealed, setRevealed] = React.useState(Number.POSITIVE_INFINITY)
  const [playing, setPlaying] = React.useState(false)
  const [hovered, setHovered] = React.useState<Cell | null>(null)
  const [pinned, setPinned] = React.useState<Cell | null>(null)

  const [showNumbers, setShowNumbers] = React.useState(DEFAULTS.showNumbers)
  const [colorWeights, setColorWeights] = React.useState(DEFAULTS.colorWeights)
  const [speed, setSpeed] = React.useState(DEFAULTS.speed)

  const imageCache = React.useRef<Map<string, HTMLImageElement>>(new Map())
  const stripRef = React.useRef<HTMLDivElement>(null)
  const mapRef = React.useRef<HTMLDivElement>(null)
  const counterRef = React.useRef<HTMLElement>(null)
  const reducedMotion = usePrefersReducedMotion()
  const photo = source === "photo" ? findCnnImage(imageId) : undefined

  const network = React.useMemo(
    () =>
      buildNetwork({
        size,
        inChannels: channels,
        depth,
        filters,
        kernelSize,
        padding,
        poolWindow,
        poolMode,
        filterMode,
        seed,
      }),
    [
      size,
      channels,
      depth,
      filters,
      kernelSize,
      padding,
      poolWindow,
      poolMode,
      filterMode,
      seed,
    ]
  )

  const inputTensor = React.useMemo(() => toChannelFirst(input), [input])
  const stages = React.useMemo(
    () => forwardPass(inputTensor, network.specs),
    [inputTensor, network]
  )
  const names = React.useMemo(() => layerNames(network.specs), [network])
  const rfSizes = React.useMemo(
    () => receptiveFieldSizes(network.specs),
    [network]
  )

  const stageIndex = playing
    ? Math.min(Math.max(0, revealed - 1), Math.max(0, stages.length - 1))
    : Math.min(selected, Math.max(0, stages.length - 1))
  const stage = stages[stageIndex]
  const activeChannel = Math.min(
    selectedChannel,
    Math.max(0, (stage?.channels ?? 1) - 1)
  )

  const probe = pinned ?? hovered
  const activeProbe = React.useMemo(() => {
    if (!probe || !stage) return null
    if (probe.i < 0 || probe.i >= stage.rows) return null
    if (probe.j < 0 || probe.j >= stage.cols) return null
    return probe
  }, [probe, stage])

  const regions = React.useMemo(
    () =>
      activeProbe && stage
        ? traceReceptiveField(
            stages,
            stageIndex,
            activeChannel,
            activeProbe.i,
            activeProbe.j
          )
        : [],
    [stages, stage, stageIndex, activeChannel, activeProbe]
  )

  const resetProbe = React.useCallback(() => {
    setHovered(null)
    setPinned(null)
  }, [])

  /** Any config change shows the whole network instantly and stops playback. */
  const applyConfig = React.useCallback(() => {
    setPlaying(false)
    setRevealed(Number.POSITIVE_INFINITY)
    setSelectedChannel(0)
    resetProbe()
  }, [resetProbe])

  React.useEffect(() => {
    if (!playing) return
    if (revealed >= stages.length) return
    const id = window.setTimeout(() => {
      const next = revealed + 1
      setRevealed(next)
      setSelected(next - 1)
      if (next >= stages.length) setPlaying(false)
    }, speed)
    return () => window.clearTimeout(id)
  }, [playing, revealed, speed, stages.length])

  React.useEffect(() => {
    if (source !== "photo") return
    const selectedPhoto = findCnnImage(imageId)
    if (!selectedPhoto) return

    let cancelled = false

    const sample = async () => {
      try {
        let element = imageCache.current.get(selectedPhoto.id)
        if (!element) {
          element = await loadImageElement(selectedPhoto.src)
          imageCache.current.set(selectedPhoto.id, element)
        }
        if (cancelled) return
        setInput(imageToGrid(element, size, channels))
        setImageState("ready")
      } catch {
        if (!cancelled) setImageState("error")
      }
    }

    void sample()
    return () => {
      cancelled = true
    }
  }, [source, imageId, size, channels])

  const previousRevealed = React.useRef(Number.POSITIVE_INFINITY)

  /** A stage that the forward pass just reached pops in and scrolls into view. */
  useIsomorphicLayoutEffect(() => {
    const previous = previousRevealed.current
    previousRevealed.current = revealed
    const strip = stripRef.current
    if (!strip) return
    if (!Number.isFinite(revealed) || revealed <= previous) return

    const card = strip.querySelector<HTMLElement>(
      `[data-stage-card="${revealed - 1}"]`
    )
    if (!card) return
    card.scrollIntoView({
      inline: "nearest",
      block: "nearest",
      behavior: reducedMotion ? "auto" : "smooth",
    })
    if (reducedMotion) return

    const animation = animate(card, {
      opacity: [0, 1],
      scale: [0.85, 1],
      duration: 380,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [revealed, reducedMotion])

  /** The inspector map cross-fades when the stage or channel changes. */
  useIsomorphicLayoutEffect(() => {
    const node = mapRef.current
    if (!node || reducedMotion) return
    const animation = animate(node, {
      opacity: [0.35, 1],
      duration: 220,
      ease: "out(2)",
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, activeChannel, reducedMotion])

  useIsomorphicLayoutEffect(() => {
    const badge = counterRef.current
    if (!badge || reducedMotion) return
    const animation = animate(badge, {
      scale: [1.15, 1],
      duration: 240,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [revealed, reducedMotion])

  const runPass = () => {
    resetProbe()
    setSelected(0)
    setRevealed(0)
    setPlaying(true)
  }

  const step = () => {
    setPlaying(false)
    resetProbe()
    const next = revealed >= stages.length ? 1 : revealed + 1
    setRevealed(next)
    setSelected(next - 1)
  }

  const resetAnimation = () => {
    setPlaying(false)
    setRevealed(Number.POSITIVE_INFINITY)
    resetProbe()
  }

  const selectStage = (index: number) => {
    setPlaying(false)
    setSelected(index)
    resetProbe()
  }

  const moveProbe = (deltaI: number, deltaJ: number) => {
    if (!stage || stage.rows === 0 || stage.cols === 0) return
    const base = activeProbe ?? {
      i: Math.floor(stage.rows / 2),
      j: Math.floor(stage.cols / 2),
    }
    const next = {
      i: Math.min(stage.rows - 1, Math.max(0, base.i + deltaI)),
      j: Math.min(stage.cols - 1, Math.max(0, base.j + deltaJ)),
    }
    if (pinned) setPinned(next)
    else setHovered(next)
  }

  const togglePin = (cell: Cell) => {
    setPinned((current) =>
      current && current.i === cell.i && current.j === cell.j ? null : cell
    )
  }

  const changeChannelMode = (value: string) => {
    if (value !== "grayscale" && value !== "rgb") return
    const nextChannels = value === "rgb" ? 3 : 1
    setChannelMode(value)
    if (source === "photo") {
      setImageState("loading")
    } else {
      setInput(buildScene(scene, size, nextChannels))
    }
    applyConfig()
  }

  const changeSource = (value: string) => {
    if (value !== "synthetic" && value !== "photo") return
    if (value === source) return
    setSource(value)
    if (value === "photo") setImageState("loading")
    else {
      setInput(buildScene(scene, size, channels))
      setImageState("idle")
    }
    applyConfig()
  }

  const changeImage = (value: string) => {
    setImageId(value)
    setImageState("loading")
    applyConfig()
  }

  const changeSize = (value: number) => {
    setSize(value)
    if (source === "photo") setImageState("loading")
    else setInput(buildScene(scene, value, channels))
    applyConfig()
  }

  const changeScene = (value: string) => {
    setScene(value)
    setInput(buildScene(value, size, channels))
    applyConfig()
  }

  const shuffleFilters = () => {
    setSeed(Math.floor(Math.random() * 1_000_000) + 1)
    applyConfig()
  }

  const resetAll = () => {
    setChannelMode(DEFAULTS.channelMode)
    setSource("synthetic")
    setScene(DEFAULTS.scene)
    setImageId(DEFAULT_CNN_IMAGE_ID)
    setImageState("idle")
    setSize(DEFAULTS.size)
    setDepth(DEFAULTS.depth)
    setFilters(DEFAULTS.filters)
    setKernelSize(DEFAULTS.kernelSize)
    setPadding(DEFAULTS.padding)
    setPoolWindow(DEFAULTS.poolWindow)
    setPoolMode(DEFAULTS.poolMode)
    setFilterMode(DEFAULTS.filterMode)
    setSeed(DEFAULTS.seed)
    setShowNumbers(DEFAULTS.showNumbers)
    setColorWeights(DEFAULTS.colorWeights)
    setSpeed(DEFAULTS.speed)
    setSelected(0)
    setSelectedChannel(0)
    setInput(buildScene(DEFAULTS.scene, DEFAULTS.size, 1))
    applyConfig()
  }

  const value =
    activeProbe && stage
      ? stage.output[activeChannel]?.[activeProbe.i]?.[activeProbe.j]
      : undefined

  const rfHint = React.useMemo(() => {
    if (!stage) return ""
    const field = rfSizes[stageIndex] ?? 1
    const base = `sees ${field}×${field} px of the input`
    const region = stageIndex > 0 ? regions[stageIndex - 1] : null
    if (!activeProbe || !region) return base
    const everyChannel =
      region.channel === "all" && (stages[stageIndex - 1]?.channels ?? 1) > 1
    return `${base} · RF on ${names[stageIndex - 1]}: ${region.rows}×${
      region.cols
    } @ (${region.row}, ${region.col})${everyChannel ? " · every map" : ""}`
  }, [stage, activeProbe, regions, stages, stageIndex, names, rfSizes])

  const canShowNumbers =
    stage !== undefined &&
    stage.spec.kind !== "input" &&
    stage.cols <= NUMBER_CELL_LIMIT

  const reluStages = stages.filter((entry) => entry.spec.kind === "relu")
  const convStages = stages.filter((entry) => entry.spec.kind === "conv")
  const total = React.useMemo(() => totalParams(stages), [stages])
  const convIndexes = names.reduce<number[]>((list, name, index) => {
    if (name.startsWith("Conv")) list.push(index)
    return list
  }, [])
  const firstConvIndex = convIndexes[0]
  const lastConvIndex = convIndexes[convIndexes.length - 1]
  const firstName =
    firstConvIndex !== undefined ? names[firstConvIndex] : undefined
  const lastName =
    lastConvIndex !== undefined ? names[lastConvIndex] : undefined
  const firstRf = firstConvIndex !== undefined ? rfSizes[firstConvIndex] : 1
  const lastRf = lastConvIndex !== undefined ? rfSizes[lastConvIndex] : 1

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid content-start gap-4">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              CNN Pipeline
              <InfoTip>
                <span className="block">
                  A real forward pass: every stage is a tensor, computed from
                  the stage before it. Click a stage to inspect it, hover a
                  cell to trace its receptive field back through the network.
                </span>
                <span className="mt-1 block font-mono text-[11px]">
                  conv: out = Σ_c Σ_m Σ_n X_c[i·s+m−p, j·s+n−p] · W_c[m,n] + b
                </span>
                <span className="mt-1 block font-mono text-[11px]">
                  relu: max(0, x) · pool: max / mean over a k×k window
                </span>
              </InfoTip>
            </CardTitle>
            <CardAction>
              <div className="flex flex-wrap items-center justify-end gap-2">
                {playing ? (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setPlaying(false)}
                  >
                    <PauseIcon data-icon="inline-start" />
                    Pause
                  </Button>
                ) : (
                  <Button size="sm" onClick={runPass}>
                    <PlayIcon data-icon="inline-start" />
                    Run pass
                  </Button>
                )}
                <Button size="sm" variant="outline" onClick={step}>
                  <StepForwardIcon data-icon="inline-start" />
                  Step
                </Button>
                <Button size="sm" variant="ghost" onClick={resetAnimation}>
                  <RotateCcwIcon data-icon="inline-start" />
                  Reset
                </Button>
                <Badge
                  ref={counterRef}
                  variant="secondary"
                  className="font-mono tabular-nums"
                >
                  {Math.min(revealed, stages.length)} / {stages.length}
                </Badge>
              </div>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-4">
            {source === "photo" && imageState === "error" ? (
              <Alert>
                <LightbulbIcon />
                <AlertTitle>Image could not be loaded</AlertTitle>
                <AlertDescription>
                  {photo
                    ? `${photo.label} is missing from public/images/cnn. Pick another photo or switch back to a synthetic scene.`
                    : "Pick another photo or switch back to a synthetic scene."}
                </AlertDescription>
              </Alert>
            ) : null}
            {source === "photo" && imageState !== "ready" ? (
              <p className="text-sm text-muted-foreground">
                {imageState === "error"
                  ? "No photo loaded — the strip below still shows the last input."
                  : "Sampling the photo onto the grid…"}
              </p>
            ) : null}

            <div
              ref={stripRef}
              className="flex items-stretch gap-1.5 overflow-x-auto pb-1"
              aria-label="Network stages"
            >
              {stages.map((entry, index) => (
                <React.Fragment key={index}>
                  {index > 0 ? (
                    <span className="flex shrink-0 items-center text-muted-foreground/40">
                      <ArrowRightIcon className="size-3.5" />
                    </span>
                  ) : null}
                  <StageCard
                    index={index}
                    name={names[index]}
                    stage={entry}
                    revealed={index < revealed}
                    selected={index === stageIndex}
                    region={regions[index] ?? null}
                    onSelect={selectStage}
                  />
                </React.Fragment>
              ))}
            </div>

            {network.blocks < network.requestedBlocks ? (
              <p className="text-xs text-muted-foreground">
                Valid padding ends the pipeline after {network.blocks} block
                {network.blocks === 1 ? "" : "s"} — another{" "}
                {kernelSize}×{kernelSize} kernel no longer fits the remaining
                map.
              </p>
            ) : null}

            {stage ? (
              <section className="@container/inspector grid gap-3 rounded-lg bg-muted/40 p-3 ring-1 ring-foreground/10">
                <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <span className="flex min-w-0 flex-wrap items-center gap-2 text-sm font-medium">
                    <Badge variant="outline">{stageIndex + 1}</Badge>
                    {names[stageIndex]}
                    <span className="font-mono text-xs font-normal text-muted-foreground">
                      {stage.rows}×{stage.cols}×{stage.channels}
                      {stage.params > 0 ? ` · ${stage.params} params` : ""}
                    </span>
                    <InfoTip>{stageTip(stage)}</InfoTip>
                  </span>
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    {value !== undefined && activeProbe ? (
                      <>
                        ({activeProbe.i}, {activeProbe.j}) ={" "}
                        <span className="font-semibold text-foreground">
                          {formatValue(value)}
                        </span>
                      </>
                    ) : (
                      <>
                        min {formatValue(stage.stats.min)} · max{" "}
                        {formatValue(stage.stats.max)} ·{" "}
                        {Math.round(stage.stats.zeroFraction * 100)}% zeros
                      </>
                    )}
                  </span>
                </header>
                <div className="grid gap-3 @lg/inspector:grid-cols-[minmax(0,1fr)_auto]">
                  <div className="grid content-start gap-2">
                    {stage.spec.kind === "input" ? null : (
                      <div className="flex flex-wrap gap-1">
                        {stage.output.map((_, channel) => (
                          <button
                            key={channel}
                            type="button"
                            onClick={() => {
                              setSelectedChannel(channel)
                              setPinned(null)
                            }}
                            className={cn(
                              "relative size-8 overflow-hidden rounded ring-1 ring-foreground/10 transition-shadow hover:ring-foreground/40",
                              channel === activeChannel && "ring-2 ring-primary"
                            )}
                            aria-label={`Channel ${channel + 1}`}
                          >
                            <MapCanvas
                              tensor={stage.output}
                              channel={channel}
                              mode="heat"
                              bound={stage.stats.bound}
                            />
                          </button>
                        ))}
                      </div>
                    )}
                    <div
                      ref={mapRef}
                      role="grid"
                      aria-label={`${names[stageIndex]} map, use arrow keys to inspect cells`}
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft") moveProbe(0, -1)
                        else if (event.key === "ArrowRight") moveProbe(0, 1)
                        else if (event.key === "ArrowUp") moveProbe(-1, 0)
                        else if (event.key === "ArrowDown") moveProbe(1, 0)
                        else return
                        event.preventDefault()
                      }}
                      onPointerMove={(event) => {
                        const rect =
                          event.currentTarget.getBoundingClientRect()
                        const j = Math.floor(
                          ((event.clientX - rect.left) / rect.width) *
                            stage.cols
                        )
                        const i = Math.floor(
                          ((event.clientY - rect.top) / rect.height) *
                            stage.rows
                        )
                        if (
                          i < 0 ||
                          j < 0 ||
                          i >= stage.rows ||
                          j >= stage.cols
                        )
                          return
                        setHovered((current) =>
                          current && current.i === i && current.j === j
                            ? current
                            : { i, j }
                        )
                      }}
                      onPointerLeave={() => setHovered(null)}
                      onClick={() => {
                        if (activeProbe) togglePin(activeProbe)
                      }}
                      className="relative w-full max-w-[22rem] cursor-crosshair rounded-md outline-none ring-1 ring-foreground/10 focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <MapCanvas
                        tensor={stage.output}
                        channel={activeChannel}
                        mode={stage.spec.kind === "input" ? "image" : "heat"}
                        bound={stage.stats.bound}
                      />
                      {showNumbers &&
                      canShowNumbers &&
                      stage.output[activeChannel] ? (
                        <NumberOverlay
                          map={stage.output[activeChannel]}
                          bound={stage.stats.bound}
                        />
                      ) : null}
                      {activeProbe ? (
                        <span
                          aria-hidden
                          className="pointer-events-none absolute rounded-[2px] bg-primary/15 ring-2 ring-primary"
                          style={{
                            left: `${(activeProbe.j / stage.cols) * 100}%`,
                            top: `${(activeProbe.i / stage.rows) * 100}%`,
                            width: `${100 / stage.cols}%`,
                            height: `${100 / stage.rows}%`,
                          }}
                        />
                      ) : null}
                    </div>
                    <p className="font-mono text-[11px] text-muted-foreground tabular-nums">
                      {rfHint}
                    </p>
                  </div>
                  {stage.spec.kind === "conv" &&
                  stage.spec.filters[activeChannel] ? (
                    <FilterPanel
                      filter={stage.spec.filters[activeChannel]}
                      index={activeChannel}
                      inChannels={
                        stages[stageIndex - 1]?.channels ?? channels
                      }
                      kernelSize={stage.spec.kernelSize}
                      colorWeights={colorWeights}
                    />
                  ) : null}
                </div>
              </section>
            ) : null}
          </CardContent>
        </Card>

        <Collapsible>
          <Card>
            <CardHeader>
              <CollapsibleTrigger asChild>
                <button
                  type="button"
                  className="group flex w-full items-center justify-between gap-2 rounded-md text-left focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none"
                >
                  <span className="flex items-center gap-2">
                    <SparklesIcon className="size-4 text-muted-foreground" />
                    <CardTitle>Live insights</CardTitle>
                  </span>
                  <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
                </button>
              </CollapsibleTrigger>
            </CardHeader>
            <CollapsibleContent>
              <CardContent className="grid gap-2 text-sm text-muted-foreground">
                <p>
                  Every stage between the input and the last pooled map is a
                  hidden layer: {stages.length - 1} activations from{" "}
                  {network.blocks} conv block
                  {network.blocks === 1 ? "" : "s"} here.
                </p>
                {firstName ? (
                  <p>
                    {firstName === lastName
                      ? `A neuron in ${lastName} sees ${lastRf}×${lastRf} px of the input.`
                      : `A neuron in ${lastName} sees ${lastRf}×${lastRf} px of the input; in ${firstName} only ${firstRf}×${firstRf}. Stacking conv and pool is what grows that view.`}
                  </p>
                ) : null}
                {convStages.length > 0 ? (
                  <p>
                    Parameters:{" "}
                    {convStages
                      .map(
                        (entry) =>
                          `${names[stages.indexOf(entry)]} ${entry.params}`
                      )
                      .join(" · ")}{" "}
                    = {total} weights. Hand-designed and seeded here — a
                    trained net would have learned the same count.
                  </p>
                ) : null}
                {reluStages.length > 0 ? (
                  <p>
                    {reluStages
                      .map(
                        (entry) =>
                          `${names[stages.indexOf(entry)]} keeps ${Math.round(
                            (1 - entry.stats.zeroFraction) * 100
                          )}%`
                      )
                      .join(" · ")}{" "}
                    of its values. The zeroed rest is the non-linearity doing
                    its job — without it, depth would be pointless.
                  </p>
                ) : null}
                <p>
                  {poolMode === "max"
                    ? "Max pooling keeps the strongest value per window, so the strongest edge survives the shrink; small shifts stop moving the map."
                    : "Average pooling smooths each window, trading sharpness for stability — every value contributes, none dominates."}
                </p>
                <p>
                  Hover or tap any cell in the inspector: its receptive field
                  lights up on every earlier stage in the strip above.
                </p>
                <p>
                  Filters are hand-designed presets in the first conv and
                  seeded random weights deeper down. Shapes, activations and
                  receptive fields are exactly what a real CNN computes — only
                  the weights are not trained.
                </p>
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            Controls
            <InfoTip>Every control recomputes the whole forward pass.</InfoTip>
          </CardTitle>
          <CardAction>
            <Button variant="ghost" size="sm" onClick={resetAll}>
              <RotateCcwIcon data-icon="inline-start" />
              Reset all
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-5">
          <ControlGroup label="Input">
            <ChoiceField
              label="Source"
              value={source}
              options={SOURCE_OPTIONS}
              onChange={changeSource}
              info="Synthetic scenes come from formulas; photos are real bitmaps resampled onto the same grid."
            />
            {source === "photo" ? (
              <SelectField
                label="Photo"
                value={imageId}
                onValueChange={changeImage}
                options={PHOTO_OPTIONS}
                info="Five Wikimedia Commons images, resampled onto the input grid."
              />
            ) : (
              <SelectField
                label="Scene"
                value={scene}
                onValueChange={changeScene}
                options={SCENE_OPTIONS}
                info="The whole network recomputes from this image."
              />
            )}
            <ChoiceField
              label="Channels"
              value={channelMode}
              options={CHANNEL_OPTIONS.map((option) => ({
                value: option.value,
                label: option.label,
              }))}
              onChange={changeChannelMode}
              info="RGB gives the first conv three input channels, so each filter holds three weight grids."
            />
            <SliderField
              label="Input size"
              value={size}
              onValueChange={changeSize}
              min={MIN_SIZE}
              max={MAX_SIZE}
              step={SIZE_STEP}
              format={(value) => `${value}×${value}`}
              info="Side length of the square input tensor."
            />
            {photo ? (
              <div className="flex items-center gap-2 rounded-lg bg-muted/40 p-1.5">
                <Image
                  src={photo.src}
                  alt={photo.alt}
                  width={256}
                  height={256}
                  className="size-10 shrink-0 rounded-md object-cover ring-1 ring-foreground/10"
                />
                <span className="flex min-w-0 items-center gap-1.5 text-xs">
                  <span className="truncate font-medium">{photo.label}</span>
                  <InfoTip>
                    {photo.note} {photo.credit} · {photo.license} ·{" "}
                    <a
                      href={photo.sourceUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="underline underline-offset-2"
                    >
                      source
                    </a>
                  </InfoTip>
                </span>
              </div>
            ) : null}
          </ControlGroup>
          <Separator />
          <ControlGroup label="Network">
            <ChoiceField
              label="Depth"
              value={String(depth)}
              options={DEPTH_OPTIONS}
              onChange={(value) => {
                setDepth(Number(value))
                applyConfig()
              }}
              info="One block is conv → ReLU → pool. Deeper stacks shrink the map and grow the receptive field."
            />
            <ChoiceField
              label="Filters per layer"
              value={String(filters)}
              options={FILTER_OPTIONS}
              onChange={(value) => {
                setFilters(Number(value))
                applyConfig()
              }}
              info="How many feature maps each conv layer produces (its output channels)."
            />
            <ChoiceField
              label="Kernel size"
              value={String(kernelSize)}
              options={KERNEL_SIZE_OPTIONS}
              onChange={(value) => {
                setKernelSize(Number(value))
                applyConfig()
              }}
            />
            <ChoiceField
              label="Padding"
              value={padding}
              options={PADDING_OPTIONS}
              onChange={(value) => {
                if (value !== "same" && value !== "valid") return
                setPadding(value)
                applyConfig()
              }}
              info="Same keeps each conv map at the input size; Valid lets it shrink at every conv."
            />
            <ChoiceField
              label="Pooling"
              value={poolMode}
              options={POOL_MODE_OPTIONS}
              onChange={(value) => {
                if (value !== "max" && value !== "avg") return
                setPoolMode(value)
                applyConfig()
              }}
              info="Max keeps the strongest value per window; average smooths every window."
            />
            <ChoiceField
              label="Pool window"
              value={String(poolWindow)}
              options={POOL_WINDOW_OPTIONS}
              onChange={(value) => {
                setPoolWindow(Number(value))
                applyConfig()
              }}
              info="Windows never overlap: stride equals the window size, so each side of the map halves (or thirds)."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="Filters">
            <ChoiceField
              label="First conv"
              value={filterMode}
              options={FILTER_MODE_OPTIONS}
              onChange={(value) => {
                if (value !== "interpretable" && value !== "random") return
                setFilterMode(value)
                applyConfig()
              }}
              info="Interpretable presets are named edge/blob detectors; random draws seeded weights. Deeper convs are always random."
            />
            {filterMode === "random" ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={shuffleFilters}
              >
                <DicesIcon data-icon="inline-start" />
                New filters
              </Button>
            ) : null}
          </ControlGroup>
          <Separator />
          <ControlGroup label="Display">
            <SwitchField
              label="Show numbers"
              checked={showNumbers}
              onCheckedChange={setShowNumbers}
              disabled={!canShowNumbers}
              info={
                canShowNumbers
                  ? "Print the value inside each cell of the inspected map."
                  : `Only drawn for maps up to ${NUMBER_CELL_LIMIT}×${NUMBER_CELL_LIMIT} cells.`
              }
            />
            <SwitchField
              label="Colour weights"
              checked={colorWeights}
              onCheckedChange={setColorWeights}
              info="Tint filter cells green for positive and red for negative weights."
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="Playback">
            <SliderField
              label="Pass speed"
              value={speed}
              onValueChange={setSpeed}
              min={200}
              max={1500}
              step={50}
              format={(value) => `${value} ms`}
              info="Delay per stage while the forward pass is running."
            />
          </ControlGroup>
        </CardContent>
      </Card>
    </div>
  )
}
