"use client"

import * as React from "react"
import Image from "next/image"
import { animate, createMotionPath, stagger, utils } from "animejs"
import {
  ChevronDownIcon,
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
  buildScene,
  CHANNEL_COLOR,
  CHANNEL_LABEL,
  toChannelFirst,
} from "@/lib/cnn-scenes"
import { formatValue, heatColor, heatRgb, heatTextColor, weightStyle } from "@/lib/heatmap"
import {
  CNN_IMAGES,
  DEFAULT_CNN_IMAGE_ID,
  findCnnImage,
  imageToGrid,
  loadImageElement,
} from "@/lib/cnn-images"
import {
  clampNumber,
  convolveChannels,
  effectiveKernel,
  extractWindow,
  KERNEL_PRESETS,
  kernelFromPreset,
  kernelSum,
  outputDimension,
  poolChannels,
  windowCoverage,
  type PaddingMode,
  type PoolingMode,
} from "@/lib/convolution"
import { cn } from "@/lib/utils"

type Operation = "convolution" | "max-pool" | "avg-pool"

const CHANNEL_OPTIONS = [
  { value: "grayscale", label: "Grayscale" },
  { value: "rgb", label: "RGB" },
] as const

const OPERATION_OPTIONS: { value: Operation; label: string }[] = [
  { value: "convolution", label: "Convolution" },
  { value: "max-pool", label: "Max pool" },
  { value: "avg-pool", label: "Average pool" },
]

const PADDING_OPTIONS: { value: PaddingMode; label: string }[] = [
  { value: "valid", label: "Valid" },
  { value: "same", label: "Same" },
  { value: "zero", label: "Zero" },
]

const KERNEL_SIZE_OPTIONS = [
  { value: "1", label: "1×1" },
  { value: "3", label: "3×3" },
  { value: "5", label: "5×5" },
]

const SCENE_OPTIONS = [
  { value: "gradient", label: "Gradient" },
  { value: "edge", label: "Hard edge" },
  { value: "diagonal", label: "Diagonal" },
  { value: "cross", label: "Cross" },
  { value: "noise", label: "Noise" },
  { value: "smiley", label: "Smiley" },
]

type InputSource = "synthetic" | "photo"

const SOURCE_OPTIONS: { value: InputSource; label: string }[] = [
  { value: "synthetic", label: "Synthetic" },
  { value: "photo", label: "Photo" },
]

const PHOTO_OPTIONS = CNN_IMAGES.map((image) => ({
  value: image.id,
  label: image.label,
}))

/** Synthetic scenes are hand-sized; photos need a few more cells to read. */
const PHOTO_GRID_SIZE = 12

/** Above this many cells per side the grids switch to a canvas renderer. */
const CANVAS_CELL_LIMIT = 16

/** Sweeps always take about this many ticks, however many cells there are. */
const SWEEP_TICKS = 144

const MAX_SYNTHETIC_SIZE = 12
const PHOTO_SIZE_STEP = 4
const PHOTO_FALLBACK_MAX = 256

/** Matches `gap-[3px]` on the DOM grids; canvas grids have no gaps. */
const INPUT_CELL_GAP = 3

const DEFAULTS = {
  channelMode: "grayscale",
  scene: "gradient",
  size: 7,
  operation: "convolution" as Operation,
  kernelSize: 3,
  kernelPreset: "identity",
  stride: 1,
  padding: "valid" as PaddingMode,
  padWidth: 1,
  bias: 0,
  flip: false,
  showStride: true,
  speed: 320,
}

type Cell = { i: number; j: number }

function buildKernels(
  preset: string,
  size: number,
  channels: number
): number[][][] {
  return Array.from({ length: channels }, () =>
    kernelFromPreset(preset, size).map((row) => row.slice())
  )
}

function hexToRgb(hex: string): [number, number, number] {
  const value = hex.replace("#", "")
  const expanded =
    value.length === 3
      ? value
          .split("")
          .map((character) => character + character)
          .join("")
      : value
  const parsed = Number.parseInt(expanded, 16)
  if (Number.isNaN(parsed)) return [0.5, 0.5, 0.5]
  return [((parsed >> 16) & 255) / 255, ((parsed >> 8) & 255) / 255, (parsed & 255) / 255]
}

function inputBackground(pixel: number[], channels: number): string {
  if (channels === 1) {
    const level = clampNumber(pixel[0] ?? 0, 0, 1)
    return `hsl(0 0% ${(level * 100).toFixed(1)}%)`
  }
  const [r, g, b] = pixel
  return `rgb(${Math.round((r ?? 0) * 255)} ${Math.round((g ?? 0) * 255)} ${Math.round(
    (b ?? 0) * 255
  )})`
}

function inputTextColor(pixel: number[], channels: number): string {
  const [r, g, b] =
    channels === 1 ? [pixel[0], pixel[0], pixel[0]] : pixel
  const luminance = 0.2126 * (r ?? 0) + 0.7152 * (g ?? 0) + 0.0722 * (b ?? 0)
  return luminance > 0.6 ? "#0f172a" : "#f8fafc"
}

function inputLabel(pixel: number[], channels: number): string {
  if (channels === 1) return String(Math.round((pixel[0] ?? 0) * 100))
  const luminance =
    (0.2126 * (pixel[0] ?? 0) +
      0.7152 * (pixel[1] ?? 0) +
      0.0722 * (pixel[2] ?? 0)) *
    100
  return String(Math.round(luminance))
}

function KernelCell({
  value,
  onChange,
  colorWeights,
}: {
  value: number
  onChange: (value: number) => void
  colorWeights: boolean
}) {
  const start = React.useRef<{ x: number; value: number } | null>(null)

  return (
    <button
      type="button"
      data-pulse-cell
      title="Drag horizontally to change · double-click to zero"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        start.current = { x: event.clientX, value }
      }}
      onPointerMove={(event) => {
        if (!start.current) return
        const delta = (event.clientX - start.current.x) * 0.08
        const next = clampNumber(start.current.value + delta, -8, 8)
        onChange(Math.round(next * 100) / 100)
      }}
      onPointerUp={(event) => {
        event.currentTarget.releasePointerCapture(event.pointerId)
        start.current = null
      }}
      onPointerCancel={() => {
        start.current = null
      }}
      onDoubleClick={() => onChange(0)}
      style={weightStyle(value, colorWeights)}
      className={cn(
        "flex aspect-square min-w-0 items-center justify-center rounded-[3px] bg-muted/60 text-[10px] font-mono tabular-nums text-foreground ring-1 ring-foreground/10 transition-colors hover:ring-foreground/30",
        value === 0 && "text-muted-foreground"
      )}
    >
      {formatValue(value)}
    </button>
  )
}

function InputGrid({
  input,
  channels,
  active,
  kernelSize,
  stride,
  pad,
  showNumbers,
  coveredRows,
  coveredCols,
  markSkipped,
  onPaintStart,
  onPaintMove,
  onPaintEnd,
}: {
  input: number[][][]
  channels: number
  active: Cell | null
  kernelSize: number
  stride: number
  pad: number
  showNumbers: boolean
  coveredRows: boolean[]
  coveredCols: boolean[]
  markSkipped: boolean
  onPaintStart: (cell: Cell) => void
  onPaintMove: (cell: Cell | null) => void
  onPaintEnd: () => void
}) {
  const size = input.length
  const padded = size + 2 * pad
  const windowTop = active ? active.i * stride : 0
  const windowLeft = active ? active.j * stride : 0

  const cellFromEvent = (
    event: React.PointerEvent<HTMLDivElement>
  ): Cell | null => {
    const target = (event.target as HTMLElement).closest<HTMLElement>(
      "[data-cell]"
    )
    if (!target) return null
    const i = Number(target.dataset.row)
    const j = Number(target.dataset.col)
    if (Number.isNaN(i) || Number.isNaN(j)) return null
    return { i, j }
  }

  return (
    <div
      className="grid touch-none gap-[3px]"
      style={{ gridTemplateColumns: `repeat(${padded}, minmax(0, 1fr))` }}
      onPointerDown={(event) => {
        const cell = cellFromEvent(event)
        if (cell) onPaintStart(cell)
      }}
      onPointerMove={(event) => {
        if (event.buttons !== 1) return
        onPaintMove(cellFromEvent(event))
      }}
      onPointerUp={onPaintEnd}
      onPointerLeave={onPaintEnd}
      onPointerCancel={onPaintEnd}
    >
      {Array.from({ length: padded }, (_, pi) =>
        Array.from({ length: padded }, (_, pj) => {
          const i = pi - pad
          const j = pj - pad
          const inside = i >= 0 && i < size && j >= 0 && j < size
          const inWindow =
            active !== null &&
            pi >= windowTop &&
            pi < windowTop + kernelSize &&
            pj >= windowLeft &&
            pj < windowLeft + kernelSize

          if (!inside) {
            return (
              <div
                key={`pad-${pi}-${pj}`}
                title="Zero padding"
                className={cn(
                  "flex aspect-square items-center justify-center rounded-[3px] border border-dashed border-foreground/15 bg-muted/20 text-[10px] font-mono tabular-nums text-muted-foreground/40",
                  inWindow && "z-10 text-muted-foreground"
                )}
              >
                {showNumbers ? "0" : null}
              </div>
            )
          }

          const pixel = input[i][j]
          const skipped =
            markSkipped && (!coveredRows[pi] || !coveredCols[pj])
          return (
            <div
              key={`${i}-${j}`}
              data-cell
              data-row={i}
              data-col={j}
              style={{
                backgroundColor: inputBackground(pixel, channels),
                color: inputTextColor(pixel, channels),
              }}
              className={cn(
                "flex aspect-square cursor-crosshair items-center justify-center rounded-[3px] text-[10px] font-mono tabular-nums ring-1 ring-foreground/10",
                skipped && "opacity-40"
              )}
            >
              {showNumbers ? inputLabel(pixel, channels) : null}
            </div>
          )
        })
      )}
    </div>
  )
}

/**
 * Canvas twin of `InputGrid` for large grids: one device pixel per cell.
 * Painting is off at this density (a cell is a pixel); the active receptive
 * field is drawn by the animated window frame that overlays this canvas.
 */
function CanvasInputGrid({
  input,
  channels,
  pad,
}: {
  input: number[][][]
  channels: number
  pad: number
}) {
  const size = input.length
  const padded = size + 2 * pad
  const canvasRef = React.useRef<HTMLCanvasElement>(null)

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || size === 0) return
    const context = canvas.getContext("2d")
    if (!context) return
    const image = context.createImageData(padded, padded)
    const data = image.data
    for (let pi = 0; pi < padded; pi += 1) {
      for (let pj = 0; pj < padded; pj += 1) {
        const i = pi - pad
        const j = pj - pad
        if (i < 0 || i >= size || j < 0 || j >= size) continue
        const offset = (pi * padded + pj) * 4
        const pixel = input[i][j]
        if (channels === 1) {
          const level = Math.round(clampNumber(pixel[0] ?? 0, 0, 1) * 255)
          data[offset] = level
          data[offset + 1] = level
          data[offset + 2] = level
        } else {
          data[offset] = Math.round(clampNumber(pixel[0] ?? 0, 0, 1) * 255)
          data[offset + 1] = Math.round(clampNumber(pixel[1] ?? 0, 0, 1) * 255)
          data[offset + 2] = Math.round(clampNumber(pixel[2] ?? 0, 0, 1) * 255)
        }
        data[offset + 3] = 255
      }
    }
    context.putImageData(image, 0, 0)
  }, [input, channels, size, pad, padded])

  return (
    <div className="relative overflow-hidden rounded-md bg-muted/30 ring-1 ring-foreground/10">
      <canvas
        ref={canvasRef}
        width={padded}
        height={padded}
        aria-hidden
        className="block w-full"
        style={{ imageRendering: "pixelated" }}
      />
    </div>
  )
}

/**
 * Canvas twin of the output grid: heatmap pixels plus the reveal sweep,
 * so a 256×256 feature map stays interactive. Hover, click-to-pin and the
 * container's arrow keys keep working.
 */
function CanvasOutputGrid({
  values,
  rows,
  cols,
  bound,
  visibleCount,
  showHeatmap,
  active,
  reducedMotion,
  onHover,
  onTogglePin,
}: {
  values: number[][]
  rows: number
  cols: number
  bound: number
  visibleCount: number
  showHeatmap: boolean
  active: Cell | null
  reducedMotion: boolean
  onHover: (cell: Cell | null) => void
  onTogglePin: (cell: Cell) => void
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const activeRef = React.useRef<HTMLDivElement>(null)
  const lastCell = React.useRef<Cell | null>(null)

  useIsomorphicLayoutEffect(() => {
    const overlay = activeRef.current
    if (!overlay || reducedMotion) return
    const animation = animate(overlay, {
      opacity: [0.25, 1],
      duration: 260,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [active, reducedMotion])

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || cols === 0) return
    const context = canvas.getContext("2d")
    if (!context) return
    const image = context.createImageData(cols, rows)
    const data = image.data
    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        const index = i * cols + j
        if (!showHeatmap || index >= visibleCount) continue
        const [r, g, b] = heatRgb(values[i][j], bound)
        const offset = index * 4
        data[offset] = r
        data[offset + 1] = g
        data[offset + 2] = b
        data[offset + 3] = 255
      }
    }
    context.putImageData(image, 0, 0)
  }, [values, rows, cols, bound, visibleCount, showHeatmap])

  const cellFromPoint = (clientX: number, clientY: number): Cell | null => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || cols === 0) return null
    const rect = canvas.getBoundingClientRect()
    const j = Math.floor(((clientX - rect.left) / rect.width) * cols)
    const i = Math.floor(((clientY - rect.top) / rect.height) * rows)
    if (i < 0 || j < 0 || i >= rows || j >= cols) return null
    return { i, j }
  }

  return (
    <div
      className="relative overflow-hidden rounded-md bg-muted/40 ring-1 ring-foreground/10"
      onPointerMove={(event) => {
        const cell = cellFromPoint(event.clientX, event.clientY)
        const previous = lastCell.current
        if (cell?.i === previous?.i && cell?.j === previous?.j) return
        lastCell.current = cell
        onHover(cell)
      }}
      onPointerLeave={() => {
        lastCell.current = null
        onHover(null)
      }}
      onClick={(event) => {
        const cell = cellFromPoint(event.clientX, event.clientY)
        if (cell) onTogglePin(cell)
      }}
    >
      <canvas
        ref={canvasRef}
        width={cols}
        height={rows}
        aria-hidden
        className="block w-full"
        style={{ imageRendering: "pixelated" }}
      />
      {active ? (
        <div
          ref={activeRef}
          data-output-marker
          className="pointer-events-none absolute rounded-[1px] ring-1 ring-primary"
          style={{
            left: `${(active.j / cols) * 100}%`,
            top: `${(active.i / rows) * 100}%`,
            width: `${100 / cols}%`,
            height: `${100 / rows}%`,
          }}
        />
      ) : null}
    </div>
  )
}

export function CnnConvolution() {
  const [channelMode, setChannelMode] = React.useState(DEFAULTS.channelMode)
  const channels = channelMode === "rgb" ? 3 : 1
  const [scene, setScene] = React.useState(DEFAULTS.scene)
  const [size, setSize] = React.useState(DEFAULTS.size)
  const [operation, setOperation] = React.useState<Operation>(
    DEFAULTS.operation
  )
  const pooling = operation !== "convolution"
  const poolMode: PoolingMode = operation === "avg-pool" ? "avg" : "max"
  const poolLabel = poolMode === "max" ? "max" : "mean"
  const [kernelSize, setKernelSize] = React.useState(DEFAULTS.kernelSize)
  const [kernelPreset, setKernelPreset] = React.useState(DEFAULTS.kernelPreset)
  const [stride, setStride] = React.useState(DEFAULTS.stride)
  const [padding, setPadding] = React.useState<PaddingMode>(DEFAULTS.padding)
  const [padWidth, setPadWidth] = React.useState(DEFAULTS.padWidth)
  const [bias, setBias] = React.useState(DEFAULTS.bias)
  const [flip, setFlip] = React.useState(DEFAULTS.flip)
  const [showStride, setShowStride] = React.useState(DEFAULTS.showStride)
  const [speed, setSpeed] = React.useState(DEFAULTS.speed)
  const [outputChannel, setOutputChannel] = React.useState(0)

  const [input, setInput] = React.useState<number[][][]>(() =>
    buildScene(DEFAULTS.scene, DEFAULTS.size, 1)
  )
  const [kernels, setKernels] = React.useState<number[][][]>(() =>
    buildKernels(DEFAULTS.kernelPreset, DEFAULTS.kernelSize, 1)
  )

  const [level, setLevel] = React.useState(0.8)
  const [brush, setBrush] = React.useState("#38bdf8")
  const [hovered, setHovered] = React.useState<Cell | null>(null)
  const [pinned, setPinned] = React.useState<Cell | null>(null)
  const [playing, setPlaying] = React.useState(false)
  const [revealed, setRevealed] = React.useState(Number.POSITIVE_INFINITY)
  const [showNumbers, setShowNumbers] = React.useState(true)
  const [showHeatmap, setShowHeatmap] = React.useState(true)
  const [colorWeights, setColorWeights] = React.useState(true)
  const [source, setSource] = React.useState<InputSource>("synthetic")
  const [imageId, setImageId] = React.useState(DEFAULT_CNN_IMAGE_ID)
  const [photoMaxSize, setPhotoMaxSize] = React.useState(PHOTO_FALLBACK_MAX)
  const [imageState, setImageState] = React.useState<
    "idle" | "loading" | "ready" | "error"
  >("idle")
  const painting = React.useRef(false)
  const imageCache = React.useRef<Map<string, HTMLImageElement>>(new Map())
  const reducedMotion = usePrefersReducedMotion()
  const stagesRef = React.useRef<HTMLDivElement>(null)
  const inputVisualRef = React.useRef<HTMLDivElement>(null)
  const outputVisualRef = React.useRef<HTMLDivElement>(null)
  const windowFrameRef = React.useRef<HTMLDivElement>(null)
  const kernelAreaRef = React.useRef<HTMLDivElement>(null)
  const insightPanelRef = React.useRef<HTMLElement>(null)
  const counterRef = React.useRef<HTMLSpanElement>(null)
  const beamPathRef = React.useRef<SVGPathElement>(null)
  const beamDotRef = React.useRef<SVGCircleElement>(null)
  const [layoutTick, setLayoutTick] = React.useState(0)
  const photo = source === "photo" ? findCnnImage(imageId) : undefined
  const canvasMode = size > CANVAS_CELL_LIMIT
  const sizeMax = source === "photo" ? photoMaxSize : MAX_SYNTHETIC_SIZE
  const sizeStep = source === "photo" ? PHOTO_SIZE_STEP : 1

  const channelData = React.useMemo(() => toChannelFirst(input), [input])
  const result = React.useMemo(
    () =>
      operation === "convolution"
        ? convolveChannels(channelData, kernels, {
            stride,
            padding,
            pad: padWidth,
            bias,
            flip,
          })
        : poolChannels(channelData, kernelSize, {
            stride,
            padding,
            pad: padWidth,
            mode: poolMode,
          }),
    [
      operation,
      channelData,
      kernels,
      kernelSize,
      stride,
      padding,
      padWidth,
      bias,
      flip,
      poolMode,
    ]
  )

  const displayedChannel = pooling
    ? Math.min(outputChannel, Math.max(0, channels - 1))
    : 0
  const values = React.useMemo(() => {
    if (!result.valid) return [] as number[][]
    return "values" in result ? result.values : result.perChannel[displayedChannel] ?? []
  }, [result, displayedChannel])

  const paddedSize = size + 2 * result.pad
  const coveredRows = React.useMemo(
    () => windowCoverage(paddedSize, kernelSize, stride, result.rows),
    [paddedSize, kernelSize, stride, result.rows]
  )
  const coveredCols = React.useMemo(
    () => windowCoverage(paddedSize, kernelSize, stride, result.cols),
    [paddedSize, kernelSize, stride, result.cols]
  )
  const markSkipped = showStride && stride > 1 && result.valid
  const skippedPixels = React.useMemo(() => {
    if (!markSkipped) return 0
    let count = 0
    for (let pi = result.pad; pi < result.pad + size; pi += 1) {
      for (let pj = result.pad; pj < result.pad + size; pj += 1) {
        if (!coveredRows[pi] || !coveredCols[pj]) count += 1
      }
    }
    return count
  }, [markSkipped, result.pad, size, coveredRows, coveredCols])

  const strideComparison = React.useMemo(
    () =>
      [1, 2, 3, 4]
        .map((candidate) => {
          const dimension = outputDimension(
            size,
            kernelSize,
            candidate,
            result.pad
          )
          return { stride: candidate, dimension }
        })
        .filter((entry) => entry.dimension > 0),
    [size, kernelSize, result.pad]
  )

  const total = result.valid ? result.rows * result.cols : 0
  const visibleCount = Math.min(
    Number.isFinite(revealed) ? revealed : total,
    total
  )
  /** One cell per tick on small grids; bigger grids reveal in chunks. */
  const revealStep = Math.max(1, Math.ceil(total / SWEEP_TICKS))
  const sweepCell: Cell | null = React.useMemo(
    () =>
      playing && visibleCount > 0
        ? {
            i: Math.floor((visibleCount - 1) / result.cols),
            j: (visibleCount - 1) % result.cols,
          }
        : null,
    [playing, visibleCount, result.cols]
  )
  const rawActive = pinned ?? (playing ? sweepCell : hovered)
  const active: Cell | null = React.useMemo(() => {
    if (!rawActive || !result.valid) return null
    if (rawActive.i < 0 || rawActive.i >= result.rows) return null
    if (rawActive.j < 0 || rawActive.j >= result.cols) return null
    return rawActive
  }, [rawActive, result])

  const cursor = React.useRef(0)

  React.useEffect(() => {
    if (!playing || total <= 0) return
    const id = window.setInterval(() => {
      cursor.current = Math.min(total, cursor.current + revealStep)
      setRevealed(cursor.current)
      if (cursor.current >= total) {
        window.clearInterval(id)
        setPlaying(false)
      }
    }, speed)
    return () => window.clearInterval(id)
  }, [playing, speed, total, revealStep])

  const resetAnimation = React.useCallback(() => {
    cursor.current = 0
    setPlaying(false)
    setRevealed(Number.POSITIVE_INFINITY)
    setPinned(null)
    setHovered(null)
  }, [])

  React.useEffect(() => {
    if (source !== "photo") return
    const photo = findCnnImage(imageId)
    if (!photo) return

    let cancelled = false

    const sample = async () => {
      try {
        let element = imageCache.current.get(photo.id)
        if (!element) {
          element = await loadImageElement(photo.src)
          imageCache.current.set(photo.id, element)
        }
        if (cancelled) return
        const native =
          Math.max(element.naturalWidth, element.naturalHeight) ||
          PHOTO_FALLBACK_MAX
        setPhotoMaxSize(Math.max(4, native))
        if (size > native) {
          setSize(native)
          return
        }
        setInput(imageToGrid(element, size, channels))
        setImageState("ready")
        resetAnimation()
      } catch {
        if (!cancelled) setImageState("error")
      }
    }

    void sample()
    return () => {
      cancelled = true
    }
  }, [source, imageId, size, channels, resetAnimation])

  /** Keeps the animated geometry in sync when the stage width changes. */
  React.useEffect(() => {
    const stages = stagesRef.current
    if (!stages || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(() => {
      setLayoutTick((tick) => tick + 1)
    })
    observer.observe(stages)
    return () => observer.disconnect()
  }, [])

  const activeKey = active ? `${active.i}:${active.j}` : null
  const lastWindowKey = React.useRef<string | null>(null)
  const lastWindowTick = React.useRef(0)

  /**
   * The receptive field glides with anime.js instead of jumping between
   * cells: one smooth frame that slides over the input while the sweep runs.
   */
  useIsomorphicLayoutEffect(() => {
    const frame = windowFrameRef.current
    const container = inputVisualRef.current
    if (!frame || !container) return

    const layoutChanged = lastWindowTick.current !== layoutTick
    const firstPlacement = lastWindowKey.current === null
    lastWindowTick.current = layoutTick
    lastWindowKey.current = activeKey

    if (!active) {
      if (reducedMotion) utils.set(frame, { opacity: 0 })
      else animate(frame, { opacity: 0, duration: 160, ease: "out(2)" })
      return
    }

    const gap = canvasMode ? 0 : INPUT_CELL_GAP
    const width = container.getBoundingClientRect().width
    const cell = (width - gap * (paddedSize - 1)) / paddedSize
    if (!Number.isFinite(cell) || cell <= 0) return

    const x = active.j * stride * (cell + gap)
    const y = active.i * stride * (cell + gap)
    const size = kernelSize * cell + Math.max(0, kernelSize - 1) * gap

    if (reducedMotion) {
      utils.set(frame, {
        translateX: x,
        translateY: y,
        width: size,
        height: size,
        opacity: 1,
      })
      return
    }

    const travel = playing ? clampNumber(speed * 0.75, 120, 600) : 170
    if (layoutChanged || firstPlacement) {
      utils.set(frame, {
        translateX: x,
        translateY: y,
        width: size,
        height: size,
        opacity: 1,
      })
    } else {
      animate(frame, {
        translateX: x,
        translateY: y,
        width: size,
        height: size,
        opacity: 1,
        duration: travel,
        ease: playing ? "inOutSine" : "out(3)",
      })
    }
  }, [
    active,
    activeKey,
    layoutTick,
    canvasMode,
    kernelSize,
    stride,
    paddedSize,
    playing,
    speed,
    reducedMotion,
  ])

  const previousVisibleRef = React.useRef(0)

  /** Each newly computed output cell pops in — the sweep writes the map. */
  useIsomorphicLayoutEffect(() => {
    const count = visibleCount
    const previous = previousVisibleRef.current
    previousVisibleRef.current = count
    if (reducedMotion || canvasMode) return
    if (!Number.isFinite(count) || count <= previous) return

    const container = outputVisualRef.current
    if (!container) return
    const cells = Array.from(
      container.querySelectorAll<HTMLElement>("[data-output-cell]")
    )
    const newly = cells.slice(Math.max(0, Math.floor(previous)), Math.floor(count))
    if (newly.length === 0) return

    const popDuration = playing
      ? Math.round(clampNumber(speed * 0.4, 150, 320))
      : 300
    const animation = animate(newly, {
      opacity: [0, 1],
      scale: [0.4, 1],
      duration: popDuration,
      delay: playing ? Math.round(speed * 0.45) : stagger(newly.length > 24 ? 8 : 16),
      ease: "out(3)",
    })
    return () => {
      animation.pause()
      utils.set(newly, { opacity: 1, scale: 1 })
    }
  }, [visibleCount, canvasMode, playing, speed, reducedMotion])

  /** The kernel (or pooling window) ripples as its weights are applied. */
  useIsomorphicLayoutEffect(() => {
    const container = kernelAreaRef.current
    if (!container || reducedMotion || !active) return
    const cells = Array.from(
      container.querySelectorAll<HTMLElement>("[data-pulse-cell]")
    )
    if (cells.length === 0) return

    const animation = animate(cells, {
      scale: [
        { to: 1.09, duration: playing ? 90 : 80 },
        { to: 1, duration: playing ? 130 : 120 },
      ],
      delay: stagger(
        [0, playing ? Math.min(180, Math.max(60, speed * 0.4)) : 90],
        { grid: [kernelSize, kernelSize], from: "first" }
      ),
      ease: "inOutQuad",
    })
    return () => {
      animation.revert()
    }
  }, [active, kernelSize, playing, speed, reducedMotion])

  /** Term chips land one after another; the result value gets the last word. */
  useIsomorphicLayoutEffect(() => {
    const panel = insightPanelRef.current
    if (!panel || reducedMotion || !active) return
    const chips = Array.from(
      panel.querySelectorAll<HTMLElement>("[data-term-chip]")
    )
    const sum = panel.querySelector<HTMLElement>("[data-sum-value]")
    const animations: ReturnType<typeof animate>[] = []
    if (chips.length > 0) {
      animations.push(
        animate(chips, {
          opacity: [0.35, 1],
          translateY: [6, 0],
          duration: 260,
          delay: stagger(14),
          ease: "out(3)",
        })
      )
    }
    if (sum) {
      animations.push(
        animate(sum, { scale: [0.8, 1], duration: 300, ease: "out(3)" })
      )
    }
    return () => {
      for (const animation of animations) animation.revert()
    }
  }, [active, reducedMotion])

  useIsomorphicLayoutEffect(() => {
    const badge = counterRef.current
    if (!badge || reducedMotion) return
    const animation = animate(badge, {
      scale: [1.18, 1],
      duration: 240,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [visibleCount, reducedMotion])

  /**
   * A patch-to-cell beam: the dot carries the weighted sum from the active
   * receptive field into the output cell it produces.
   */
  useIsomorphicLayoutEffect(() => {
    const path = beamPathRef.current
    const dot = beamDotRef.current
    if (!path || !dot) return

    const hide = () => utils.set([path, dot], { opacity: 0 })

    const stages = stagesRef.current
    const input = inputVisualRef.current
    const output = outputVisualRef.current
    if (!active || reducedMotion || !stages || !input || !output) {
      hide()
      return
    }

    const stagesRect = stages.getBoundingClientRect()
    const inputRect = input.getBoundingClientRect()
    const outputRect = output.getBoundingClientRect()
    // Stacked stages: the beam would cross the kernel column, so skip it.
    if (outputRect.top >= inputRect.bottom - 4) {
      hide()
      return
    }

    const gap = canvasMode ? 0 : INPUT_CELL_GAP
    const cell = (inputRect.width - gap * (paddedSize - 1)) / paddedSize
    if (!Number.isFinite(cell) || cell <= 0) return

    const frameX =
      inputRect.left - stagesRect.left + active.j * stride * (cell + gap)
    const frameY =
      inputRect.top - stagesRect.top + active.i * stride * (cell + gap)
    const frameSize = kernelSize * cell + Math.max(0, kernelSize - 1) * gap
    const startX = frameX + frameSize - 1
    const startY = frameY + frameSize / 2

    let endX: number
    let endY: number
    if (canvasMode) {
      endX =
        outputRect.left -
        stagesRect.left +
        ((active.j + 0.5) / result.cols) * outputRect.width
      endY =
        outputRect.top -
        stagesRect.top +
        ((active.i + 0.5) / result.rows) * outputRect.height
    } else {
      const index = active.i * result.cols + active.j
      const element = output.querySelector<HTMLElement>(
        `[data-output-cell][data-index="${index}"]`
      )
      if (!element) {
        hide()
        return
      }
      const rect = element.getBoundingClientRect()
      endX = rect.left - stagesRect.left
      endY = rect.top - stagesRect.top + rect.height / 2
    }

    const bend = Math.max(28, (endX - startX) * 0.45)
    path.setAttribute(
      "d",
      `M ${startX} ${startY} C ${startX + bend} ${startY}, ${endX - bend} ${endY}, ${endX} ${endY}`
    )
    const length = path.getTotalLength()
    if (!Number.isFinite(length) || length < 1) {
      hide()
      return
    }
    path.setAttribute("stroke-dasharray", `${length}`)

    const duration = playing
      ? Math.round(clampNumber(speed * 0.6, 140, 700))
      : 620
    const draw = animate(path, {
      strokeDashoffset: [length, 0],
      opacity: [0, 0.85],
      duration,
      ease: "inOutSine",
      onComplete: () => {
        animate(path, { opacity: 0, duration: 220, delay: 140 })
      },
    })
    const flight = animate(dot, {
      ...createMotionPath(path, 0.02),
      opacity: [0, 1],
      duration,
      ease: "inOutSine",
      onComplete: () => {
        animate(dot, { opacity: 0, duration: 200, delay: 120 })
      },
    })

    return () => {
      draw.pause()
      flight.pause()
      hide()
    }
  }, [
    active,
    activeKey,
    layoutTick,
    canvasMode,
    kernelSize,
    stride,
    paddedSize,
    playing,
    speed,
    reducedMotion,
    result.cols,
    result.rows,
  ])

  const loadScene = (nextScene: string, nextSize = size, nextChannels = channels) => {
    setInput(buildScene(nextScene, nextSize, nextChannels))
    resetAnimation()
  }

  const loadKernels = (nextPreset: string, nextSize = kernelSize, nextChannels = channels) => {
    setKernels(buildKernels(nextPreset, nextSize, nextChannels))
  }

  const paint = (cell: Cell) => {
    setInput((current) => {
      const pixel = current[cell.i]?.[cell.j]
      if (!pixel) return current
      const nextPixel: number[] = channels === 1 ? [level] : hexToRgb(brush)
      const nextRow = current[cell.i].map((value, j) =>
        j === cell.j ? nextPixel : value
      )
      return current.map((row, i) => (i === cell.i ? nextRow : row))
    })
  }

  const changeChannelMode = (value: string) => {
    if (value !== "grayscale" && value !== "rgb") return
    const nextChannels = value === "rgb" ? 3 : 1
    setChannelMode(value)
    setOutputChannel(0)
    setKernels(buildKernels(kernelPreset, kernelSize, nextChannels))
    if (source === "photo") {
      // The sampling effect re-reads the cached bitmap at the new depth.
      setImageState("loading")
    } else {
      setInput(buildScene(scene, size, nextChannels))
      resetAnimation()
    }
  }

  const changeOperation = (value: string) => {
    if (
      value !== "convolution" &&
      value !== "max-pool" &&
      value !== "avg-pool"
    )
      return
    setOperation(value)
    setOutputChannel(0)
    // Classic pooling downsamples: 3×3 window, stride 2.
    if (value !== "convolution" && stride === 1) setStride(2)
    resetAnimation()
  }

  const changeSize = (value: number) => {
    setSize(value)
    if (source === "photo") {
      setImageState("loading")
    } else {
      setInput(buildScene(scene, value, channels))
      resetAnimation()
    }
  }

  const changeSource = (value: string) => {
    if (value !== "synthetic" && value !== "photo") return
    if (value === source) return
    setSource(value)
    if (value === "photo") {
      setSize(PHOTO_GRID_SIZE)
      setImageState("loading")
    } else {
      setSize(DEFAULTS.size)
      setInput(buildScene(scene, DEFAULTS.size, channels))
      setKernels(buildKernels(kernelPreset, kernelSize, channels))
      setImageState("idle")
      resetAnimation()
    }
  }

  const changeImage = (value: string) => {
    setImageId(value)
    setImageState("loading")
    resetAnimation()
  }

  const changeKernelSize = (value: number) => {
    setKernelSize(value)
    setKernels(buildKernels(kernelPreset, value, channels))
    resetAnimation()
  }

  const changeKernelPreset = (value: string) => {
    setKernelPreset(value)
    loadKernels(value)
  }

  const editWeight = (channel: number, row: number, col: number, value: number) => {
    setKernels((current) =>
      current.map((kernel, channelIndex) =>
        channelIndex === channel
          ? kernel.map((kernelRow, rowIndex) =>
              rowIndex === row
                ? kernelRow.map((weight, colIndex) =>
                    colIndex === col ? value : weight
                  )
                : kernelRow
            )
          : kernel
      )
    )
  }

  const startSweep = () => {
    cursor.current = 0
    setPinned(null)
    setRevealed(0)
    setPlaying(true)
  }

  const step = () => {
    setPlaying(false)
    const next =
      total > 0 && cursor.current >= total
        ? revealStep
        : cursor.current + revealStep
    cursor.current = Math.min(total, next)
    setRevealed(cursor.current)
  }

  const togglePin = (cell: Cell) => {
    setPinned((current) =>
      current && current.i === cell.i && current.j === cell.j ? null : cell
    )
  }

  const resetAll = () => {
    setSource("synthetic")
    setImageId(DEFAULT_CNN_IMAGE_ID)
    setImageState("idle")
    setChannelMode(DEFAULTS.channelMode)
    setScene(DEFAULTS.scene)
    setSize(DEFAULTS.size)
    setOperation(DEFAULTS.operation)
    setKernelSize(DEFAULTS.kernelSize)
    setKernelPreset(DEFAULTS.kernelPreset)
    setStride(DEFAULTS.stride)
    setPadding(DEFAULTS.padding)
    setPadWidth(DEFAULTS.padWidth)
    setBias(DEFAULTS.bias)
    setFlip(DEFAULTS.flip)
    setShowStride(DEFAULTS.showStride)
    setSpeed(DEFAULTS.speed)
    setOutputChannel(0)
    setInput(buildScene(DEFAULTS.scene, DEFAULTS.size, 1))
    setKernels(buildKernels(DEFAULTS.kernelPreset, DEFAULTS.kernelSize, 1))
    setLevel(0.8)
    setBrush("#38bdf8")
    resetAnimation()
  }

  const moveActive = (deltaI: number, deltaJ: number) => {
    if (!result.valid) return
    const current = pinned ?? hovered ?? { i: 0, j: 0 }
    const next = {
      i: clampNumber(current.i + deltaI, 0, result.rows - 1),
      j: clampNumber(current.j + deltaJ, 0, result.cols - 1),
    }
    if (pinned) setPinned(next)
    else setHovered(next)
  }

  const bound = React.useMemo(() => {
    if (!result.valid) return 1
    let max = 1e-6
    for (const row of values) {
      for (const value of row) max = Math.max(max, Math.abs(value))
    }
    return max
  }, [result.valid, values])

  const arithmetic =
    !pooling && active && result.valid
      ? channelData.map((channel, channelIndex) => {
          const window = extractWindow(
            channel,
            active.i,
            active.j,
            kernelSize,
            stride,
            result.pad
          )
          const weights = effectiveKernel(
            kernels[channelIndex] ?? [],
            flip
          )
          const terms = window.map((row, m) =>
            row.map(
              (value, n) => value * (weights[m]?.[n] ?? 0)
            )
          )
          const sum = terms.reduce(
            (totalTerms, row) =>
              totalTerms + row.reduce((rowSum, value) => rowSum + value, 0),
            0
          )
          return { channel: channelIndex, window, weights, terms, sum }
        })
      : []
  const arithmeticTotal = arithmetic.reduce(
    (totalTerms, entry) => totalTerms + entry.sum,
    0
  )

  const poolWindows = React.useMemo(() => {
    if (!pooling || !active || !result.valid) return []
    return channelData.map((channel, channelIndex) => {
      const window = extractWindow(
        channel,
        active.i,
        active.j,
        kernelSize,
        stride,
        result.pad
      )
      const flat = window.flat()
      const sum = flat.reduce((totalTerms, value) => totalTerms + value, 0)
      const max = flat.reduce(
        (best, value) => (value > best ? value : best),
        Number.NEGATIVE_INFINITY
      )
      const mean = sum / Math.max(1, flat.length)
      return {
        channel: channelIndex,
        window,
        sum,
        max,
        mean,
        result: poolMode === "max" ? max : mean,
      }
    })
  }, [
    pooling,
    active,
    result.valid,
    result.pad,
    channelData,
    kernelSize,
    stride,
    poolMode,
  ])
  const activePoolWindow =
    poolWindows[displayedChannel] ?? poolWindows[0] ?? null

  const totalKernelSum = kernels.reduce(
    (sum, kernel) => sum + kernelSum(kernel),
    0
  )
  const cost = total * channels * kernelSize * kernelSize
  const poolCost = total * kernelSize * kernelSize
  const padLabel = result.pad

  return (
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1fr)_19rem]">
      <div className="grid content-start gap-4">
        <Card className="overflow-hidden">
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              Convolution Playground
            <InfoTip>
              <span className="block">
                Slide the kernel across the input: every output cell is one
                weighted sum over its receptive field. Switch the layer type to
                pooling to swap the weights for a max/mean reduction.
              </span>
              <span className="mt-1 block font-mono text-[11px]">
                conv: out[i,j] = b + Σ_c Σ_m Σ_n I_c[i·s+m−p, j·s+n−p] · K_c[m,n]
              </span>
              <span className="mt-1 block font-mono text-[11px]">
                pool: out[i,j] = max / mean of the same window
              </span>
            </InfoTip>
            </CardTitle>
            <CardAction>
              <div className="flex flex-wrap items-center justify-end gap-2">
                {playing ? (
                  <Button size="sm" variant="outline" onClick={() => setPlaying(false)}>
                    <PauseIcon data-icon="inline-start" />
                    Pause
                  </Button>
                ) : (
                  <Button size="sm" onClick={startSweep}>
                    <PlayIcon data-icon="inline-start" />
                    Run sweep
                  </Button>
                )}
                <Button size="sm" variant="outline" onClick={step} disabled={total <= 0}>
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
                  {visibleCount} / {total}
                </Badge>
              </div>
            </CardAction>
          </CardHeader>
          <CardContent
            ref={stagesRef}
            className="relative @container/stages grid gap-4"
          >
            <svg
              aria-hidden
              className="pointer-events-none absolute inset-0 z-20 size-full"
            >
              <defs>
                <linearGradient id="cnn-patch-beam" x1="0" y1="0" x2="1" y2="0">
                  <stop
                    offset="0%"
                    style={{ stopColor: "var(--primary)", stopOpacity: 0.15 }}
                  />
                  <stop
                    offset="55%"
                    style={{ stopColor: "var(--primary)", stopOpacity: 0.9 }}
                  />
                  <stop
                    offset="100%"
                    style={{ stopColor: "var(--primary)", stopOpacity: 0.3 }}
                  />
                </linearGradient>
              </defs>
              <path
                ref={beamPathRef}
                fill="none"
                stroke="url(#cnn-patch-beam)"
                strokeWidth={1.5}
                strokeLinecap="round"
                opacity={0}
              />
              <circle
                ref={beamDotRef}
                r={3.5}
                style={{ fill: "var(--primary)" }}
                opacity={0}
              />
            </svg>
            <div className="grid gap-5 @xl/stages:grid-cols-3 @xl/stages:gap-4">
              <section className="grid content-start gap-2">
                <header className="flex min-h-6 items-center justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-2 text-sm font-medium">
                    <Badge variant="outline">1</Badge>
                    Input
                    <span className="truncate font-mono text-xs font-normal text-muted-foreground">
                      {size}×{size}×{channels}
                      {result.pad > 0 ? ` · p=${result.pad}` : ""}
                    </span>
                    <InfoTip>
                      {photo
                        ? `${photo.label} is resampled onto this grid. `
                        : "Click and drag to paint. "}
                      {result.pad > 0
                        ? `The dashed ring is zero padding: reads there count as 0 (p = ${padLabel}). `
                        : "Reads beyond the border count as 0 (padding p = 0). "}
                      {markSkipped
                        ? canvasMode
                          ? "At this stride some pixels are never read by any window. "
                          : "Faded pixels are never read by any window at this stride. "
                        : ""}
                      {canvasMode
                        ? ` Above ${CANVAS_CELL_LIMIT}×${CANVAS_CELL_LIMIT} the grid is drawn on a canvas, so painting and per-cell numbers are off.`
                        : ""}
                    </InfoTip>
                  </span>
                  {source === "photo" && imageState === "loading" ? (
                    <span className="shrink-0 text-[11px] text-muted-foreground">
                      Sampling…
                    </span>
                  ) : null}
                </header>
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
                      <span className="shrink-0 text-muted-foreground">
                        {size}×{size}
                        {size === photoMaxSize ? " (native)" : ""}
                      </span>
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
                <div ref={inputVisualRef} className="relative">
                  {canvasMode ? (
                    <CanvasInputGrid
                      input={input}
                      channels={channels}
                      pad={result.pad}
                    />
                  ) : (
                    <InputGrid
                      input={input}
                      channels={channels}
                      active={active}
                      kernelSize={kernelSize}
                      stride={stride}
                      pad={result.pad}
                      showNumbers={showNumbers}
                      coveredRows={coveredRows}
                      coveredCols={coveredCols}
                      markSkipped={markSkipped}
                      onPaintStart={(cell) => {
                        painting.current = true
                        paint(cell)
                      }}
                      onPaintMove={(cell) => {
                        if (painting.current && cell) paint(cell)
                      }}
                      onPaintEnd={() => {
                        painting.current = false
                      }}
                    />
                  )}
                  <div
                    ref={windowFrameRef}
                    aria-hidden
                    data-window-frame
                    className="pointer-events-none absolute top-0 left-0 z-20 rounded-[5px] bg-primary/10 opacity-0 ring-2 ring-primary"
                  />
                </div>
              </section>

              <section className="grid content-start gap-2">
                <header className="flex min-h-6 items-center justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-2 text-sm font-medium">
                    <Badge variant="outline">2</Badge>
                    {pooling ? "Window" : "Kernel"}
                    <span className="truncate font-mono text-xs font-normal text-muted-foreground">
                      {kernelSize}×{kernelSize}
                      {pooling ? "" : `×${channels}`}
                    </span>
                    <InfoTip>
                      {pooling ? (
                        <>
                          Pooling has no weights: the window slides like a
                          kernel and reduces each patch to one number. Hover an
                          output cell to inspect the patch under it.
                        </>
                      ) : (
                        <>
                          Drag a weight horizontally to change it, double-click
                          to zero it.
                          {flip
                            ? " Flip is on: the kernel rotates 180° before it slides."
                            : ""}
                        </>
                      )}
                    </InfoTip>
                  </span>
                  <span className="shrink-0 font-mono text-xs text-muted-foreground tabular-nums">
                    {pooling ? poolLabel : `ΣK = ${formatValue(totalKernelSum)}`}
                  </span>
                </header>
                {pooling ? (
                  <div ref={kernelAreaRef} className="grid gap-3">
                    <div
                      className="grid max-w-64 gap-[3px]"
                      style={{
                        gridTemplateColumns: `repeat(${kernelSize}, minmax(0, 1fr))`,
                      }}
                    >
                      {Array.from({ length: kernelSize }, (_, m) =>
                        Array.from({ length: kernelSize }, (_, n) => {
                          const value = activePoolWindow?.window[m]?.[n]
                          const isWinner =
                            poolMode === "max" &&
                            value !== undefined &&
                            activePoolWindow !== null &&
                            Math.abs(value - activePoolWindow.max) < 1e-9
                          return (
                            <div
                              key={`${m}-${n}`}
                              data-pulse-cell
                              style={
                                isWinner
                                  ? {
                                      backgroundColor: heatColor(
                                        value,
                                        bound
                                      ),
                                      color: heatTextColor(value, bound),
                                    }
                                  : undefined
                              }
                              className={cn(
                                "flex aspect-square min-w-0 items-center justify-center rounded-[3px] text-[10px] font-mono tabular-nums ring-1 ring-foreground/10",
                                value === undefined
                                  ? "bg-muted/60 text-muted-foreground"
                                  : "bg-muted/40",
                                isWinner && "ring-2 ring-primary"
                              )}
                            >
                              {value === undefined ? poolLabel : formatValue(value)}
                            </div>
                          )
                        })
                      )}
                    </div>
                    <p className="font-mono text-xs text-muted-foreground tabular-nums">
                      {activePoolWindow
                        ? `${poolLabel} = ${formatValue(activePoolWindow.result)}${
                            poolMode === "avg"
                              ? ` (Σ ${formatValue(activePoolWindow.sum)} / ${
                                  kernelSize * kernelSize
                                })`
                              : ""
                          }`
                        : `${kernelSize}×${kernelSize} → 1 · no weights`}
                    </p>
                  </div>
                ) : (
                  <div ref={kernelAreaRef} className="grid gap-3">
                    {kernels.map((kernel, channelIndex) => (
                      <div key={channelIndex} className="grid gap-1">
                        {channels > 1 ? (
                          <span
                            className="flex items-center gap-2 text-[11px] font-medium"
                            style={{ color: CHANNEL_COLOR[channelIndex] }}
                          >
                            {CHANNEL_LABEL[channelIndex]}
                            <span className="text-muted-foreground">
                              ΣK = {formatValue(kernelSum(kernel))}
                            </span>
                          </span>
                        ) : null}
                        <div
                          className="grid max-w-64 gap-[3px]"
                          style={{
                            gridTemplateColumns: `repeat(${kernelSize}, minmax(0, 1fr))`,
                          }}
                        >
                          {kernel.map((row, m) =>
                            row.map((weight, n) => (
                              <KernelCell
                                key={`${channelIndex}-${m}-${n}`}
                                value={weight}
                                colorWeights={colorWeights}
                                onChange={(value) =>
                                  editWeight(channelIndex, m, n, value)
                                }
                              />
                            ))
                          )}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </section>

              <section className="grid content-start gap-2">
                <header className="flex min-h-6 items-center justify-between gap-2">
                  <span className="flex min-w-0 items-center gap-2 text-sm font-medium">
                    <Badge variant="outline">3</Badge>
                    Output
                    <span className="truncate font-mono text-xs font-normal text-muted-foreground">
                      {result.valid ? `${result.rows}×${result.cols}` : "—"}
                      {pooling && channels > 1
                        ? ` ×${channels}`
                        : ""}
                    </span>
                    <InfoTip>
                      Hover a cell to see its receptive field, click to pin it;
                      arrow keys nudge the pinned cell.
                      {pooling && channels > 1
                        ? " Pooling keeps the channels separate — pick the one to inspect above the grid."
                        : ""}
                    </InfoTip>
                  </span>
                  {pooling && channels > 1 ? (
                    <span className="flex shrink-0 items-center gap-1">
                      {CHANNEL_LABEL.map((label, index) => (
                        <Button
                          key={label}
                          type="button"
                          size="sm"
                          variant={
                            displayedChannel === index ? "default" : "outline"
                          }
                          className="h-6 px-2 text-[11px]"
                          onClick={() => setOutputChannel(index)}
                        >
                          {label}
                        </Button>
                      ))}
                    </span>
                  ) : null}
                </header>
                {source === "photo" && imageState !== "ready" ? (
                  <p className="text-sm text-muted-foreground">
                    {imageState === "error"
                      ? "No image loaded — pick another photo."
                      : "Sampling the photo onto the grid…"}
                  </p>
                ) : result.valid ? (
                  <>
                    <div
                      role="grid"
                      aria-label="Convolution output, use arrow keys to inspect cells"
                      tabIndex={0}
                      onKeyDown={(event) => {
                        if (event.key === "ArrowLeft") moveActive(0, -1)
                        else if (event.key === "ArrowRight") moveActive(0, 1)
                        else if (event.key === "ArrowUp") moveActive(-1, 0)
                        else if (event.key === "ArrowDown") moveActive(1, 0)
                        else return
                        event.preventDefault()
                      }}
                      className="rounded-md outline-none focus-visible:ring-2 focus-visible:ring-ring"
                    >
                      <div ref={outputVisualRef} className="relative">
                      {canvasMode ? (
                        <CanvasOutputGrid
                          values={values}
                          rows={result.rows}
                          cols={result.cols}
                          bound={bound}
                          visibleCount={visibleCount}
                          showHeatmap={showHeatmap}
                          active={active}
                          reducedMotion={reducedMotion}
                          onHover={setHovered}
                          onTogglePin={togglePin}
                        />
                      ) : (
                      <div
                        className="grid gap-[3px]"
                        style={{
                          gridTemplateColumns: `repeat(${result.cols}, minmax(0, 1fr))`,
                        }}
                      >
                      {values.map((row, i) =>
                        row.map((value, j) => {
                          const index = i * result.cols + j
                          const isRevealed = index < visibleCount
                          const isActive =
                            active !== null && active.i === i && active.j === j
                          return (
                            <button
                              key={`${i}-${j}`}
                              type="button"
                              data-output-cell
                              data-index={index}
                              onMouseEnter={() => setHovered({ i, j })}
                              onClick={() => togglePin({ i, j })}
                              style={
                                isRevealed && showHeatmap
                                  ? {
                                      backgroundColor: heatColor(value, bound),
                                      color: heatTextColor(value, bound),
                                    }
                                  : undefined
                              }
                              className={cn(
                                "flex aspect-square min-w-0 items-center justify-center rounded-[3px] bg-muted/40 text-[10px] font-mono tabular-nums ring-1 ring-foreground/10 transition-shadow hover:ring-foreground/40",
                                !isRevealed && "text-transparent",
                                isActive && "z-10 ring-2 ring-primary"
                              )}
                            >
                              {isRevealed && showNumbers ? formatValue(value) : null}
                            </button>
                          )
                        })
                      )}
                      </div>
                      )}
                      </div>
                    </div>
                  </>
                ) : (
                  <Alert>
                    <LightbulbIcon />
                    <AlertTitle>
                      {pooling ? "This window does not fit" : "This kernel does not fit"}
                    </AlertTitle>
                    <AlertDescription>
                      A {kernelSize}×{kernelSize} {pooling ? "window" : "kernel"}{" "}
                      with {padding} padding and stride {stride} produces an output
                      of {Math.max(0, result.rows)}×{Math.max(0, result.cols)}.{" "}
                      {padding === "same"
                        ? "Grow the input or lower the stride."
                        : "Switch to Same padding, lower the stride or grow the input."}
                    </AlertDescription>
                  </Alert>
                )}
              </section>
            </div>

            {active && pooling && poolWindows.length > 0 ? (
              <section
                ref={insightPanelRef}
                className="grid gap-3 rounded-lg bg-muted/50 p-3 ring-1 ring-foreground/10 @container/arithmetic"
              >
                <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <span className="flex items-center gap-1.5 text-sm font-medium">
                    Output cell
                    <span className="font-mono">
                      ({active.i}, {active.j})
                    </span>
                    <InfoTip>
                      {poolMode === "max"
                        ? "The window keeps its largest value — the strongest activation in the patch."
                        : "The window averages all of its values. Padded zeros count toward the average."}
                    </InfoTip>
                  </span>
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    {channels > 1
                      ? `${CHANNEL_LABEL[displayedChannel]} · `
                      : ""}
                    {poolLabel} ={" "}
                    <span data-sum-value className="inline-block font-semibold text-foreground">
                      {activePoolWindow
                        ? formatValue(activePoolWindow.result)
                        : "—"}
                    </span>
                  </span>
                </header>
                <div className="grid gap-3 @lg/arithmetic:grid-cols-2">
                  {poolWindows.map((entry) => (
                    <div key={entry.channel} className="grid content-start gap-1.5">
                      <div className="flex items-center gap-2 text-xs">
                        <span
                          className="font-medium"
                          style={{
                            color:
                              channels > 1
                                ? CHANNEL_COLOR[entry.channel]
                                : undefined,
                          }}
                        >
                          {channels > 1
                            ? CHANNEL_LABEL[entry.channel]
                            : "Grayscale"}
                        </span>
                        <span className="text-muted-foreground tabular-nums">
                          {poolMode === "max"
                            ? `max = ${formatValue(entry.max)}`
                            : `Σ ${formatValue(entry.sum)} / ${
                                kernelSize * kernelSize
                              } = ${formatValue(entry.mean)}`}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1 font-mono text-[11px]">
                        {entry.window.map((row, m) =>
                          row.map((value, n) => {
                            const isWinner =
                              poolMode === "max" &&
                              Math.abs(value - entry.max) < 1e-9
                            return (
                              <span
                                key={`${m}-${n}`}
                                data-term-chip
                                className={cn(
                                  "rounded bg-background/80 px-1.5 py-0.5 ring-1 ring-foreground/10 whitespace-nowrap",
                                  isWinner &&
                                    "font-semibold text-foreground ring-primary"
                                )}
                              >
                                {formatValue(value)}
                              </span>
                            )
                          })
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            ) : active && arithmetic.length > 0 ? (
              <section
                ref={insightPanelRef}
                className="grid gap-3 rounded-lg bg-muted/50 p-3 ring-1 ring-foreground/10 @container/arithmetic"
              >
                <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
                  <span className="flex items-center gap-1.5 text-sm font-medium">
                    Output cell
                    <span className="font-mono">
                      ({active.i}, {active.j})
                    </span>
                    <InfoTip>
                      The kernel dotted with this receptive field, plus the bias.
                    </InfoTip>
                  </span>
                  <span className="font-mono text-xs text-muted-foreground tabular-nums">
                    Σ channels {formatValue(arithmeticTotal)} + bias{" "}
                    {formatValue(bias)} ={" "}
                    <span data-sum-value className="inline-block font-semibold text-foreground">
                      {formatValue(arithmeticTotal + bias)}
                    </span>
                  </span>
                </header>
                <div className="grid gap-3 @lg/arithmetic:grid-cols-2">
                  {arithmetic.map((entry) => (
                    <div key={entry.channel} className="grid content-start gap-1.5">
                      <div className="flex items-center gap-2 text-xs">
                        <span
                          className="font-medium"
                          style={{
                            color:
                              channels > 1
                                ? CHANNEL_COLOR[entry.channel]
                                : undefined,
                          }}
                        >
                          {channels > 1
                            ? CHANNEL_LABEL[entry.channel]
                            : "Grayscale"}
                        </span>
                        <span className="text-muted-foreground tabular-nums">
                          Σ = {formatValue(entry.sum)}
                        </span>
                      </div>
                      <div className="flex flex-wrap gap-1 font-mono text-[11px]">
                        {entry.window.map((row, m) =>
                          row.map((value, n) => (
                            <span
                              key={`${m}-${n}`}
                              data-term-chip
                              className="rounded bg-background/80 px-1.5 py-0.5 ring-1 ring-foreground/10 whitespace-nowrap"
                            >
                              {formatValue(value)}·
                              {formatValue(entry.weights[m]?.[n] ?? 0)}
                              <span className="ml-1 text-muted-foreground">
                                = {formatValue(entry.terms[m][n])}
                              </span>
                            </span>
                          ))
                        )}
                      </div>
                    </div>
                  ))}
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
                {photo ? (
                  <p>
                    {photo.label} arrives as a bitmap and is resampled to a{" "}
                    {size}×{size}×{channels} tensor = {size * size * channels}{" "}
                    numbers — the same shape the synthetic scenes have, so every
                    kernel and stride setting behaves identically.
                  </p>
                ) : null}
                <p>
                  Output {result.valid ? `${result.rows}×${result.cols}` : "n/a"}{" "}
                  from ⌊({size} + 2·{result.pad} − {kernelSize}) / {stride}⌋ + 1.
                </p>
                {result.pad > 0 ? (
                  <p>
                    Padding p = {result.pad} wraps the input in a ring of zeros,
                    so border windows read 0 instead of running out of grid —
                    the dashed cells on the input.
                  </p>
                ) : null}
                {stride > 1 ? (
                  <>
                    <p>
                      Output by stride:{" "}
                      {strideComparison
                        .map(
                          (entry) =>
                            `${entry.stride} → ${entry.dimension}×${entry.dimension}`
                        )
                        .join(" · ")}
                      .
                    </p>
                    <p>
                      At stride {stride} the window never reads {skippedPixels} of
                      the {size * size} input pixels
                      {markSkipped ? " — they are faded in the input grid." : "."}
                    </p>
                  </>
                ) : null}
                {pooling ? (
                  <>
                    <p>
                      {poolMode === "max" ? "Max" : "Average"} pooling with a{" "}
                      {kernelSize}×{kernelSize} window, stride {stride} and p ={" "}
                      {result.pad} turns {size}×{size} into{" "}
                      {result.valid ? `${result.rows}×${result.cols}` : "n/a"} per
                      channel.
                    </p>
                    <p>
                      No weights: the same reduction runs at every position, so
                      pooling adds no parameters to learn and keeps channels
                      separate.
                    </p>
                    <p>
                      {poolMode === "max"
                        ? "Max keeps the strongest activation per window (edges survive); average smooths every patch instead."
                        : "Average smooths every patch; max keeps the strongest activation per window instead."}
                    </p>
                    <p>
                      Cost: {result.valid ? `${result.rows}×${result.cols}` : "n/a"}{" "}
                      × {kernelSize}×{kernelSize} = {poolCost}{" "}
                      {poolMode === "max" ? "comparisons" : "additions"} per
                      channel — no multiplies.
                    </p>
                  </>
                ) : (
                  <>
                    <p>
                      ΣK = {formatValue(totalKernelSum)}
                      {Math.abs(totalKernelSum) < 1e-6
                        ? " → flat regions go dark: this filter reacts to change, not brightness."
                        : " → flat regions keep a constant response, scaled by this sum."}
                    </p>
                    <p>
                      {channels === 1
                        ? `One output value = a dot product over ${kernelSize * kernelSize} numbers.`
                        : `RGB: one output = a dot product over 3·${kernelSize}·${kernelSize} = ${
                            3 * kernelSize * kernelSize
                          } numbers.`}
                    </p>
                    <p>
                      Cost: {result.valid ? `${result.rows}×${result.cols}` : "n/a"}{" "}
                      × {channels} × {kernelSize}×{kernelSize} = {cost} multiplies
                      for one output channel.
                    </p>
                    <p>
                      Cross-correlation is the default;{" "}
                      {flip
                        ? "flip is on, so this is true convolution and symmetric kernels look identical."
                        : "toggle flip to rotate the kernel 180° and compare — symmetric kernels do not change."}
                    </p>
                  </>
                )}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      </div>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            Controls
            <InfoTip>Every control recomputes the output instantly.</InfoTip>
          </CardTitle>
          <CardAction>
            <Button variant="ghost" size="sm" onClick={resetAll}>
              <RotateCcwIcon data-icon="inline-start" />
              Reset all
            </Button>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-5">
          <ControlGroup label="Operation">
            <ChoiceField
              label="Layer type"
              value={operation}
              options={OPERATION_OPTIONS}
              onChange={changeOperation}
              info="Convolution learns kernel weights; max/average pooling has none and only downsamples. Stride and padding apply to all three."
            />
          </ControlGroup>
          <Separator />
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
                onValueChange={(value) => {
                  setScene(value)
                  loadScene(value)
                }}
                options={SCENE_OPTIONS}
                info="Painting over a scene is fine; pick a new scene to reset it."
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
              info={
                pooling
                  ? "Pooling runs on each channel separately, so the output stays 3 maps deep."
                  : "RGB sums three channel dot products, then adds the bias once."
              }
            />
            <SliderField
              label="Input size"
              value={size}
              onValueChange={changeSize}
              min={4}
              max={sizeMax}
              step={sizeStep}
              format={(value) => `${value}×${value}`}
              info={
                source === "photo"
                  ? `The photo is resampled up to its native ${photoMaxSize}×${photoMaxSize}; beyond ${CANVAS_CELL_LIMIT}×${CANVAS_CELL_LIMIT} it is drawn as a canvas image.`
                  : "More cells means finer detail and more computation."
              }
            />
            {canvasMode ? null : channelMode === "rgb" ? (
              <div className="grid gap-2">
                <span className="flex items-center gap-1.5 text-sm font-medium">
                  Brush colour
                  <InfoTip>
                    Colour painted into the input grid while you drag.
                  </InfoTip>
                </span>
                <input
                  type="color"
                  value={brush}
                  onChange={(event) => setBrush(event.target.value)}
                  className="h-8 w-full cursor-pointer rounded-md border border-input bg-background"
                />
              </div>
            ) : (
              <SliderField
                label="Ink level"
                value={level}
                onValueChange={setLevel}
                min={0}
                max={1}
                step={0.05}
                format={(value) => value.toFixed(2)}
                info="Value painted into the input grid while you drag."
              />
            )}
          </ControlGroup>
          <Separator />
          <ControlGroup label={pooling ? "Window" : "Kernel"}>
            <SelectField
              label="Preset"
              value={kernelPreset}
              onValueChange={changeKernelPreset}
              options={KERNEL_PRESETS}
              disabled={pooling}
              info={
                pooling
                  ? "Presets are convolution kernels — pooling has no weights to load."
                  : "Presets are just weights — edit the cells afterwards."
              }
            />
            <ChoiceField
              label={pooling ? "Window size" : "Kernel size"}
              value={String(kernelSize)}
              options={KERNEL_SIZE_OPTIONS}
              onChange={(value) => changeKernelSize(Number(value))}
            />
            <SliderField
              label="Stride"
              value={stride}
              onValueChange={(value) => {
                setStride(value)
                resetAnimation()
              }}
              min={1}
              max={4}
              format={(value) => `${value}`}
              info={
                pooling
                  ? "Each step skips s pixels; classic pooling uses stride = window size, so windows do not overlap."
                  : "Each step skips s pixels of the input."
              }
            />
            <ChoiceField
              label="Padding"
              value={padding}
              options={PADDING_OPTIONS}
              onChange={(value) => {
                setPadding(value)
                resetAnimation()
              }}
              info="Valid adds no border; Same pads (k−1)/2 so the output keeps the input size; Zero adds an explicit ring of zeros (width below)."
            />
            {padding === "zero" ? (
              <SliderField
                label="Zero pad"
                value={padWidth}
                onValueChange={(value) => {
                  setPadWidth(value)
                  resetAnimation()
                }}
                min={1}
                max={3}
                format={(value) => `${value}`}
                info="Thickness of the zero ring added on every side. Visible as dashed cells on the input."
              />
            ) : null}
            <SliderField
              label="Bias"
              value={bias}
              onValueChange={setBias}
              min={-5}
              max={5}
              step={0.5}
              format={(value) => formatValue(value)}
              disabled={pooling}
              info={
                pooling
                  ? "Pooling has no bias — it only reduces the values already present."
                  : "Added once to every output cell, after all weighted sums."
              }
            />
            <SwitchField
              label="Flip kernel"
              checked={flip}
              onCheckedChange={setFlip}
              disabled={pooling}
              info={
                pooling
                  ? "There is no kernel to rotate in pooling."
                  : "Rotate the kernel 180° before sliding: cross-correlation becomes true convolution."
              }
            />
          </ControlGroup>
          <Separator />
          <ControlGroup label="Display">
            <SwitchField
              label="Heatmap output"
              checked={showHeatmap}
              onCheckedChange={setShowHeatmap}
              info="Colour output cells by value: red positive, blue negative."
            />
            <SwitchField
              label="Show numbers"
              checked={showNumbers}
              onCheckedChange={setShowNumbers}
              disabled={canvasMode}
              info={
                canvasMode
                  ? `Only drawn up to ${CANVAS_CELL_LIMIT}×${CANVAS_CELL_LIMIT} — above that a cell is a single pixel.`
                  : "Print the value inside each grid cell, including the zero-pad ring."
              }
            />
            <SwitchField
              label="Colour weights"
              checked={colorWeights}
              onCheckedChange={setColorWeights}
              disabled={pooling}
              info={
                pooling
                  ? "Pooling has no weights to colour."
                  : "Tint kernel cells green for positive and red for negative weights."
              }
            />
            {stride > 1 ? (
              <SwitchField
                label="Stride markers"
                checked={showStride}
                onCheckedChange={setShowStride}
                info="Fade input pixels that no window position ever reads at this stride."
              />
            ) : null}
          </ControlGroup>
          <Separator />
          <ControlGroup label="Playback">
            <SliderField
              label="Sweep speed"
              value={speed}
              onValueChange={setSpeed}
              min={80}
              max={900}
              step={20}
              format={(value) => `${value} ms`}
              info="Delay per reveal step while the sweep is running."
            />
          </ControlGroup>
        </CardContent>
      </Card>

    </div>
  )
}
