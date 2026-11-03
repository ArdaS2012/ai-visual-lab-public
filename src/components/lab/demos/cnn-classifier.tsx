"use client"

import * as React from "react"
import { animate, stagger } from "animejs"
import { toast } from "sonner"
import {
  ChevronDownIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  DicesIcon,
  EraserIcon,
  LightbulbIcon,
  PlayIcon,
  RotateCcwIcon,
  SparklesIcon,
  StepForwardIcon,
  XIcon,
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
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTitle,
} from "@/components/ui/dialog"
import { Switch } from "@/components/ui/switch"
import { InfoTip } from "@/components/lab/controls"
import {
  DenseContributionChips,
  ExplanationCard,
  ExplanationControls,
  ExplanationCta,
  GuideBeams,
  PoolInset,
  ReluInset,
  type BeamLine,
} from "@/components/lab/demos/cnn-explanation"
import { HeatCanvas } from "@/components/lab/heat-canvas"
import {
  useIsomorphicLayoutEffect,
  usePrefersReducedMotion,
} from "@/hooks/use-animation"
import { useExplanation } from "@/hooks/use-explanation"
import { EXPLANATION_STEPS, buildGuideData } from "@/lib/explanation"
import {
  classify,
  classTemplate,
  gridFromCanvas,
  MNIST_SIZE,
  MODEL_SPECS,
  parseModel,
  parseSamples,
  pickStarterSamples,
  sampleGrid,
  traceCell,
  TRACE_STAGE_COUNT,
  verifySample,
  type ClassifierRun,
  type ClassifierStage,
  type FloatMap,
  type MnistModel,
  type RfTrace,
  type SampleDigit,
} from "@/lib/mnist-cnn"
import { receptiveFieldSizes, type RfRegion } from "@/lib/cnn"
import { RF_ACCENT, RF_ACCENT_EDGE, RF_ACCENT_SOFT, formatValue } from "@/lib/heatmap"
import { cn } from "@/lib/utils"

/** Drawing pad resolution; the model grid is 28×28. */
const PAD_SIZE = 280
const BRUSH_WIDTH = 26

/**
 * Screen-pixel breathing room above and beside the scaled 3D board. The hover
 * tilt moves content — the labels at translateZ(96px) smear sideways by up to
 * ~12px and the near-side planes drift — and the viewport clips, so without
 * this margin the outermost text would be sliced. Nothing ever leaves the
 * bottom of the scene, so there is no bottom pad.
 */
const BOARD_PAD = 30

/**
 * Internal coordinate system of the 3D board, scaled to fit its wrapper. The
 * content sits low in the box (the tilt only ever moves it up), which also
 * keeps the board vertically balanced in its card.
 */
const BOARD = {
  width: 920,
  height: 330,
  centerY: 216,
  labelTop: 50,
  layers: [
    { id: "input", x: 24, plane: 76, gap: 0 },
    { id: "conv1", x: 140, plane: 76, gap: 7 },
    { id: "relu1", x: 254, plane: 76, gap: 7 },
    { id: "pool1", x: 368, plane: 56, gap: 7 },
    { id: "conv2", x: 456, plane: 56, gap: 5 },
    { id: "relu2", x: 544, plane: 56, gap: 5 },
    { id: "pool2", x: 632, plane: 40, gap: 5 },
    { id: "flatten", x: 706, plane: 54, gap: 0 },
    { id: "dense", x: 800, plane: 96, gap: 0 },
  ],
}

/** Classes in the dense head and cells per side of the flattened pooled volume. */
const CLASS_COUNT = 10
const FLATTEN_CELLS = 16

/**
 * Board-space geometry of the dense head, mirroring its Tailwind layout
 * (`p-1`, `grid-cols-10`, `gap-[2px]`, a 0.625rem digit row): the connection
 * fan lands exactly on the top of each class column's probability track.
 */
const DENSE_GEOMETRY = (() => {
  const layout = BOARD.layers[BOARD.layers.length - 1]
  const pad = 4
  const gap = 2
  const column = (layout.plane - pad * 2 - gap * (CLASS_COUNT - 1)) / CLASS_COUNT
  return {
    /** y of every column's track top. */
    top: BOARD.centerY - layout.plane / 2 + pad,
    /** x of column `unit`'s centre. */
    x: (unit: number) => layout.x + pad + unit * (column + gap) + column / 2,
  }
})()

const FLATTEN_LAYOUT = BOARD.layers[7]

/** Centre of feature `index` (channels first) on the flattened 16×16 plane. */
function flattenPoint(index: number) {
  const cell = FLATTEN_LAYOUT.plane / FLATTEN_CELLS
  return {
    x: FLATTEN_LAYOUT.x + ((index % FLATTEN_CELLS) + 0.5) * cell,
    y:
      BOARD.centerY -
      FLATTEN_LAYOUT.plane / 2 +
      (Math.floor(index / FLATTEN_CELLS) + 0.5) * cell,
  }
}

type Cell = { row: number; col: number }

/** One thumbnail for a whole stack: per cell the value with the largest magnitude. */
function compositeMap(maps: FloatMap[]): FloatMap {
  const first = maps[0]
  if (!first) return { data: new Float32Array(0), rows: 0, cols: 0 }
  if (maps.length === 1) return first
  const data = new Float32Array(first.data.length)
  data.set(first.data)
  for (let index = 1; index < maps.length; index += 1) {
    const source = maps[index].data
    for (let cell = 0; cell < data.length; cell += 1) {
      if (Math.abs(source[cell]) > Math.abs(data[cell])) data[cell] = source[cell]
    }
  }
  return { data, rows: first.rows, cols: first.cols }
}

/** One-sentence explanation per stage kind, shown in the inspector's ⓘ. */
function stageInfo(stage: ClassifierStage): string {
  if (stage.kind === "conv") {
    return "Each filter slides over all input channels: one dot product per position, plus its bias — one output map per filter. Weights are shared across positions."
  }
  if (stage.kind === "relu") {
    return "Pointwise gate max(0, x): negative values are zeroed. Sparsity is the fingerprint of the non-linearity."
  }
  if (stage.kind === "pool") {
    return "Each window collapses to its largest value, shrinking the map and making small shifts irrelevant."
  }
  if (stage.kind === "flatten") {
    return "The 16×4×4 pooled volume flattened to 256×1, channels first — every value feeds all 10 class scores."
  }
  if (stage.kind === "dense") {
    return "Fully connected: every one of the 256 features wires to each of the 10 classes (2,560 weights). A class score is the features dotted with that class's template, plus a bias; softmax turns scores into probabilities."
  }
  return "The 28×28 grid the model sees, values 0…1."
}

/** Dimensions line for a layer: the flattened vector is 256×1, the head has class scores. */
function stageDims(stage: ClassifierStage): string {
  if (stage.kind === "flatten") return "256×1"
  if (stage.kind === "dense") return `${stage.channels} class scores`
  return `${stage.rows}×${stage.cols}×${stage.channels}`
}

/**
 * Bounding box of every stage's contribution to one cell — the union of the
 * nested windows, so the sizes match `receptiveFieldSizes` and the input box
 * really is the patch the cell sees.
 */
function footprintTrail(
  stages: ClassifierStage[],
  stageIndex: number,
  channel: number,
  row: number,
  col: number
): RfTrace {
  const regions: RfTrace = Array.from({ length: TRACE_STAGE_COUNT }, () => null)
  if (stageIndex < 0 || stageIndex >= TRACE_STAGE_COUNT) return regions

  let top = row
  let bottom = row
  let left = col
  let right = col
  let activeChannel: RfRegion["channel"] = channel
  regions[stageIndex] = { channel: activeChannel, row: top, col: left, rows: 1, cols: 1 }

  for (let index = stageIndex; index >= 1; index -= 1) {
    const spec = stages[index]?.spec
    if (!spec) break
    if (spec.kind === "conv" || spec.kind === "pool") {
      const window = spec.kind === "conv" ? spec.kernelSize : spec.window
      top = top * spec.stride - spec.pad
      bottom = bottom * spec.stride - spec.pad + window - 1
      left = left * spec.stride - spec.pad
      right = right * spec.stride - spec.pad + window - 1
      if (spec.kind === "conv") activeChannel = "all"
    }
    regions[index - 1] = {
      channel: activeChannel,
      row: top,
      col: left,
      rows: bottom - top + 1,
      cols: right - left + 1,
    }
  }
  return regions
}

/** The window an earlier layer contributed, drawn on a map. */
function RegionOverlay({
  region,
  rows,
  cols,
  variant = "cell",
}: {
  region: { row: number; col: number; rows: number; cols: number }
  rows: number
  cols: number
  /** `carried` marks a deeper cell's footprint that survived a layer jump. */
  variant?: "cell" | "carried"
}) {
  return (
    <span
      aria-hidden
      data-rf="true"
      className={cn(
        "pointer-events-none absolute rounded-[2px] duration-200 animate-in fade-in zoom-in-95",
        variant === "carried" && "border-2 border-dashed"
      )}
      style={{
        left: `${(region.col / cols) * 100}%`,
        top: `${(region.row / rows) * 100}%`,
        width: `${(region.cols / cols) * 100}%`,
        height: `${(region.rows / rows) * 100}%`,
        backgroundColor: RF_ACCENT_SOFT,
        borderColor: variant === "carried" ? RF_ACCENT : undefined,
        boxShadow:
          variant === "carried"
            ? `0 0 0 1px ${RF_ACCENT_EDGE}`
            : `inset 0 0 0 2px ${RF_ACCENT}, 0 0 0 1px ${RF_ACCENT_EDGE}`,
      }}
    />
  )
}

/**
 * The footprint a deeper cell left behind when its inspector was replaced by
 * an earlier layer's: a dashed, clipped marking on the earlier map.
 */
function CarriedRegion({
  region,
  rows,
  cols,
}: {
  region: RfRegion
  rows: number
  cols: number
}) {
  return (
    <span aria-hidden className="pointer-events-none absolute inset-0 overflow-hidden">
      <RegionOverlay region={region} rows={rows} cols={cols} variant="carried" />
    </span>
  )
}

/**
 * A receptive-field shell carried from a deeper layer into an earlier one: the
 * deeper cell's footprint on every stage (as the trail steps show it) plus
 * where it came from, so the earlier map can mark the region that fed it.
 */
type RfCarry = {
  /** The earlier layer the marking lands on. */
  target: number
  /** Footprint per stage, in trail-step coordinates. */
  trace: RfTrace
  /** The deeper layer (and cell, when it had one) the trace came from. */
  origin: { layer: number; row: number | null; col: number | null }
}

/** Hue follows the template heatmap: positive red (8), negative blue (220). */
const CONNECTION_POSITIVE = "hsl(8 72% 46%)"
const CONNECTION_NEGATIVE = "hsl(220 72% 46%)"

type FanLine = { x1: number; y1: number; x2: number; y2: number; positive: boolean; opacity: number }

/**
 * One class's 256 incoming connections: a line per pooled feature from its cell
 * on the flattened 16×16 plane to the class column, alpha scaled by the weight
 * magnitude and colour by its sign. These are the same weights the inspector
 * unfolds as the class template, so fan and template explain each other.
 */
function classConnections(weights: Float32Array, unit: number): FanLine[] {
  const target = { x: DENSE_GEOMETRY.x(unit), y: DENSE_GEOMETRY.top }
  let max = 1e-6
  for (let index = 0; index < FLATTEN_CELLS * FLATTEN_CELLS; index += 1) {
    max = Math.max(max, Math.abs(weights[index * CLASS_COUNT + unit]))
  }
  const lines: FanLine[] = []
  for (let index = 0; index < FLATTEN_CELLS * FLATTEN_CELLS; index += 1) {
    const value = weights[index * CLASS_COUNT + unit]
    const magnitude = Math.abs(value) / max
    const start = flattenPoint(index)
    lines.push({
      x1: start.x,
      y1: start.y,
      x2: target.x,
      y2: target.y,
      positive: value >= 0,
      opacity: 0.03 + 0.62 * magnitude ** 1.35,
    })
  }
  return lines
}

/**
 * The fully connected head drawn as what it is: every pooled feature wires to
 * every class. All ten fans stay mounted as a faint fabric; the active class
 * (hovered, selected or predicted) is lit. Pure SVG, memoised by weights —
 * drawing re-renders it only when the highlighted class changes.
 */
const DenseFan = React.memo(function DenseFan({
  weights,
  activeUnit,
  visible,
  emphasis,
}: {
  weights: Float32Array
  activeUnit: number
  visible: boolean
  /** Flattened feature indices whose wires are the class's top contributions. */
  emphasis?: readonly number[]
}) {
  const fans = React.useMemo(
    () => Array.from({ length: CLASS_COUNT }, (_, unit) => classConnections(weights, unit)),
    [weights]
  )
  const emphasized = React.useMemo(() => new Set(emphasis ?? []), [emphasis])
  return (
    <svg
      aria-hidden
      data-dense-fan=""
      viewBox={`0 0 ${BOARD.width} ${BOARD.height}`}
      className={cn(
        "pointer-events-none absolute inset-0 transition-opacity duration-300 motion-reduce:transition-none",
        visible ? "opacity-100" : "opacity-0"
      )}
    >
      {/* Wires only count once they leave the flattened plane, so its heatmap stays readable. */}
      <defs>
        {[
          ["dense-fan-positive", CONNECTION_POSITIVE],
          ["dense-fan-negative", CONNECTION_NEGATIVE],
        ].map(([id, color]) => (
          <linearGradient
            key={id}
            id={id}
            gradientUnits="userSpaceOnUse"
            x1={FLATTEN_LAYOUT.x + FLATTEN_LAYOUT.plane - 8}
            y1={0}
            x2={FLATTEN_LAYOUT.x + FLATTEN_LAYOUT.plane + 8}
            y2={0}
          >
            <stop offset="0" stopColor={color} stopOpacity="0" />
            <stop offset="1" stopColor={color} stopOpacity="1" />
          </linearGradient>
        ))}
      </defs>
      {fans.map((fan, unit) => (
        <g
          key={unit}
          data-fan={unit}
          className="transition-opacity duration-200 motion-reduce:transition-none"
          opacity={unit === activeUnit ? 1 : 0.15}
        >
          {fan.map((line, index) => (
            <line
              key={index}
              x1={line.x1}
              y1={line.y1}
              x2={line.x2}
              y2={line.y2}
              stroke={
                line.positive ? "url(#dense-fan-positive)" : "url(#dense-fan-negative)"
              }
              strokeOpacity={
                emphasized.has(index) ? Math.min(1, line.opacity + 0.5) : line.opacity
              }
              strokeWidth={emphasized.has(index) ? 2.25 : 1}
              strokeLinecap="round"
            />
          ))}
        </g>
      ))}
    </svg>
  )
})

function useElementWidth<T extends HTMLElement>() {
  const [width, setWidth] = React.useState(0)
  const observerRef = React.useRef<ResizeObserver | null>(null)

  const ref = React.useCallback((node: T | null) => {
    observerRef.current?.disconnect()
    observerRef.current = null
    if (!node) return
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0]
      if (entry) setWidth(entry.contentRect.width)
    })
    observer.observe(node)
    observerRef.current = observer
  }, [])

  return [ref, width] as const
}

/** Drawing surface: white ink on black, committed as a 28×28 grid. */
function DigitPad({
  clearToken,
  centered,
  onCommit,
}: {
  clearToken: number
  centered: boolean
  onCommit: (grid: Float32Array) => void
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const drawing = React.useRef(false)
  const lastPoint = React.useRef<{ x: number; y: number } | null>(null)
  const lastCommit = React.useRef(0)
  const paint = React.useCallback((x: number, y: number) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d")
    if (!context) return
    const previous = lastPoint.current
    context.strokeStyle = "#fff"
    context.lineWidth = BRUSH_WIDTH
    context.lineCap = "round"
    context.lineJoin = "round"
    if (previous) {
      context.beginPath()
      context.moveTo(previous.x, previous.y)
      context.lineTo(x, y)
      context.stroke()
    } else {
      context.beginPath()
      context.arc(x, y, BRUSH_WIDTH / 2, 0, Math.PI * 2)
      context.fillStyle = "#fff"
      context.fill()
    }
    lastPoint.current = { x, y }
  }, [])

  const pointFrom = (event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget
    const rect = canvas.getBoundingClientRect()
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    }
  }

  const commit = React.useCallback(() => {
    const canvas = canvasRef.current
    if (canvas) onCommit(gridFromCanvas(canvas, MNIST_SIZE, centered))
  }, [centered, onCommit])

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d")
    if (!context) return
    context.fillStyle = "#000"
    context.fillRect(0, 0, canvas.width, canvas.height)
    lastPoint.current = null
  }, [clearToken])

  return (
    <canvas
      ref={canvasRef}
      width={PAD_SIZE}
      height={PAD_SIZE}
      aria-label="Drawing pad, draw a digit"
      className="aspect-square w-full cursor-crosshair touch-none rounded-lg bg-black ring-1 ring-foreground/15"
      onPointerDown={(event) => {
        event.currentTarget.setPointerCapture(event.pointerId)
        drawing.current = true
        lastPoint.current = null
        const point = pointFrom(event)
        paint(point.x, point.y)
      }}
      onPointerMove={(event) => {
        if (!drawing.current) return
        const point = pointFrom(event)
        paint(point.x, point.y)
        const now = performance.now()
        if (now - lastCommit.current > 90) {
          lastCommit.current = now
          commit()
        }
      }}
      onPointerUp={(event) => {
        drawing.current = false
        event.currentTarget.releasePointerCapture(event.pointerId)
        commit()
      }}
      onPointerCancel={() => {
        drawing.current = false
      }}
    />
  )
}

function LayerChip({
  index,
  stage,
  selected,
  onSelect,
}: {
  index: number
  stage: ClassifierStage
  selected: boolean
  onSelect: (index: number) => void
}) {
  const short = stage.name.replace("MaxPool", "Pool")
  return (
    <button
      type="button"
      data-layer-chip={index}
      onClick={() => onSelect(index)}
      className={cn(
        "flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium ring-1 transition-colors",
        selected
          ? "bg-primary text-primary-foreground ring-primary"
          : "bg-muted/40 text-muted-foreground ring-foreground/10 hover:bg-muted hover:text-foreground"
      )}
    >
      {short}
      <span
        className={cn(
          "font-mono text-[9px] tabular-nums",
          selected ? "text-primary-foreground/70" : "text-muted-foreground"
        )}
      >
        {stageDims(stage)}
      </span>
    </button>
  )
}

function ClassBars({
  run,
  selectedClass,
  onSelectClass,
  reducedMotion,
}: {
  run: ClassifierRun
  selectedClass: number | null
  onSelectClass: (unit: number) => void
  reducedMotion: boolean
}) {
  const containerRef = React.useRef<HTMLDivElement>(null)

  useIsomorphicLayoutEffect(() => {
    const nodes = containerRef.current?.querySelectorAll<HTMLElement>("[data-bar]")
    if (!nodes || nodes.length === 0) return
    const values = Array.from(nodes, (_, index) => run.probabilities[index] ?? 0)
    if (reducedMotion) {
      values.forEach((value, index) => {
        nodes[index].style.height = `${value * 100}%`
      })
      return
    }
    const animation = animate(nodes, {
      height: (_, index) => `${(values[index ?? 0] ?? 0) * 100}%`,
      duration: 460,
      ease: "out(3)",
      delay: stagger(18),
    })
    return () => {
      animation.revert()
    }
  }, [run.probabilities, reducedMotion])

  return (
    <div ref={containerRef} className="flex items-end gap-1.5">
      {Array.from(run.probabilities, (probability, unit) => {
        const isPredicted = unit === run.predicted
        const isSelected = unit === selectedClass
        return (
          <button
            key={unit}
            type="button"
            onClick={() => onSelectClass(unit)}
            className={cn(
              "group grid min-w-0 flex-1 content-end gap-1 rounded-md p-1 ring-1 transition-colors hover:bg-muted/50",
              isSelected
                ? "bg-primary/10 ring-primary"
                : "ring-foreground/10"
            )}
            aria-label={`Class ${unit}, ${Math.round(probability * 100)} percent`}
          >
            <div className="relative h-16 w-full overflow-hidden rounded bg-muted/40">
              <span
                data-bar={unit}
                style={{ height: "0%" }}
                className={cn(
                  "absolute inset-x-0 bottom-0 rounded-b",
                  isPredicted ? "bg-primary" : "bg-primary/35"
                )}
              />
            </div>
            <span
              className={cn(
                "text-center font-mono text-[11px] tabular-nums",
                isPredicted && "font-bold"
              )}
            >
              {unit}
            </span>
            <span className="text-center font-mono text-[9px] text-muted-foreground tabular-nums">
              {Math.round(probability * 100)}%
            </span>
          </button>
        )
      })}
    </div>
  )
}

export function CnnClassifier() {
  const [model, setModel] = React.useState<MnistModel | null>(null)
  const [samples, setSamples] = React.useState<SampleDigit[]>([])
  const [loadError, setLoadError] = React.useState<string | null>(null)
  const [selfCheck, setSelfCheck] = React.useState<number | null>(null)

  const [input, setInput] = React.useState<Float32Array>(
    () => new Float32Array(MNIST_SIZE * MNIST_SIZE)
  )
  const [origin, setOrigin] = React.useState<{ kind: "draw" | "sample"; sample?: SampleDigit }>(
    { kind: "draw" }
  )
  const [clearToken, setClearToken] = React.useState(0)
  const [centered, setCentered] = React.useState(true)
  const [showNumbers, setShowNumbers] = React.useState(true)

  const [selected, setSelected] = React.useState<{ layer: number; channel: number } | null>(
    null
  )
  const [selectedClass, setSelectedClass] = React.useState<number | null>(null)
  const [previewClass, setPreviewClass] = React.useState<number | null>(null)
  const [probe, setProbe] = React.useState<Cell | null>(null)
  const [pinned, setPinned] = React.useState<Cell | null>(null)
  /** Footprint carried into an earlier layer by clicking an RF trail step. */
  const [carry, setCarry] = React.useState<RfCarry | null>(null)
  const [revealed, setRevealed] = React.useState(Number.POSITIVE_INFINITY)
  const [playing, setPlaying] = React.useState(false)
  const [inspecting, setInspecting] = React.useState(false)
  /** Guided "explain this prediction" story — an orchestration layer only. */
  const explanation = useExplanation(EXPLANATION_STEPS)

  const reducedMotion = usePrefersReducedMotion()
  const [viewportRef, viewportWidth] = useElementWidth<HTMLDivElement>()
  const boardRef = React.useRef<HTMLDivElement>(null)
  const sceneRef = React.useRef<HTMLDivElement>(null)
  const tiltRef = React.useRef<HTMLDivElement>(null)
  const firstPaint = React.useRef(true)

  React.useEffect(() => {
    let cancelled = false
    Promise.all([
      fetch("/models/mnist-cnn.json").then((response) => response.json()),
      fetch("/models/mnist-samples.json").then((response) => response.json()),
    ])
      .then(([rawModel, rawSamples]) => {
        if (cancelled) return
        const parsed = parseModel(rawModel)
        const parsedSamples = parseSamples(rawSamples)
        setModel(parsed)
        setSamples(parsedSamples)
        if (parsedSamples[0]) {
          const check = verifySample(parsed, parsedSamples[0])
          setSelfCheck(check.maxLogitDiff)
        }
      })
      .catch(() => {
        if (!cancelled) setLoadError("The pretrained weights could not be loaded.")
      })
    return () => {
      cancelled = true
    }
  }, [])

  const run = React.useMemo(
    () => (model ? classify(model, input) : null),
    [model, input]
  )

  const guideActive = explanation.state.active
  const guideStep = explanation.step
  const guideLayer = guideActive && guideStep ? guideStep.layer : -1
  /** Real activations for the current beat — never a mock visualization. */
  const guide = React.useMemo(
    () => (run && model && guideStep ? buildGuideData(model, run, guideStep) : null),
    [run, model, guideStep]
  )

  /**
   * Every beat moves the camera by selecting the layer it explains; the
   * existing camera effect then glides there like any other layer focus.
   */
  useIsomorphicLayoutEffect(() => {
    if (!guideActive || !guideStep || !run) return
    setSelected({ layer: guideStep.layer, channel: guide?.channel ?? 0 })
    setSelectedClass(null)
    setPreviewClass(null)
    setProbe(null)
    setPinned(null)
    setCarry(null)
    setInspecting(false)
  }, [guideActive, guideStep, guide?.channel, run])

  const starterSamples = React.useMemo(() => pickStarterSamples(samples), [samples])
  const starterMaps = React.useMemo(
    () =>
      starterSamples.map((sample) => ({
        sample,
        map: { data: sampleGrid(sample), rows: MNIST_SIZE, cols: MNIST_SIZE },
      })),
    [starterSamples]
  )
  const rfSizes = React.useMemo(() => receptiveFieldSizes(MODEL_SPECS), [])
  const scale =
    viewportWidth > 0
      ? Math.max(0.2, Math.min(1, (viewportWidth - BOARD_PAD * 2) / BOARD.width))
      : 1
  /** Centre the board in the card; narrower than the board it lands on BOARD_PAD. */
  const sceneX = viewportWidth > 0 ? (viewportWidth - BOARD.width * scale) / 2 : BOARD_PAD

  const stageIndex = selected?.layer ?? -1
  const stage = run && stageIndex >= 0 ? run.stages[stageIndex] : undefined
  const channel = stage ? Math.min(selected?.channel ?? 0, stage.channels - 1) : 0
  const selectedMap = stage && stage.kind !== "dense" ? stage.maps[channel] : undefined

  const hovered = probe
  const effectiveProbe = pinned ?? hovered

  const activeProbe = React.useMemo(() => {
    if (!effectiveProbe || !stage || !selectedMap) return null
    if (effectiveProbe.row < 0 || effectiveProbe.row >= selectedMap.rows) return null
    if (effectiveProbe.col < 0 || effectiveProbe.col >= selectedMap.cols) return null
    return effectiveProbe
  }, [effectiveProbe, stage, selectedMap])

  /**
   * A carried footprint replaces the live one while the user is on the layer
   * it was carried into: the map keeps the deeper cell's marking until a cell
   * of this layer is picked.
   */
  const carriedTrace = carry && carry.target === stageIndex ? carry.trace : null

  const trace = React.useMemo<RfTrace | null>(() => {
    if (carriedTrace) return carriedTrace
    if (!run || !activeProbe || !stage || stageIndex < 0) return null
    if (stage.kind === "flatten") {
      const index = activeProbe.row * stage.cols + activeProbe.col
      const poolChannel = Math.floor(index / 16)
      const cell = index % 16
      return traceCell(
        run.stages,
        6,
        poolChannel,
        Math.floor(cell / 4),
        cell % 4
      )
    }
    if (stageIndex <= 6 && selectedMap) {
      return traceCell(run.stages, stageIndex, channel, activeProbe.row, activeProbe.col)
    }
    return null
  }, [run, stage, stageIndex, channel, activeProbe, selectedMap, carriedTrace])

  /** Same cell, but every stage as the bounding box it truly depends on. */
  const footprint = React.useMemo<RfTrace | null>(() => {
    if (carriedTrace) return carriedTrace
    if (!run || !activeProbe || !stage || stageIndex < 0) return null
    if (stage.kind === "flatten") {
      const index = activeProbe.row * stage.cols + activeProbe.col
      const poolChannel = Math.floor(index / 16)
      const cell = index % 16
      return footprintTrail(
        run.stages,
        6,
        poolChannel,
        Math.floor(cell / 4),
        cell % 4
      )
    }
    if (stageIndex <= 6 && selectedMap) {
      return footprintTrail(run.stages, stageIndex, channel, activeProbe.row, activeProbe.col)
    }
    return null
  }, [run, stage, stageIndex, channel, activeProbe, selectedMap, carriedTrace])

  /**
   * Dashed links from the previous stage's strongest map to the maps this beat
   * highlights — a hint of how pooled responses fan into the next convolution.
   */
  const guideBeams = React.useMemo<BeamLine[]>(() => {
    if (!run || !guideStep || !guide) return []
    if (guideStep.id !== "conv1" && guideStep.id !== "conv2") return []
    const layout = BOARD.layers[guideStep.layer]
    const previousLayout = BOARD.layers[guideStep.layer - 1]
    if (!layout || !previousLayout) return []
    const sourceX = previousLayout.x + previousLayout.plane
    return guide.strong.slice(0, 3).map((_, index) => ({
      x1: sourceX,
      y1: BOARD.centerY,
      x2: layout.x,
      y2: BOARD.centerY + (index - 1) * 16,
    }))
  }, [run, guideStep, guide])

  /**
   * Camera: focus the selected layer, otherwise show the whole network. It
   * only pans — the board rests front-on so the hover tilt starts from neutral
   * and swings symmetrically in both directions.
   */
  const camera = React.useMemo(() => {
    if (stageIndex < 0) return { x: 0, rotate: 0 }
    const layout = BOARD.layers[stageIndex]
    return {
      x: BOARD.width / 2 - (layout.x + layout.plane / 2 + 12),
      rotate: 0,
    }
  }, [stageIndex])

  useIsomorphicLayoutEffect(() => {
    const node = boardRef.current
    if (!node) return
    if (reducedMotion) {
      node.style.transform = `translateX(${camera.x}px) rotateY(${camera.rotate}deg)`
      return
    }
    const animation = animate(node, {
      x: camera.x,
      rotateY: camera.rotate,
      duration: 560,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
    // `model` re-runs the effect when the board first mounts: the initial
    // whole-network camera must also apply on load, not only after a reset.
  }, [camera, reducedMotion, model])

  /** Board entrance: layers slide in left to right. */
  useIsomorphicLayoutEffect(() => {
    const node = boardRef.current
    if (!node || reducedMotion || !firstPaint.current || !model) return
    firstPaint.current = false
    const groups = node.querySelectorAll("[data-layer-group]")
    const animation = animate(groups, {
      opacity: [0, 1],
      x: [26, 0],
      duration: 520,
      ease: "out(3)",
      delay: stagger(55),
    })
    return () => {
      animation.revert()
    }
  }, [model, reducedMotion])

  /** Forward pass: one layer pops in per tick. */
  const isRunning =
    playing && run !== null && revealed < run.stages.length

  React.useEffect(() => {
    if (!playing || !run || !isRunning) return
    const id = window.setTimeout(() => {
      const next = revealed + 1
      setRevealed(next)
      const node = boardRef.current?.querySelector<HTMLElement>(
        `[data-layer-group="${next - 1}"]`
      )
      if (node && !reducedMotion) {
        animate(node, {
          opacity: [0.12, 1],
          x: [22, 0],
          duration: 420,
          ease: "out(3)",
        })
      }
      if (next >= run.stages.length) setPlaying(false)
    }, 620)
    return () => window.clearTimeout(id)
  }, [playing, isRunning, revealed, run, reducedMotion])

  const selectLayer = React.useCallback(
    (layer: number | null, nextChannel = 0, nextCarry: RfCarry | null = null) => {
      setSelected(layer === null ? null : { layer, channel: nextChannel })
      setProbe(null)
      setPinned(null)
      setCarry(nextCarry)
      setPreviewClass(null)
      if (layer !== null && run && run.stages[layer]?.kind === "dense") {
        setSelectedClass(null)
      }
    },
    [run]
  )

  /** A click on the board or a chip pops the layer inspector out. */
  const openInspector = React.useCallback(
    (layer: number, nextChannel = 0) => {
      selectLayer(layer, nextChannel)
      setInspecting(true)
    },
    [selectLayer]
  )

  /** A click on a class column pops the dense head out focused on that class. */
  const openClassInspector = React.useCallback(
    (unit: number) => {
      selectLayer(BOARD.layers.length - 1, 0)
      setSelectedClass(unit)
      setInspecting(true)
    },
    [selectLayer]
  )

  const loadSample = React.useCallback(
    (sample: SampleDigit) => {
      // A new input is a new prediction: the guided story would be stale.
      explanation.exit()
      setInput(sampleGrid(sample))
      setOrigin({ kind: "sample", sample })
      setPlaying(false)
      setRevealed(Number.POSITIVE_INFINITY)
      setProbe(null)
      setPinned(null)
      setCarry(null)
    },
    [explanation]
  )

  const commitDrawing = React.useCallback(
    (grid: Float32Array) => {
      explanation.exit()
      setInput(grid)
      setOrigin({ kind: "draw" })
      setProbe(null)
      setPinned(null)
      setCarry(null)
    },
    [explanation]
  )

  const runPass = () => {
    setPlaying(false)
    setRevealed(1)
    setPlaying(true)
    setProbe(null)
    setPinned(null)
    setCarry(null)
  }

  const stepPass = () => {
    if (!run) return
    setPlaying(false)
    setRevealed((current) => {
      const value = Number.isFinite(current) ? current : run.stages.length
      return value >= run.stages.length ? 1 : value + 1
    })
  }

  const resetView = () => {
    setSelected(null)
    setSelectedClass(null)
    setPreviewClass(null)
    setProbe(null)
    setPinned(null)
    setCarry(null)
    setPlaying(false)
    setRevealed(Number.POSITIVE_INFINITY)
    setInspecting(false)
  }

  /** Enter the guided story: full board visible, camera on the first beat. */
  const startExplanation = () => {
    if (!run || !model) return
    setPlaying(false)
    setRevealed(Number.POSITIVE_INFINITY)
    setSelected(null)
    setSelectedClass(null)
    setPreviewClass(null)
    setProbe(null)
    setPinned(null)
    setCarry(null)
    setInspecting(false)
    explanation.start()
  }

  /** Leave the guided story and hand the whole-network view back. */
  const exitExplanation = () => {
    explanation.exit()
    resetView()
  }

  /**
   * Closing the inspector — X, Escape or a click outside — returns the board
   * to the whole-network view. Without this the camera stays parked on the
   * last inspected layer and the rest of the network is cut off at the card.
   */
  const handleInspectOpenChange = (open: boolean) => {
    if (open) setInspecting(true)
    else resetView()
  }

  const togglePin = (cell: Cell) => {
    // Picking a cell on the layer starts a fresh trace: the deeper layer's
    // carried footprint is no longer what this map is about.
    setCarry(null)
    setPinned((current) =>
      current && current.row === cell.row && current.col === cell.col ? null : cell
    )
  }

  const replay = () => {
    explanation.exit()
    setInput(new Float32Array(MNIST_SIZE * MNIST_SIZE))
    setOrigin({ kind: "draw" })
    setClearToken((token) => token + 1)
    setProbe(null)
    setPinned(null)
    setCarry(null)
  }

  /**
   * Hover parallax: the board leans towards the cursor and drifts opposite to
   * it. Written straight to the tilt layer so pointer moves never re-render;
   * the camera transform on the parent stays untouched. Reduced motion keeps
   * the board static.
   */
  const tiltBoard = (event: React.PointerEvent<HTMLDivElement>) => {
    if (reducedMotion || event.pointerType !== "mouse") return
    const node = tiltRef.current
    const scene = sceneRef.current
    if (!node || !scene) return
    const rect = scene.getBoundingClientRect()
    if (rect.width === 0 || rect.height === 0) return
    const px = Math.min(0.5, Math.max(-0.5, (event.clientX - rect.left) / rect.width - 0.5))
    const py = Math.min(0.5, Math.max(-0.5, (event.clientY - rect.top) / rect.height - 0.5))
    node.style.transform = `translate3d(${(-px * 24).toFixed(1)}px, ${(-py * 12).toFixed(
      1
    )}px, 0) rotateX(${(6 + py * 8).toFixed(2)}deg) rotateY(${(-px * 14).toFixed(2)}deg)`
  }

  const restBoard = () => {
    const node = tiltRef.current
    if (node) node.style.transform = ""
  }

  useIsomorphicLayoutEffect(() => {
    if (!reducedMotion) return
    const node = tiltRef.current
    if (node) node.style.transform = ""
  }, [reducedMotion])

  if (loadError) {
    return (
      <Alert>
        <LightbulbIcon />
        <AlertTitle>Model unavailable</AlertTitle>
        <AlertDescription>
          {loadError} Expected <code>public/models/mnist-cnn.json</code> and{" "}
          <code>mnist-samples.json</code>.
        </AlertDescription>
      </Alert>
    )
  }

  const predicted = run?.predicted ?? null
  /** The class whose wiring is lit: hovered column, inspected class, or prediction. */
  const fanUnit = previewClass ?? selectedClass ?? predicted ?? 0
  const sampleTruth = origin.kind === "sample" ? origin.sample?.label : undefined
  const correct = sampleTruth !== undefined && predicted !== null && sampleTruth === predicted

  /**
   * Guided overlay geometry: the card lives in the scaled board but counter-
   * scales so its annotation stays readable on narrow screens. Tall beats
   * (pool/ReLU) anchor near the top and shrink just enough to never clip at
   * the board edge; the rest sit close above the focused layer.
   */
  const guideTallCard =
    guideStep?.id === "pool1" ||
    guideStep?.id === "pool2" ||
    guideStep?.id === "relu1" ||
    guideStep?.id === "relu2"
  const guideCardHeight =
    guideStep?.id === "pool1" || guideStep?.id === "pool2"
      ? 218
      : guideStep?.id === "relu1" || guideStep?.id === "relu2"
        ? 182
        : 132
  const guideCardTop = guideTallCard
    ? 4
    : guideStep?.id === "dense" || guideStep?.id === "prediction"
      ? 14
      : 92
  const guideDenseCard =
    guideStep?.id === "dense" || guideStep?.id === "prediction"
  const guidePlaneTop =
    guideLayer >= 0
      ? BOARD.centerY - BOARD.layers[guideLayer].plane / 2
      : BOARD.centerY
  const guideCardAvailable = guideDenseCard
    ? // Keep the class columns visible below a dense/prediction card.
      (guidePlaneTop - guideCardTop) * scale - 6
    : guideTallCard
      ? BOARD.height * scale - 8 - guideCardTop * scale
      : Number.POSITIVE_INFINITY
  const guideCardFit = Math.min(
    1,
    Math.max(0.7, guideCardAvailable / guideCardHeight)
  )
  const guideCardScale = guideCardFit / Math.max(0.2, scale)

  /** The mono line under a guided beat's title, read from the current run. */
  const guideDetail = (() => {
    if (!guideActive || !guideStep || !run) return undefined
    switch (guideStep.id) {
      case "input":
        return `${MNIST_SIZE}×${MNIST_SIZE} · ${
          origin.kind === "sample" ? "test digit" : "your drawing"
        }`
      case "conv1":
      case "conv2":
        return guide && guide.strong.length > 0
          ? `strongest maps ${guide.strong.map((entry) => entry + 1).join(", ")}`
          : undefined
      case "relu1":
      case "relu2":
        return guide?.relu ? `strongest map ${guide.relu.channel + 1}` : undefined
      case "pool1":
      case "pool2":
        return guide?.pool ? `stride ${guide.pool.stride}` : undefined
      case "dense":
        return guide?.dense
          ? `logit ${guide.dense.unit} = ${formatValue(run.logits[guide.dense.unit])}`
          : undefined
      // Prediction keeps the readout in the card body; no duplicate detail line.
      default:
        return undefined
    }
  })()

  return (
    <div className="grid gap-4">
      <Card className="overflow-hidden">
        <CardHeader>
          <CardTitle className="flex items-center gap-1.5">
            Trained CNN — 3D board
            <InfoTip>
              A real pretrained MNIST network (ONNX Model Zoo, CNTK): two
              convolution blocks and a fully connected classification head (256
              → 10, 5,994 weights). Wires connect the flattened features to the
              classes — hover a class column to light up its 256 connections.
              Each prediction is a full TypeScript forward pass over the loaded
              weights.
            </InfoTip>
          </CardTitle>
          <CardAction>
            <div className="flex flex-wrap items-center justify-end gap-2">
              <Button
                size="sm"
                onClick={runPass}
                disabled={!run || isRunning || guideActive}
              >
                <PlayIcon data-icon="inline-start" />
                {isRunning ? "Running…" : "Run pass"}
              </Button>
              <Button
                size="sm"
                variant={guideActive ? "secondary" : "default"}
                aria-pressed={guideActive}
                onClick={guideActive ? exitExplanation : startExplanation}
                disabled={!run || isRunning}
              >
                <SparklesIcon data-icon="inline-start" />
                {guideActive ? "Explaining…" : "Explain this prediction"}
              </Button>
              <Button
                size="sm"
                variant="outline"
                onClick={stepPass}
                disabled={!run || guideActive}
              >
                <StepForwardIcon data-icon="inline-start" />
                Step
              </Button>
              <Button size="sm" variant="ghost" onClick={resetView} disabled={guideActive}>
                <RotateCcwIcon data-icon="inline-start" />
                Reset view
              </Button>
              {model ? (
                <Badge variant="secondary" className="font-mono tabular-nums">
                  {Math.min(revealed, run?.stages.length ?? 0)} / {run?.stages.length ?? 0}
                </Badge>
              ) : null}
            </div>
          </CardAction>
        </CardHeader>
        <CardContent className="grid gap-3">
          {!run || !model ? (
            <p className="text-sm text-muted-foreground">
              Loading the pretrained weights and test digits…
            </p>
          ) : (
            <>
              <div
                ref={viewportRef}
                data-scene="classifier"
                className="relative overflow-hidden rounded-lg bg-gradient-to-br from-muted/60 to-muted/20 ring-1 ring-foreground/10"
                style={{
                  height: BOARD.height * scale + BOARD_PAD,
                  opacity: viewportWidth > 0 ? 1 : 0,
                }}
                onPointerMove={tiltBoard}
                onPointerLeave={restBoard}
                onPointerCancel={restBoard}
              >
                <div
                  ref={sceneRef}
                  className="absolute origin-top-left [perspective:1200px]"
                  style={{
                    top: BOARD_PAD,
                    left: sceneX,
                    width: BOARD.width,
                    height: BOARD.height,
                    transform: `scale(${scale})`,
                  }}
                >
                  <div
                    ref={boardRef}
                    className="absolute inset-0 will-change-transform [transform-style:preserve-3d]"
                  >
                    <div
                      ref={tiltRef}
                      data-tilt="board"
                      className="absolute inset-0 transition-transform duration-200 ease-out will-change-transform [transform:rotateX(6deg)] [transform-style:preserve-3d]"
                    >
                      {BOARD.layers.map((layout, layerIndex) => {
                        const layerStage = run.stages[layerIndex]
                        if (!layerStage) return null
                        const isSelected = selected?.layer === layerIndex
                        const isRevealed = layerIndex < revealed
                        const isGuideLayer = guideLayer === layerIndex
                        const guidedOut = guideLayer >= 0 && !isGuideLayer
                        /** Conv beats fan the stack out and lift the strongest maps. */
                        const fanStage =
                          isGuideLayer &&
                          (guideStep?.id === "conv1" || guideStep?.id === "conv2")
                        const fanFactor = fanStage ? 1.9 : 1
                        const depth = (layerStage.channels - 1) * layout.gap * fanFactor
                        const planeSize = layout.plane
                        const top = BOARD.centerY - planeSize / 2
                        const strongHere = fanStage ? (guide?.strong ?? []) : []
                        /** ReLU beats crossfade each plane from its pre-activation. */
                        const reluPeek =
                          isGuideLayer &&
                          (guideStep?.id === "relu1" || guideStep?.id === "relu2") &&
                          guide?.relu != null &&
                          !reducedMotion
                        const guidePrediction =
                          isGuideLayer && guideStep?.id === "prediction"

                        return (
                          <div
                            key={layout.id}
                            data-layer-group={layerIndex}
                            data-guide-dim={guidedOut ? "" : undefined}
                            className={cn(
                              "absolute [transform-style:preserve-3d]",
                              guideLayer >= 0 && "pointer-events-none"
                            )}
                            style={{ left: layout.x, top: 0, width: planeSize, height: BOARD.height }}
                          >
                            <button
                              type="button"
                              data-layer-label={layerIndex}
                              onClick={() => openInspector(layerIndex, 0)}
                              className={cn(
                                "absolute left-0 z-20 grid gap-0 rounded-md px-1.5 py-1 text-left ring-1 transition-colors [transform:translateZ(96px)]",
                                isSelected
                                  ? "bg-primary text-primary-foreground ring-primary"
                                  : "bg-card/90 text-foreground ring-foreground/10 hover:ring-foreground/30"
                              )}
                              style={{
                                top: BOARD.labelTop,
                                width: "max-content",
                                minWidth: planeSize,
                                opacity: guidedOut ? 0.15 : 1,
                              }}
                            >
                              <span className="text-[11px] font-medium whitespace-nowrap">
                                {layerStage.name}
                              </span>
                              <span
                                className={cn(
                                  "font-mono text-[9px] tabular-nums",
                                  isSelected
                                    ? "text-primary-foreground/70"
                                    : "text-muted-foreground"
                                )}
                              >
                                {stageDims(layerStage)}
                              </span>
                            </button>

                            {layerStage.kind === "dense" ? (
                              <div
                                data-dense-plane=""
                                className={cn(
                                  "absolute grid grid-cols-10 gap-[2px] rounded-md bg-card/80 p-1 ring-1 transition-opacity",
                                  isSelected ? "ring-primary" : "ring-foreground/10"
                                )}
                                style={{
                                  top,
                                  left: 0,
                                  width: planeSize,
                                  height: planeSize,
                                  opacity: (isRevealed ? 1 : 0.12) * (guidedOut ? 0.15 : 1),
                                }}
                              >
                                {Array.from(run.probabilities, (probability, unit) => (
                                  <button
                                    key={unit}
                                    type="button"
                                    data-class-column={unit}
                                    aria-label={`Class ${unit}, ${Math.round(probability * 100)} percent${
                                      unit === run.predicted ? ", predicted" : ""
                                    }`}
                                    onClick={() => openClassInspector(unit)}
                                    onPointerEnter={() => setPreviewClass(unit)}
                                    onPointerLeave={() => setPreviewClass(null)}
                                    className="grid h-full grid-rows-[1fr_0.625rem] gap-[2px] rounded-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
                                  >
                                    <span
                                      className={cn(
                                        "relative block overflow-hidden rounded-[3px] ring-1 transition-colors",
                                        unit === fanUnit
                                          ? "bg-primary/5 ring-primary/30"
                                          : "bg-muted/60 ring-foreground/10",
                                        guidePrediction &&
                                          unit === run.predicted &&
                                          "ring-2 ring-primary motion-safe:animate-[prediction-pulse_1.6s_ease-in-out_infinite]"
                                      )}
                                    >
                                      <span
                                        style={{ height: `${Math.max(4, probability * 100)}%` }}
                                        className={cn(
                                          "absolute inset-x-0 bottom-0 rounded-[3px] transition-[height] duration-500 ease-out motion-reduce:transition-none",
                                          unit === run.predicted ? "bg-primary" : "bg-primary/40"
                                        )}
                                      />
                                    </span>
                                    <span
                                      className={cn(
                                        "text-center font-mono text-[8px] leading-[0.625rem] tabular-nums",
                                        unit === fanUnit ? "text-foreground" : "text-muted-foreground"
                                      )}
                                    >
                                      {unit}
                                    </span>
                                  </button>
                                ))}
                              </div>
                            ) : (
                              Array.from({ length: layerStage.channels }, (_, channelIndex) => {
                                const map =
                                  layerStage.kind === "flatten"
                                    ? layerStage.maps[0]
                                    : layerStage.maps[channelIndex]
                                const region = trace?.[layerIndex]
                                const highlight =
                                  region &&
                                  (region.channel === "all" || region.channel === channelIndex)
                                const isActiveChannel =
                                  isSelected && (selected?.channel ?? 0) === channelIndex
                                const isStrong = strongHere.includes(channelIndex)
                                return (
                                  <button
                                    key={channelIndex}
                                    type="button"
                                    data-plane={`${layerIndex}-${channelIndex}`}
                                    data-guide-strong={isStrong ? "" : undefined}
                                    onClick={() => openInspector(layerIndex, channelIndex)}
                                    className="group absolute top-0 left-0 transition-[transform,opacity] duration-200 motion-reduce:transition-none [transform-style:preserve-3d] hover:z-50"
                                    style={{
                                      width: planeSize,
                                      height: planeSize,
                                      top,
                                      transform: `translateZ(${
                                        channelIndex * layout.gap * fanFactor -
                                        depth / 2 +
                                        (isStrong ? 16 : 0)
                                      }px)`,
                                      opacity:
                                        (isRevealed
                                          ? isSelected
                                            ? isActiveChannel
                                              ? 1
                                              : 0.5
                                            : 0.82
                                          : 0.12) * (guidedOut ? 0.15 : 1),
                                    }}
                                  >
                                    <span
                                      className={cn(
                                        "relative block size-full overflow-hidden rounded-[4px] ring-1 transition-transform duration-200 ring-foreground/10",
                                        "group-hover:[transform:translateZ(26px)] group-hover:ring-2 group-hover:ring-primary",
                                        isActiveChannel && "ring-2 ring-primary",
                                        isStrong && "ring-2 ring-primary"
                                      )}
                                    >
                                      <HeatCanvas
                                        map={map}
                                        mode={layerStage.kind === "input" ? "grey" : "heat"}
                                        bound={layerStage.stats.bound}
                                      />
                                      {highlight && region ? (
                                        <RegionOverlay
                                          key={`${region.row}-${region.col}-${region.rows}-${region.cols}`}
                                          region={region}
                                          rows={layerStage.rows}
                                          cols={layerStage.cols}
                                        />
                                      ) : null}
                                      {reluPeek && guide?.relu ? (
                                        <span
                                          aria-hidden
                                          data-relu-peek={channelIndex}
                                          className="pointer-events-none absolute inset-0 animate-[relu-fade_var(--relu-duration)_linear_forwards] motion-reduce:animate-none"
                                          style={
                                            {
                                              "--relu-duration": `${guideStep?.durationMs ?? 2000}ms`,
                                              animationPlayState: explanation.state.playing
                                                ? "running"
                                                : "paused",
                                            } as React.CSSProperties
                                          }
                                        >
                                          <HeatCanvas
                                            map={guide.relu.pre.maps[channelIndex]}
                                            bound={guide.relu.pre.stats.bound}
                                          />
                                        </span>
                                      ) : null}
                                    </span>
                                  </button>
                                )
                              })
                            )}
                          </div>
                        )
                      })}
                      <GuideBeams
                        lines={guideBeams}
                        width={BOARD.width}
                        height={BOARD.height}
                        visible={guideBeams.length > 0}
                      />
                      <DenseFan
                        weights={model.dense.weights}
                        activeUnit={fanUnit}
                        visible={
                          revealed > BOARD.layers.length - 1 &&
                          (guideLayer < 0 || guideLayer === BOARD.layers.length - 1)
                        }
                        emphasis={
                          guide?.dense && guideLayer === BOARD.layers.length - 1
                            ? [
                                ...guide.dense.positive.map((entry) => entry.index),
                                ...guide.dense.negative.map((entry) => entry.index),
                              ]
                            : undefined
                        }
                      />
                    </div>
                    {guideActive && guideStep && guideLayer >= 0 ? (
                      // The status node stays mounted so screen readers announce
                      // each beat in place; the card inside remounts per beat.
                      <div
                        role="status"
                        aria-live="polite"
                        data-guide-anchor=""
                        className="pointer-events-none absolute z-30 [transform-style:preserve-3d]"
                        style={{
                          left:
                            BOARD.layers[guideLayer].x +
                            BOARD.layers[guideLayer].plane / 2,
                          top: guideCardTop,
                          width:
                            guideStep.id === "dense" ||
                            guideStep.id === "prediction"
                              ? 304
                              : 272,
                          transformOrigin: "top center",
                          // z≈0 (no perspective magnification); the inverse
                          // scale keeps the card at 1:1 while the board scales.
                          transform: `translateX(-50%) translateZ(1px) scale(${guideCardScale})`,
                        }}
                      >
                        <ExplanationCard
                          key={guideStep.id}
                          title={guideStep.title}
                          caption={guideStep.caption}
                          detail={guideDetail}
                          reducedMotion={reducedMotion}
                        >
                          {guideStep.id === "relu1" || guideStep.id === "relu2" ? (
                            guide?.relu ? (
                              <ReluInset
                                transition={guide.relu}
                                durationMs={guideStep.durationMs}
                                playing={explanation.state.playing}
                                reducedMotion={reducedMotion}
                              />
                            ) : null
                          ) : null}
                          {(guideStep.id === "pool1" || guideStep.id === "pool2") &&
                          guide?.pool ? (
                            <PoolInset
                              sample={guide.pool}
                              durationMs={guideStep.durationMs}
                              playing={explanation.state.playing}
                              reducedMotion={reducedMotion}
                            />
                          ) : null}
                          {guideStep.id === "dense" && guide?.dense ? (
                            <span className="grid gap-1">
                              <span className="text-[10px] text-muted-foreground">
                                Top features for class {guide.dense.unit} — red
                                pushes the score up, blue down
                              </span>
                              <DenseContributionChips
                                contributions={guide.dense}
                              />
                            </span>
                          ) : null}
                          {guideStep.id === "prediction" ? (
                            <>
                              <span className="flex items-center gap-2">
                                <span className="flex size-9 items-center justify-center rounded-lg bg-primary font-heading text-xl font-semibold text-primary-foreground">
                                  {run.predicted}
                                </span>
                                <span className="font-mono text-sm tabular-nums">
                                  {Math.round(
                                    (run.probabilities[run.predicted] ?? 0) * 100
                                  )}
                                  % confidence
                                </span>
                              </span>
                              <ExplanationCta
                                onTry={() =>
                                  toast("Network surgery is coming soon", {
                                    description:
                                      "Ablation mode will let you disable filters and watch this prediction change.",
                                  })
                                }
                              />
                            </>
                          ) : null}
                        </ExplanationCard>
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>

              {guideActive ? (
                <ExplanationControls controller={explanation} onExit={exitExplanation} />
              ) : null}

              <div
                className={cn(
                  "flex flex-wrap items-center gap-1.5",
                  guideActive && "pointer-events-none opacity-50"
                )}
              >
                <button
                  type="button"
                  onClick={resetView}
                  className={cn(
                    "flex shrink-0 items-center gap-1.5 rounded-md px-2 py-1 text-[11px] font-medium ring-1 transition-colors",
                    selected === null
                      ? "bg-primary text-primary-foreground ring-primary"
                      : "bg-muted/40 text-muted-foreground ring-foreground/10 hover:bg-muted hover:text-foreground"
                  )}
                >
                  Whole network
                </button>
                {run.stages.map((layerStage, layerIndex) => (
                  <LayerChip
                    key={layerStage.id}
                    index={layerIndex}
                    stage={layerStage}
                    selected={selected?.layer === layerIndex}
                    onSelect={(index) => openInspector(index, 0)}
                  />
                ))}
                <span className="flex items-center pl-1 text-muted-foreground">
                  <InfoTip label="How to inspect layers">
                    Click any layer on the board, a plane or a chip to pop out
                    its inspector: every feature map, the exact values, the
                    filter weights and the receptive field it sees. Inside, the
                    arrows move one layer at a time.
                  </InfoTip>
                </span>
              </div>
            </>
          )}
        </CardContent>
      </Card>

      <div className="grid items-start gap-4 lg:grid-cols-[20rem_minmax(0,1fr)]">
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              Input
              <InfoTip>
                Draw with mouse or touch, or load a real test digit. The ink is
                scaled to 20×20 and centred by mass — the same normalisation
                MNIST uses, which is what the weights expect.
              </InfoTip>
            </CardTitle>
            <CardAction>
              <Button variant="ghost" size="sm" onClick={replay}>
                <EraserIcon data-icon="inline-start" />
                Clear
              </Button>
            </CardAction>
          </CardHeader>
          <CardContent className="grid gap-3">
            <DigitPad clearToken={clearToken} centered={centered} onCommit={commitDrawing} />
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                Auto-centre ink
                <InfoTip>
                  Compute the ink bounding box, scale it to 20×20 and shift its
                  centre of mass to the middle of the 28×28 grid.
                </InfoTip>
              </span>
              <Switch checked={centered} onCheckedChange={setCentered} />
            </div>
            <div className="grid gap-2">
              <div className="flex items-center justify-between gap-2">
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  Test digits
                  <InfoTip>
                    Real MNIST test images, five per class, never seen during
                    training. Click one to load it; the model predicts without
                    being told the label.
                  </InfoTip>
                </span>
                <Button
                  variant="ghost"
                  size="sm"
                  className="h-6 px-2 text-[11px]"
                  onClick={() => {
                    const sample = samples[Math.floor(Math.random() * samples.length)]
                    if (sample) loadSample(sample)
                  }}
                >
                  <DicesIcon data-icon="inline-start" />
                  Random
                </Button>
              </div>
              <div className="grid grid-cols-5 gap-1.5">
                {starterMaps.map(({ sample, map }) => (
                  <button
                    key={`${sample.label}-${sample.image[400]}`}
                    type="button"
                    onClick={() => loadSample(sample)}
                    className={cn(
                      "aspect-square overflow-hidden rounded ring-1 transition-shadow hover:ring-2 hover:ring-primary",
                      origin.kind === "sample" && origin.sample === sample
                        ? "ring-2 ring-primary"
                        : "ring-foreground/10"
                    )}
                    aria-label="Load a test digit"
                  >
                    <HeatCanvas map={map} mode="grey" />
                  </button>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-1.5">
              Classification
              <InfoTip>
                Softmax over the ten class scores. The bar for each digit is the
                model&apos;s probability; the full score is the dot product of
                the 256 pooled features with that class&apos;s template.
              </InfoTip>
            </CardTitle>
          </CardHeader>
          <CardContent className="grid gap-4">
            <div className="flex flex-wrap items-center gap-4">
              <div className="flex items-center gap-3">
                <span className="flex size-14 items-center justify-center rounded-xl bg-primary font-heading text-3xl font-semibold text-primary-foreground">
                  {predicted ?? "—"}
                </span>
                <span className="grid gap-0.5">
                  <span className="text-xs text-muted-foreground">Prediction</span>
                  <span className="font-mono text-sm tabular-nums">
                    {run
                      ? `${Math.round((run.probabilities[run.predicted] ?? 0) * 100)}% confidence`
                      : "—"}
                  </span>
                </span>
              </div>
              {sampleTruth !== undefined ? (
                <Badge variant={correct ? "secondary" : "destructive"}>
                  {correct
                    ? `Correct — true label ${sampleTruth}`
                    : `Wrong — true label ${sampleTruth}`}
                </Badge>
              ) : (
                <span className="text-xs text-muted-foreground">
                  Drawn input — no true label
                </span>
              )}
            </div>
            {run ? (
              <div className={cn(guideActive && "pointer-events-none opacity-60")}>
                <ClassBars
                  run={run}
                  selectedClass={selected?.layer === 8 ? selectedClass : null}
                  onSelectClass={(unit) => {
                    setSelected({ layer: 8, channel: 0 })
                    setSelectedClass(unit)
                    setInspecting(true)
                  }}
                  reducedMotion={reducedMotion}
                />
              </div>
            ) : null}
          </CardContent>
        </Card>
      </div>

      <Dialog open={inspecting} onOpenChange={handleInspectOpenChange}>
        {run && stage && selected && model ? (
          <LayerInspector
            run={run}
            model={model}
            stage={stage}
            stageIndex={stageIndex}
            channel={channel}
            rfSizes={rfSizes}
            selectedClass={selectedClass}
            onSelectChannel={(next) =>
              setSelected((current) => (current ? { ...current, channel: next } : current))
            }
            onSelectClass={(unit) => setSelectedClass(unit)}
            onNavigate={selectLayer}
            probe={activeProbe}
            pinned={pinned}
            onProbe={setProbe}
            onTogglePin={togglePin}
            showNumbers={showNumbers}
            onShowNumbers={setShowNumbers}
            reducedMotion={reducedMotion}
            footprint={footprint}
            carried={carry && carry.target === stageIndex ? carry : null}
          />
        ) : null}
      </Dialog>

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
                  <CardTitle>Model facts &amp; insights</CardTitle>
                </span>
                <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
              </button>
            </CollapsibleTrigger>
          </CardHeader>
          <CollapsibleContent>
            <CardContent className="grid gap-2 text-sm text-muted-foreground">
              {model ? (
                <>
                  <p>
                    <a
                      href="https://github.com/onnx/models/tree/main/validated/vision/classification/mnist"
                      target="_blank"
                      rel="noreferrer"
                      className="underline underline-offset-2"
                    >
                      {model.source}
                    </a>{" "}
                    · {model.params.toLocaleString()} weights ·{" "}
                    {(model.testAccuracy * 100).toFixed(1)}% on{" "}
                    {model.testCount.toLocaleString()} held-out test digits (about{" "}
                    {Math.round((1 - model.testAccuracy) * model.testCount)} mistakes).
                  </p>
                  <p>
                    TypeScript re-implements the graph exactly: Conv 5×5 →
                    ReLU → MaxPool 2 → Conv 5×5 → ReLU → MaxPool 3 → dense 256
                    → classification 10 → softmax.
                    {selfCheck !== null
                      ? ` Self-check on a test digit: recomputed logits match the onnxruntime reference within ${selfCheck.toExponential(1)}.`
                      : ""}
                  </p>
                </>
              ) : null}
              {run ? (
                <>
                  <p>
                    Receptive field: a Conv 1 neuron sees {rfSizes[1]}×{rfSizes[1]} px of
                    the input, Conv 2 {rfSizes[4]}×{rfSizes[4]}, the final pooled map{" "}
                    {rfSizes[6]}×{rfSizes[6]} — the outermost border never reaches the
                    classifier.
                  </p>
                  <p>
                    This digit: ReLU 1 keeps{" "}
                    {Math.round((1 - run.stages[2].stats.zeroFraction) * 100)}% of its
                    values, ReLU 2 {Math.round((1 - run.stages[6].stats.zeroFraction) * 100)}
                    %. Everything else is exactly zero — the non-linearity carving away
                    evidence.
                  </p>
                  <p>
                    The classification head holds one 256-value template per
                    class: a logit is the dot product between the pooled
                    features and that template, plus a bias. Open{" "}
                    <em>Classification</em> and click a class to see the
                    template and its arithmetic.
                  </p>
                </>
              ) : null}
              <p>
                Hover a cell in an unfolded map to trace its receptive field back
                through the board — every contributing window lights up, including all
                input channels of a convolution.
              </p>
            </CardContent>
          </CollapsibleContent>
        </Card>
      </Collapsible>
    </div>
  )
}

function LayerInspector({
  run,
  model,
  stage,
  stageIndex,
  channel,
  rfSizes,
  selectedClass,
  onSelectChannel,
  onSelectClass,
  onNavigate,
  probe,
  pinned,
  onProbe,
  onTogglePin,
  showNumbers,
  onShowNumbers,
  reducedMotion,
  footprint,
  carried,
}: {
  run: ClassifierRun
  model: MnistModel
  stage: ClassifierStage
  stageIndex: number
  channel: number
  rfSizes: number[]
  selectedClass: number | null
  onSelectChannel: (channel: number) => void
  onSelectClass: (unit: number) => void
  onNavigate: (layer: number, channel: number, carry?: RfCarry | null) => void
  probe: Cell | null
  pinned: Cell | null
  onProbe: (cell: Cell | null) => void
  onTogglePin: (cell: Cell) => void
  showNumbers: boolean
  onShowNumbers: (value: boolean) => void
  reducedMotion: boolean
  footprint: RfTrace | null
  /** A deeper cell's footprint carried into this layer, if any. */
  carried: RfCarry | null
}) {
  const popRef = React.useRef<HTMLDivElement>(null)
  const bodyRef = React.useRef<HTMLDivElement>(null)
  const chipsRef = React.useRef<HTMLDivElement>(null)
  const mapRef = React.useRef<HTMLDivElement>(null)
  const trailRef = React.useRef<HTMLDivElement>(null)
  const templateRef = React.useRef<HTMLDivElement>(null)

  const isDense = stage.kind === "dense"
  const map = isDense ? undefined : stage.maps[Math.min(channel, stage.maps.length - 1)]
  const canShowNumbers = map !== undefined && map.rows * map.cols <= 256
  const denseUnit = selectedClass ?? run.predicted
  const template = isDense ? classTemplate(model, denseUnit) : []

  /** Magnitude composite per stage — what a whole stack collapses to in a thumbnail. */
  const composites = React.useMemo(
    () => run.stages.map((entry) => (entry.kind === "dense" ? null : compositeMap(entry.maps))),
    [run.stages]
  )

  /**
   * The receptive-field trail: for the active cell, the bounding box it draws
   * from on every earlier stage down to the input. For the dense head, every
   * class reads the whole pooled volume, so each step highlights its entire map.
   */
  const trail = React.useMemo(() => {
    const entries: {
      index: number
      stage: ClassifierStage
      region: RfRegion
      map: FloatMap
    }[] = []
    if (isDense) {
      for (let index = 0; index < TRACE_STAGE_COUNT; index += 1) {
        const entryStage = run.stages[index]
        const entryMap = composites[index]
        if (!entryStage || !entryMap) continue
        entries.push({
          index,
          stage: entryStage,
          region: {
            channel: "all",
            row: 0,
            col: 0,
            rows: entryStage.rows,
            cols: entryStage.cols,
          },
          map: entryMap,
        })
      }
      return entries
    }
    if (!footprint) return entries
    for (let index = 0; index <= stageIndex && index < footprint.length; index += 1) {
      const entryRegion = footprint[index]
      const entryStage = run.stages[index]
      if (!entryRegion || !entryStage || entryStage.kind === "dense") continue
      const entryMap =
        entryRegion.channel === "all"
          ? composites[index]
          : (entryStage.maps[entryRegion.channel] ?? composites[index])
      if (!entryMap) continue
      entries.push({ index, stage: entryStage, region: entryRegion, map: entryMap })
    }
    return entries
  }, [isDense, footprint, stageIndex, run.stages, composites])

  /**
   * The trace as a stage-indexed array, so a trail click can hand the same
   * regions to the earlier layer it opens.
   */
  const trailTrace = React.useMemo(() => {
    const regions: RfTrace = Array.from({ length: TRACE_STAGE_COUNT }, () => null)
    for (const entry of trail) regions[entry.index] = entry.region
    return regions
  }, [trail])

  /** The carried region on this layer, if a deeper cell's footprint led here. */
  const carriedRegion = carried ? (carried.trace[stageIndex] ?? null) : null
  const carriedOriginStage = carried ? run.stages[carried.origin.layer] : undefined
  const carriedCoords =
    carried && carried.origin.row !== null && carried.origin.col !== null
      ? ` (${carried.origin.row}, ${carried.origin.col})`
      : ""
  const carriedOriginSize = carried
    ? carriedOriginStage?.kind === "dense"
      ? MNIST_SIZE
      : (rfSizes[Math.min(carried.origin.layer, rfSizes.length - 1)] ?? 1)
    : 0

  /**
   * Move to another layer: when the target is a trail step of the current
   * trace, the footprint rides along and lands as a marking on its map.
   */
  const navigate = (layer: number, nextChannel: number) => {
    const entry = trail.find((candidate) => candidate.index === layer)
    onNavigate(
      layer,
      nextChannel,
      entry
        ? {
            target: layer,
            trace: trailTrace,
            origin:
              carried?.origin ??
              { layer: stageIndex, row: probe?.row ?? null, col: probe?.col ?? null },
          }
        : null
    )
  }

  const rfIndex = Math.min(stageIndex, rfSizes.length - 1)
  const rfSize = isDense ? MNIST_SIZE : (rfSizes[rfIndex] ?? 1)

  /** Pop-out: the body lands with a scale/rise whenever a layer opens. */
  useIsomorphicLayoutEffect(() => {
    const node = popRef.current
    if (!node || reducedMotion) return
    const animation = animate(node, {
      opacity: [0, 1],
      scale: [0.97, 1],
      y: [10, 0],
      duration: 300,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, reducedMotion])

  /** Map chips cascade in. */
  useIsomorphicLayoutEffect(() => {
    const node = chipsRef.current
    if (!node || reducedMotion) return
    const chips = node.querySelectorAll("[data-map-chip]")
    if (chips.length === 0) return
    const animation = animate(chips, {
      opacity: [0, 1],
      scale: [0.9, 1],
      duration: 260,
      ease: "out(3)",
      delay: stagger(16),
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, reducedMotion])

  /** The large map pops when another channel is selected. */
  useIsomorphicLayoutEffect(() => {
    const node = mapRef.current
    if (!node || reducedMotion) return
    const animation = animate(node, {
      opacity: [0, 1],
      scale: [0.97, 1],
      duration: 240,
      ease: "out(3)",
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, channel, reducedMotion])

  /** Receptive-field steps cascade in with the trail. */
  useIsomorphicLayoutEffect(() => {
    const node = trailRef.current
    if (!node || reducedMotion) return
    const steps = node.querySelectorAll("[data-rf-step]")
    if (steps.length === 0) return
    const animation = animate(steps, {
      opacity: [0, 1],
      y: [10, 0],
      duration: 260,
      ease: "out(3)",
      delay: stagger(28),
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, channel, trail.length, reducedMotion])

  /** Dense template cells reshuffle when the class changes. */
  useIsomorphicLayoutEffect(() => {
    const node = templateRef.current
    if (!node || reducedMotion) return
    const cells = node.querySelectorAll("[data-template]")
    if (cells.length === 0) return
    const animation = animate(cells, {
      opacity: [0, 1],
      scale: [0.8, 1],
      duration: 240,
      ease: "out(3)",
      delay: stagger(14),
    })
    return () => {
      animation.revert()
    }
  }, [stageIndex, denseUnit, reducedMotion])

  const arithmetic = React.useMemo(() => {
    if (!map || probe === null) return null
    if (stage.kind === "conv" && map.rows === MNIST_SIZE) {
      const previous = run.stages[stageIndex - 1]?.maps[0]
      if (!previous) return null
      const kernel = 5
      const pad = 2
      const terms: { a: number; w: number; product: number; row: number; col: number }[] = []
      const weights = model.conv1.weights.subarray(channel * 25, channel * 25 + 25)
      for (let kr = 0; kr < kernel; kr += 1) {
        for (let kc = 0; kc < kernel; kc += 1) {
          const row = probe.row + kr - pad
          const col = probe.col + kc - pad
          const inBounds = row >= 0 && row < MNIST_SIZE && col >= 0 && col < MNIST_SIZE
          const a = inBounds ? previous.data[row * MNIST_SIZE + col] : 0
          const w = weights[kr * kernel + kc]
          terms.push({ a, w, product: a * w, row, col })
        }
      }
      return { kind: "conv1" as const, terms, bias: model.conv1.bias[channel] }
    }
    if (stage.kind === "conv" && stageIndex === 4) {
      const previous = run.stages[3]
      if (!previous) return null
      const kernel = 5
      const pad = 2
      const partials: { channel: number; sum: number }[] = []
      for (let ic = 0; ic < previous.channels; ic += 1) {
        const input = previous.maps[ic]
        let sum = 0
        for (let kr = 0; kr < kernel; kr += 1) {
          for (let kc = 0; kc < kernel; kc += 1) {
            const row = probe.row + kr - pad
            const col = probe.col + kc - pad
            if (row < 0 || row >= previous.rows || col < 0 || col >= previous.cols) continue
            const w =
              model.conv2.weights[
                (channel * model.conv2.inChannels + ic) * 25 + kr * kernel + kc
              ]
            sum += input.data[row * previous.cols + col] * w
          }
        }
        partials.push({ channel: ic, sum })
      }
      return { kind: "conv2" as const, partials, bias: model.conv2.bias[channel] }
    }
    return null
  }, [map, probe, stage.kind, stageIndex, channel, run.stages, model])

  const value =
    map && probe ? map.data[probe.row * map.cols + probe.col] : undefined

  const region = footprint && stageIndex > 0 ? footprint[stageIndex - 1] : null
  const mapWidth = map
    ? Math.max(160, Math.min(560, map.cols * 36))
    : 352

  return (
    <DialogContent
      showCloseButton={false}
      onOpenAutoFocus={(event) => {
        event.preventDefault()
        bodyRef.current?.focus()
      }}
      className="@container/inspector flex h-[calc(100dvh-1.5rem)] w-[calc(100vw-1.5rem)] flex-col gap-0 overflow-hidden p-0 sm:max-w-[min(90rem,calc(100vw-1.5rem))]"
    >
      <header className="flex flex-wrap items-center gap-x-3 gap-y-2 border-b bg-muted/30 px-3 py-2">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <Badge variant="outline">{stageIndex + 1}</Badge>
          <DialogTitle className="text-sm font-medium">{stage.name}</DialogTitle>
          <span className="font-mono text-xs text-muted-foreground tabular-nums">
            {stageDims(stage)}
            {stage.params > 0 ? ` · ${stage.params.toLocaleString()} params` : ""}
          </span>
          <InfoTip>{stageInfo(stage)}</InfoTip>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          {map ? (
            <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
              Values
              <Switch
                checked={showNumbers && canShowNumbers}
                disabled={!canShowNumbers}
                onCheckedChange={onShowNumbers}
              />
            </span>
          ) : null}
          <span className="hidden font-mono text-xs text-muted-foreground tabular-nums @xl/inspector:inline">
            {value !== undefined && probe ? (
              <>
                ({probe.row}, {probe.col}) ={" "}
                <span className="font-semibold text-foreground">{formatValue(value)}</span>
              </>
            ) : (
              <>
                min {formatValue(stage.stats.min)} · max {formatValue(stage.stats.max)} ·{" "}
                {Math.round(stage.stats.zeroFraction * 100)}% zeros
              </>
            )}
          </span>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Previous layer"
            disabled={stageIndex <= 0}
            onClick={() => navigate(stageIndex - 1, 0)}
          >
            <ChevronLeftIcon />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            aria-label="Next layer"
            disabled={stageIndex >= run.stages.length - 1}
            onClick={() => navigate(stageIndex + 1, 0)}
          >
            <ChevronRightIcon />
          </Button>
          <DialogClose asChild>
            <Button variant="ghost" size="icon-sm" aria-label="Close">
              <XIcon />
            </Button>
          </DialogClose>
        </div>
      </header>

      <DialogDescription className="sr-only">
        {stage.name}: every feature map, the exact values, the filter weights
        and the receptive field it sees.
      </DialogDescription>

      <div
        ref={bodyRef}
        tabIndex={-1}
        className="min-h-0 flex-1 overflow-y-auto p-3 outline-none"
      >
        <div ref={popRef} className="grid gap-3">
          {isDense ? (
            <div className="grid gap-3 @3xl/inspector:grid-cols-[minmax(0,1fr)_auto]">
              <div className="grid gap-2">
                <div className="flex flex-wrap gap-1.5">
                  {Array.from(run.probabilities, (probability, unit) => (
                    <button
                      key={unit}
                      type="button"
                      onClick={() => onSelectClass(unit)}
                      className={cn(
                        "grid w-[4.5rem] gap-1 rounded-md p-1.5 text-left ring-1 transition-colors",
                        unit === denseUnit
                          ? "bg-primary/10 ring-primary"
                          : "bg-card/60 ring-foreground/10 hover:bg-muted"
                      )}
                    >
                      <span className="flex items-baseline justify-between gap-1">
                        <span className="font-mono text-sm font-semibold">{unit}</span>
                        <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
                          {Math.round(probability * 100)}%
                        </span>
                      </span>
                      <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
                        logit {formatValue(run.logits[unit])}
                      </span>
                    </button>
                  ))}
                </div>
                <p className="font-mono text-[11px] text-muted-foreground tabular-nums">
                  logit {denseUnit} = Σ 256 features · template + bias{" "}
                  {formatValue(model.dense.bias[denseUnit])} ={" "}
                  <span className="font-semibold text-foreground">
                    {formatValue(run.logits[denseUnit])}
                  </span>
                </p>
              </div>
              <div className="grid content-start gap-2">
                <span className="flex items-center gap-1.5 text-xs font-medium">
                  Template for class {denseUnit}
                  <InfoTip>
                    The learned weights for this class, reshaped to the 16×4×4
                    pooled grid — the same wiring the board draws from the 256×1
                    vector. Red pushes the score up, blue down; the logit is how
                    strongly the pooled features line up with this pattern.
                  </InfoTip>
                </span>
                <div ref={templateRef} className="grid grid-cols-8 gap-1">
                  {template.map((templateMap, index) => (
                    <span
                      key={index}
                      data-template={index}
                      className="relative aspect-square overflow-hidden rounded-[3px] ring-1 ring-foreground/10"
                    >
                      <HeatCanvas
                        map={templateMap}
                        bound={Math.max(
                          1e-6,
                          ...template.map((entry) =>
                            Math.max(...Array.from(entry.data, (value) => Math.abs(value)))
                          )
                        )}
                      />
                    </span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="grid gap-3 @3xl/inspector:grid-cols-[auto_minmax(0,1fr)_auto]">
              <div ref={chipsRef} className="flex max-w-72 flex-wrap content-start gap-1">
                {stage.maps.map((entry, index) => (
                  <button
                    key={index}
                    type="button"
                    data-map-chip={index}
                    onClick={() => onSelectChannel(index)}
                    aria-label={`Map ${index + 1}`}
                    className={cn(
                      "relative overflow-hidden rounded ring-1 transition-shadow hover:ring-2 hover:ring-primary",
                      index === channel ? "ring-2 ring-primary" : "ring-foreground/10"
                    )}
                    style={{
                      width: entry.rows >= 20 ? 44 : entry.rows >= 10 ? 38 : 32,
                      aspectRatio: "1 / 1",
                    }}
                  >
                    <HeatCanvas
                      map={entry}
                      mode={stage.kind === "input" ? "grey" : "heat"}
                      bound={stage.stats.bound}
                    />
                  </button>
                ))}
              </div>

              <div ref={mapRef} className="grid content-start gap-2">
                <div
                  role="grid"
                  aria-label={`${stage.name} map`}
                  tabIndex={0}
                  onKeyDown={(event) => {
                    if (!map) return
                    const base = probe ?? { row: Math.floor(map.rows / 2), col: Math.floor(map.cols / 2) }
                    let next = base
                    if (event.key === "ArrowLeft") next = { row: base.row, col: base.col - 1 }
                    else if (event.key === "ArrowRight") next = { row: base.row, col: base.col + 1 }
                    else if (event.key === "ArrowUp") next = { row: base.row - 1, col: base.col }
                    else if (event.key === "ArrowDown") next = { row: base.row + 1, col: base.col }
                    else return
                    event.preventDefault()
                    const clamped = {
                      row: Math.min(map.rows - 1, Math.max(0, next.row)),
                      col: Math.min(map.cols - 1, Math.max(0, next.col)),
                    }
                    if (pinned || carried) onTogglePin(clamped)
                    else onProbe(clamped)
                  }}
                  onPointerMove={(event) => {
                    if (!map) return
                    const rect = event.currentTarget.getBoundingClientRect()
                    const col = Math.floor(
                      ((event.clientX - rect.left) / rect.width) * map.cols
                    )
                    const row = Math.floor(
                      ((event.clientY - rect.top) / rect.height) * map.rows
                    )
                    if (row < 0 || col < 0 || row >= map.rows || col >= map.cols) return
                    onProbe(
                      probe && probe.row === row && probe.col === col ? probe : { row, col }
                    )
                  }}
                  onPointerLeave={() => onProbe(null)}
                  onClick={() => {
                    if (probe) onTogglePin(probe)
                  }}
                  className="relative cursor-crosshair rounded-md outline-none ring-1 ring-foreground/10 focus-visible:ring-2 focus-visible:ring-ring"
                  style={{ width: mapWidth, maxWidth: "100%" }}
                >
                  {map ? (
                    <>
                      <HeatCanvas
                        map={map}
                        mode={stage.kind === "input" ? "grey" : "heat"}
                        bound={stage.stats.bound}
                      />
                      {showNumbers && canShowNumbers ? (
                        <span
                          aria-hidden
                          className="pointer-events-none absolute inset-0 grid"
                          style={{ gridTemplateColumns: `repeat(${map.cols}, minmax(0, 1fr))` }}
                        >
                          {Array.from({ length: map.rows * map.cols }, (_, index) => (
                            <span
                              key={index}
                              className="flex min-w-0 items-center justify-center font-mono text-[8px] leading-none tabular-nums"
                              style={{
                                color:
                                  Math.abs(map.data[index]) / stage.stats.bound > 0.55
                                    ? "#f8fafc"
                                    : "#0f172a",
                              }}
                            >
                              {formatValue(map.data[index])}
                            </span>
                          ))}
                        </span>
                      ) : null}
                    </>
                  ) : null}
                  {probe && map ? (
                    <span
                      aria-hidden
                      className="pointer-events-none absolute rounded-[2px]"
                      style={{
                        left: `${(probe.col / map.cols) * 100}%`,
                        top: `${(probe.row / map.rows) * 100}%`,
                        width: `${100 / map.cols}%`,
                        height: `${100 / map.rows}%`,
                        backgroundColor: RF_ACCENT_SOFT,
                        boxShadow: `inset 0 0 0 2px ${RF_ACCENT}, 0 0 0 1px ${RF_ACCENT_EDGE}`,
                      }}
                    />
                  ) : null}
                  {carriedRegion && map ? (
                    <CarriedRegion
                      key={`${stageIndex}-${carriedRegion.row}-${carriedRegion.col}`}
                      region={carriedRegion}
                      rows={map.rows}
                      cols={map.cols}
                    />
                  ) : null}
                </div>
                <p className="font-mono text-[11px] text-muted-foreground tabular-nums">
                  {pinned ? "pinned · " : carried ? "carried · " : ""}
                  {carried && carriedRegion
                    ? `from ${carriedOriginStage?.name ?? "?"}${carriedCoords}: ${carriedRegion.rows}×${carriedRegion.cols} @ (${carriedRegion.row}, ${carriedRegion.col})${carriedRegion.channel === "all" && stage.channels > 1 ? " · every map" : ""}`
                    : region
                      ? `RF on ${run.stages[stageIndex - 1]?.name}: ${region.rows}×${region.cols} @ (${region.row}, ${region.col})${region.channel === "all" && (run.stages[stageIndex - 1]?.channels ?? 1) > 1 ? " · every map" : ""}`
                      : stageIndex === 0
                        ? "hover a cell to read its value"
                        : "hover a cell to trace its receptive field · click to pin"}
                </p>
              </div>

              <div className="grid content-start gap-2">
                {stage.kind === "conv" ? (
                  <FilterPanel
                    model={model}
                    stageIndex={stageIndex}
                    channel={channel}
                  />
                ) : null}
                {stage.kind === "relu" ? (
                  <p className="max-w-56 text-xs text-muted-foreground">
                    max(0, x) zeroed{" "}
                    <span className="font-mono tabular-nums">
                      {Math.round(stage.stats.zeroFraction * 100)}%
                    </span>{" "}
                    of the values coming from {run.stages[stageIndex - 1]?.name}.
                  </p>
                ) : null}
                {stage.kind === "pool" ? (
                  <p className="max-w-56 text-xs text-muted-foreground">
                    {stageIndex === 3 ? "2×2 / stride 2" : "3×3 / stride 3"} max
                    windows, no weights — each window keeps its strongest
                    activation.
                  </p>
                ) : null}
                {stage.kind === "flatten" ? (
                  <p className="max-w-56 text-xs text-muted-foreground">
                    256×1, channels first: hover a cell to see the pooled cell it
                    came from — every value feeds all 10 class scores.
                  </p>
                ) : null}
              </div>
            </div>
          )}

      {trail.length > 1 ? (
        <section className="grid gap-2 rounded-lg bg-background/60 p-2.5 ring-1 ring-foreground/10">
          <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
            <span className="flex items-center gap-1.5 text-xs font-medium">
              Receptive field
              <InfoTip>
                Every activation looks back at a window of the layer before it —
                and through it at a growing patch of the input. Steps run from
                the input to this layer; click one to open it with this cell&apos;s
                region still marked (dashed). Clicking a cell there starts a new
                trace from that cell.
              </InfoTip>
            </span>
            <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
              {carried
                ? `carried from ${carriedOriginStage?.name ?? "?"}${carriedCoords} · sees ${carriedOriginSize}×${carriedOriginSize} px`
                : isDense
                  ? "each class reads all 28×28 input px"
                  : `this cell sees ${rfSize}×${rfSize} px of the input`}
            </span>
          </header>
          <div ref={trailRef} className="flex flex-wrap gap-2">
            {trail.map(({ index, stage: entryStage, region: entryRegion, map: entryMap }) => (
              <button
                key={index}
                type="button"
                data-rf-step={index}
                onClick={() =>
                  navigate(index, entryRegion.channel === "all" ? 0 : entryRegion.channel)
                }
                title={`Inspect ${entryStage.name}`}
                className={cn(
                  "grid w-[4.5rem] shrink-0 content-start gap-1 rounded-md p-1 text-left ring-1 transition-colors",
                  index === stageIndex
                    ? "bg-primary/10 ring-primary"
                    : "ring-foreground/10 hover:bg-muted/60"
                )}
              >
                <span className="relative block size-16 overflow-hidden rounded-[3px] ring-1 ring-foreground/10">
                  <HeatCanvas
                    map={entryMap}
                    mode={entryStage.kind === "input" ? "grey" : "heat"}
                    bound={entryStage.stats.bound}
                  />
                  <RegionOverlay
                    key={`${entryRegion.row}-${entryRegion.col}-${entryRegion.channel}`}
                    region={entryRegion}
                    rows={entryStage.rows}
                    cols={entryStage.cols}
                  />
                </span>
                <span className="truncate text-[10px] font-medium">
                  {entryStage.name.replace("MaxPool", "Pool")}
                </span>
                <span className="font-mono text-[9px] text-muted-foreground tabular-nums">
                  {entryRegion.rows}×{entryRegion.cols}
                  {entryRegion.channel === "all" && entryStage.channels > 1
                    ? ` ×${entryStage.channels}`
                    : ""}
                </span>
              </button>
            ))}
          </div>
        </section>
      ) : null}

      {arithmetic ? (
        <section className="grid gap-2 rounded-lg bg-background/60 p-2.5 ring-1 ring-foreground/10">
          <header className="flex flex-wrap items-baseline justify-between gap-2">
            <span className="text-xs font-medium">
              The arithmetic
              <InfoTip>
                The real numbers: each input value times its weight, summed and
                biased. That sum is the pre-activation; ReLU then keeps the
                positive part.
              </InfoTip>
            </span>
            <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
              Σ + bias {formatValue(arithmetic.bias)} ={" "}
              <span className="font-semibold text-foreground">
                {formatValue(arithmetic.kind === "conv1"
                  ? arithmetic.terms.reduce((total, term) => total + term.product, 0) +
                    arithmetic.bias
                  : arithmetic.partials.reduce((total, entry) => total + entry.sum, 0) +
                    arithmetic.bias)}
              </span>
            </span>
          </header>
          {arithmetic.kind === "conv1" ? (
            <div className="flex flex-wrap gap-1 font-mono text-[10px]">
              {arithmetic.terms.map((term, index) => (
                <span
                  key={index}
                  className="rounded bg-muted/60 px-1.5 py-0.5 whitespace-nowrap ring-1 ring-foreground/10"
                >
                  {formatValue(term.a)}·{formatValue(term.w)} ={" "}
                  <span className="font-semibold">{formatValue(term.product)}</span>
                </span>
              ))}
            </div>
          ) : (
            <div className="flex flex-wrap gap-1 font-mono text-[10px]">
              {arithmetic.partials.map((entry) => (
                <span
                  key={entry.channel}
                  className="rounded bg-muted/60 px-1.5 py-0.5 whitespace-nowrap ring-1 ring-foreground/10"
                >
                  Σ map {entry.channel + 1} = {formatValue(entry.sum)}
                </span>
              ))}
            </div>
          )}
        </section>
      ) : null}
        </div>
      </div>
    </DialogContent>
  )
}

function FilterPanel({
  model,
  stageIndex,
  channel,
}: {
  model: MnistModel
  stageIndex: number
  channel: number
}) {
  const layer = stageIndex === 1 ? model.conv1 : model.conv2
  const weightScale = React.useMemo(() => {
    let max = 0
    for (const value of layer.weights) max = Math.max(max, Math.abs(value))
    return max || 1
  }, [layer.weights])

  return (
    <div className="grid content-start gap-2">
      <span className="flex items-center gap-1.5 text-xs font-medium">
        Filter {channel + 1}
        <InfoTip>
          The learned weights behind this map: one 5×5 grid per input channel.
          Green is positive, red negative; every position of the incoming
          volume is dotted with them.
        </InfoTip>
      </span>
      <div className="flex flex-wrap gap-2">
        {Array.from({ length: layer.inChannels }, (_, inputChannel) => (
          <div
            key={inputChannel}
            className="grid w-24 gap-[2px]"
            style={{ gridTemplateColumns: "repeat(5, minmax(0, 1fr))" }}
          >
            {Array.from({ length: 25 }, (_, index) => {
              const value =
                layer.weights[
                  (channel * layer.inChannels + inputChannel) * 25 + index
                ]
              const alpha = Math.min(0.9, Math.abs(value) / weightScale + 0.12)
              return (
                <span
                  key={index}
                  title={`${value >= 0 ? "+" : ""}${value.toFixed(3)}`}
                  className="flex aspect-square items-center justify-center rounded-[3px] text-[8px] font-mono tabular-nums ring-1 ring-foreground/10"
                  style={{
                    backgroundColor:
                      value > 0
                        ? `rgba(34,197,94,${alpha})`
                        : value < 0
                          ? `rgba(239,68,68,${alpha})`
                          : undefined,
                  }}
                >
                  {Math.abs(value) > weightScale * 0.45 ? formatValue(value) : ""}
                </span>
              )
            })}
          </div>
        ))}
      </div>
      <p className="font-mono text-[10px] text-muted-foreground tabular-nums">
        bias {formatValue(layer.bias[channel])} ·{" "}
        {layer.inChannels * 25 + 1} weights
      </p>
    </div>
  )
}
