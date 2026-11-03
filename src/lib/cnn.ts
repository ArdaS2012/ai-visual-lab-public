import {
  kernelFromPreset,
  outputDimension,
  paddingFor,
  poolChannels,
  readZero,
  zeros,
  type Matrix,
  type PaddingMode,
} from "./convolution"
import { mulberry32 } from "./cnn-scenes"

/** A single-channel feature map. */
export type FeatureMap = Matrix

/** Channels-first image volume: `[channel][row][col]`. */
export type Tensor = Matrix[]

export type Filter = {
  /** `inChannels × k × k` weights. */
  weights: Tensor
  bias: number
  /** Set for hand-designed filters (e.g. "Vertical edges"). */
  label?: string
}

export type ConvSpec = {
  kind: "conv"
  filters: Filter[]
  kernelSize: number
  stride: number
  padding: PaddingMode
  pad: number
}

export type ReluSpec = { kind: "relu" }

export type PoolSpec = {
  kind: "pool"
  mode: "max" | "avg"
  window: number
  stride: number
  padding: PaddingMode
  pad: number
}

export type InputSpec = { kind: "input" }

export type LayerSpec = InputSpec | ConvSpec | ReluSpec | PoolSpec

export type ChannelStats = {
  min: number
  max: number
  mean: number
  zeros: number
  count: number
}

export type TensorStats = {
  min: number
  max: number
  mean: number
  /** Fraction of exactly-zero values (ReLU sparsity). */
  zeroFraction: number
  count: number
  /** Largest magnitude — the heatmap's symmetric colour bound. */
  bound: number
  perChannel: ChannelStats[]
}

export type StageResult = {
  spec: LayerSpec
  /** Post-operation tensor this stage hands to the next layer. */
  output: Tensor
  /** ReLU only: the tensor it was applied to (what the gate saw). */
  preActivation?: Tensor
  channels: number
  rows: number
  cols: number
  params: number
  stats: TensorStats
}

export function tensorShape(tensor: Tensor): {
  channels: number
  rows: number
  cols: number
} {
  return {
    channels: tensor.length,
    rows: tensor[0]?.length ?? 0,
    cols: tensor[0]?.[0]?.length ?? 0,
  }
}

/** Convolution of a channel stack with one kernel per input channel, per filter. */
export function convolveTensor(input: Tensor, spec: ConvSpec): Tensor {
  const { rows: height, cols: width } = tensorShape(input)
  if (height === 0 || width === 0 || spec.filters.length === 0) return []

  const kernelSize = spec.kernelSize
  const stride = Math.max(1, Math.trunc(spec.stride))
  const pad = spec.pad
  const rows = outputDimension(height, kernelSize, stride, pad)
  const cols = outputDimension(width, kernelSize, stride, pad)
  if (rows <= 0 || cols <= 0) return []

  return spec.filters.map((filter) => {
    const map = zeros(rows, cols)
    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        let sum = filter.bias
        for (let m = 0; m < kernelSize; m += 1) {
          for (let n = 0; n < kernelSize; n += 1) {
            const row = i * stride + m - pad
            const col = j * stride + n - pad
            for (let c = 0; c < input.length; c += 1) {
              sum +=
                readZero(input[c], row, col) *
                (filter.weights[c]?.[m]?.[n] ?? 0)
            }
          }
        }
        map[i][j] = sum
      }
    }
    return map
  })
}

export function reluTensor(input: Tensor): Tensor {
  return input.map((map) => map.map((row) => row.map((value) => (value > 0 ? value : 0))))
}

export function poolTensor(
  input: Tensor,
  spec: PoolSpec
): { output: Tensor; rows: number; cols: number; pad: number } {
  const result = poolChannels(input, spec.window, {
    stride: spec.stride,
    padding: spec.padding,
    pad: spec.pad,
    mode: spec.mode,
  })
  return {
    output: result.valid ? result.perChannel : [],
    rows: Math.max(0, result.rows),
    cols: Math.max(0, result.cols),
    pad: result.pad,
  }
}

export function tensorStats(tensor: Tensor): TensorStats {
  const perChannel: ChannelStats[] = tensor.map((map) => {
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY
    let sum = 0
    let zeros = 0
    let count = 0
    for (const row of map) {
      for (const value of row) {
        if (value < min) min = value
        if (value > max) max = value
        sum += value
        if (value === 0) zeros += 1
        count += 1
      }
    }
    if (count === 0) return { min: 0, max: 0, mean: 0, zeros: 0, count: 0 }
    return { min, max, mean: sum / count, zeros, count }
  })

  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let sum = 0
  let zeros = 0
  let count = 0
  for (const channel of perChannel) {
    if (channel.count === 0) continue
    if (channel.min < min) min = channel.min
    if (channel.max > max) max = channel.max
    sum += channel.mean * channel.count
    zeros += channel.zeros
    count += channel.count
  }

  if (count === 0) {
    return { min: 0, max: 0, mean: 0, zeroFraction: 0, count: 0, bound: 1, perChannel }
  }

  return {
    min,
    max,
    mean: sum / count,
    zeroFraction: zeros / count,
    count,
    bound: Math.max(1e-6, Math.max(Math.abs(min), Math.abs(max))),
    perChannel,
  }
}

export function convParams(spec: ConvSpec, inChannels: number): number {
  return spec.filters.length * (inChannels * spec.kernelSize * spec.kernelSize + 1)
}

/** Runs the layered forward pass and captures every stage's activation volume. */
export function forwardPass(input: Tensor, specs: LayerSpec[]): StageResult[] {
  const stages: StageResult[] = []
  let current = input
  let inChannels = tensorShape(input).channels

  for (const spec of specs) {
    if (spec.kind === "input") {
      const shape = tensorShape(input)
      stages.push({
        spec,
        output: input,
        channels: shape.channels,
        rows: shape.rows,
        cols: shape.cols,
        params: 0,
        stats: tensorStats(input),
      })
      continue
    }

    if (spec.kind === "conv") {
      const output = convolveTensor(current, spec)
      const shape = tensorShape(output)
      stages.push({
        spec,
        output,
        channels: shape.channels,
        rows: shape.rows,
        cols: shape.cols,
        params: convParams(spec, inChannels),
        stats: tensorStats(output),
      })
      current = output
      inChannels = shape.channels
      continue
    }

    if (spec.kind === "relu") {
      const output = reluTensor(current)
      const shape = tensorShape(output)
      stages.push({
        spec,
        output,
        preActivation: current,
        channels: shape.channels,
        rows: shape.rows,
        cols: shape.cols,
        params: 0,
        stats: tensorStats(output),
      })
      current = output
      continue
    }

    const { output, rows, cols } = poolTensor(current, spec)
    const shape = tensorShape(output)
    stages.push({
      spec,
      output,
      channels: shape.channels,
      rows: shape.rows || rows,
      cols: shape.cols || cols,
      params: 0,
      stats: tensorStats(output),
    })
    current = output
  }

  return stages
}

export function totalParams(stages: StageResult[]): number {
  return stages.reduce((total, stage) => total + stage.params, 0)
}

/** Human names for a spec chain: Input, Conv 1, ReLU 1, MaxPool 1, Conv 2, … */
export function layerNames(specs: LayerSpec[]): string[] {
  let block = 0
  return specs.map((spec) => {
    if (spec.kind === "input") return "Input"
    if (spec.kind === "relu") return `ReLU ${block}`
    if (spec.kind === "conv") {
      block += 1
      return `Conv ${block}`
    }
    return `${spec.mode === "max" ? "Max" : "Avg"}Pool ${block}`
  })
}

/** Region of an earlier stage that produced one cell of a later stage. */
export type RfRegion = {
  /** `"all"` when every channel of that stage contributed. */
  channel: number | "all"
  row: number
  col: number
  rows: number
  cols: number
}

/** Geometry-only view of a layer: everything tracing needs, no weights. */
export type TraceableSpec =
  | { kind: "input" }
  | { kind: "conv"; kernelSize: number; stride: number; pad: number }
  | { kind: "relu" }
  | { kind: "pool"; window: number; stride: number; pad: number }

/**
 * Walks backwards from one output cell and reports, for every earlier stage,
 * the rectangular window (and channel range) that fed it. Coordinates are in
 * each stage's own grid and may fall outside it when padding was involved.
 */
export function traceReceptiveField(
  stages: { spec: TraceableSpec }[],
  stageIndex: number,
  channel: number,
  i: number,
  j: number
): (RfRegion | null)[] {
  const regions: (RfRegion | null)[] = stages.map(() => null)
  if (stageIndex < 0 || stageIndex >= stages.length) return regions

  let current: RfRegion = { channel, row: i, col: j, rows: 1, cols: 1 }
  regions[stageIndex] = current

  for (let index = stageIndex; index >= 1; index -= 1) {
    const spec = stages[index].spec
    let previous: RfRegion

    if (spec.kind === "conv") {
      previous = {
        channel: "all",
        row: current.row * spec.stride - spec.pad,
        col: current.col * spec.stride - spec.pad,
        rows: spec.kernelSize,
        cols: spec.kernelSize,
      }
    } else if (spec.kind === "pool") {
      previous = {
        channel: current.channel,
        row: current.row * spec.stride - spec.pad,
        col: current.col * spec.stride - spec.pad,
        rows: spec.window,
        cols: spec.window,
      }
    } else {
      previous = { ...current }
    }

    regions[index - 1] = previous
    current = previous
  }

  return regions
}

/**
 * Receptive field of every stage measured in input pixels. Conv and pool
 * grow it by `(k − 1) · jump`; jump multiplies by the stride each time.
 */
export function receptiveFieldSizes(specs: TraceableSpec[]): number[] {
  let field = 1
  let jump = 1
  return specs.map((spec) => {
    if (spec.kind === "conv") {
      field += (spec.kernelSize - 1) * jump
      jump *= Math.max(1, Math.trunc(spec.stride))
    } else if (spec.kind === "pool") {
      field += (spec.window - 1) * jump
      jump *= Math.max(1, Math.trunc(spec.stride))
    }
    return field
  })
}

/** Hand-designed first-layer filters: interpretable by construction. */
export const INTERPRETABLE_FILTERS: { id: string; label: string }[] = [
  { id: "sobel-x", label: "Vertical edges" },
  { id: "sobel-y", label: "Horizontal edges" },
  { id: "laplacian", label: "Blobs & corners" },
  { id: "outline", label: "Outline" },
  { id: "sharpen", label: "Sharpen" },
  { id: "emboss", label: "Emboss" },
  { id: "gaussian", label: "Blur" },
  { id: "box", label: "Box blur" },
]

function gaussianSampler(seed: number) {
  const random = mulberry32(seed)
  return () => {
    let u = 0
    let v = 0
    while (u === 0) u = random()
    while (v === 0) v = random()
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v)
  }
}

/**
 * A pyramid of hand-designed filters (shared across input channels, scaled by
 * 1/inChannels so RGB and greyscale responses stay comparable) or seeded
 * He-style random weights. Nothing here is trained — see the insights panel.
 */
export function buildConvFilters(options: {
  outChannels: number
  inChannels: number
  kernelSize: number
  mode: "interpretable" | "random"
  seed: number
}): Filter[] {
  const { outChannels, inChannels, kernelSize, mode, seed } = options

  if (mode === "interpretable") {
    return Array.from({ length: outChannels }, (_, index) => {
      const preset = INTERPRETABLE_FILTERS[index % INTERPRETABLE_FILTERS.length]
      const kernel = kernelFromPreset(preset.id, kernelSize)
      const scale = 1 / Math.max(1, inChannels)
      return {
        weights: Array.from({ length: inChannels }, () =>
          kernel.map((row) => row.map((value) => value * scale))
        ),
        bias: 0,
        label: preset.label,
      }
    })
  }

  const gaussian = gaussianSampler(seed)
  const fanIn = Math.max(1, inChannels * kernelSize * kernelSize)
  const scale = Math.sqrt(2 / fanIn)
  const biasRandom = mulberry32(seed + 7919)

  return Array.from({ length: outChannels }, () => ({
    weights: Array.from({ length: inChannels }, () =>
      Array.from({ length: kernelSize }, () =>
        Array.from({ length: kernelSize }, () => gaussian() * scale)
      )
    ),
    bias: (biasRandom() - 0.5) * 0.3,
  }))
}

export type NetConfig = {
  size: number
  inChannels: number
  /** Requested conv → ReLU → pool blocks. */
  depth: number
  /** Filters per conv layer. */
  filters: number
  kernelSize: number
  padding: "same" | "valid"
  poolWindow: number
  poolMode: "max" | "avg"
  filterMode: "interpretable" | "random"
  seed: number
}

export type Network = {
  specs: LayerSpec[]
  /** Blocks that actually fit; < requested when the map shrinks away. */
  blocks: number
  requestedBlocks: number
}

/**
 * Builds the layer chain, stopping early when another block would shrink a
 * spatial dimension to zero (valid padding + deep stacks).
 */
export function buildNetwork(config: NetConfig): Network {
  const specs: LayerSpec[] = [{ kind: "input" }]
  const pad = paddingFor(config.padding, config.kernelSize)
  let dimension = config.size
  let inChannels = config.inChannels
  let blocks = 0

  for (let block = 0; block < config.depth; block += 1) {
    const convRows = outputDimension(dimension, config.kernelSize, 1, pad)
    if (convRows <= 0) break

    const filters = buildConvFilters({
      outChannels: config.filters,
      inChannels,
      kernelSize: config.kernelSize,
      mode: block === 0 ? config.filterMode : "random",
      seed: config.seed + block * 104729,
    })

    specs.push({
      kind: "conv",
      filters,
      kernelSize: config.kernelSize,
      stride: 1,
      padding: config.padding,
      pad,
    })
    specs.push({ kind: "relu" })

    const pooledRows = outputDimension(convRows, config.poolWindow, config.poolWindow, 0)
    if (pooledRows <= 0) {
      // Conv + ReLU still fit; a pool would not, so the chain ends here.
      blocks += 1
      break
    }

    specs.push({
      kind: "pool",
      mode: config.poolMode,
      window: config.poolWindow,
      stride: config.poolWindow,
      padding: "valid",
      pad: 0,
    })

    dimension = pooledRows
    inChannels = config.filters
    blocks += 1
  }

  return { specs, blocks, requestedBlocks: config.depth }
}
