import { traceReceptiveField, type RfRegion, type TraceableSpec } from "./cnn"

export const MNIST_SIZE = 28
export const MNIST_CLASSES = 10

export type FloatMap = { data: Float32Array; rows: number; cols: number }

export type MnistConvLayer = {
  /** `[out][in][k][k]`, row-major. */
  weights: Float32Array
  bias: Float32Array
  kernel: number
  pad: number
  inChannels: number
  outChannels: number
}

export type MnistModel = {
  source: string
  testAccuracy: number
  testCount: number
  conv1: MnistConvLayer
  conv2: MnistConvLayer
  dense: { weights: Float32Array; bias: Float32Array; inputs: number; units: number }
  params: number
}

export type SampleDigit = {
  label: number
  /** 784 bytes, 0…255, MNIST order. */
  image: Uint8Array
  /** Reference values from the NumPy/onnxruntime forward pass. */
  check: {
    logits: number[]
    conv1Sum: number
    relu1Zeros: number
    pool2Sum: number
  }
}

export type ChannelStats = {
  min: number
  max: number
  mean: number
  zeros: number
  count: number
}

export type StageStats = {
  min: number
  max: number
  mean: number
  zeroFraction: number
  /** Largest magnitude — the heatmap's symmetric colour bound. */
  bound: number
  perChannel: ChannelStats[]
}

export type StageKind = "input" | "conv" | "relu" | "pool" | "flatten" | "dense"

export type ClassifierStage = {
  id: string
  name: string
  kind: StageKind
  maps: FloatMap[]
  rows: number
  cols: number
  channels: number
  params: number
  stats: StageStats
  /** Geometry for receptive-field tracing (map stages only). */
  spec?: TraceableSpec
}

export type ClassifierRun = {
  stages: ClassifierStage[]
  logits: Float32Array
  probabilities: Float32Array
  predicted: number
}

/**
 * Geometry of the pretrained graph: Conv 5×5 same → ReLU → MaxPool 2 →
 * Conv 5×5 same → ReLU → MaxPool 3 → flatten 16·4·4 → dense 10.
 */
export const MODEL_SPECS: TraceableSpec[] = [
  { kind: "input" },
  { kind: "conv", kernelSize: 5, stride: 1, pad: 2 },
  { kind: "relu" },
  { kind: "pool", window: 2, stride: 2, pad: 0 },
  { kind: "conv", kernelSize: 5, stride: 1, pad: 2 },
  { kind: "relu" },
  { kind: "pool", window: 3, stride: 3, pad: 0 },
]

export const TRACE_STAGE_COUNT = MODEL_SPECS.length

function base64ToBytes(base64: string): Uint8Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index)
  }
  return bytes
}

export function decodeFloats(base64: string): Float32Array {
  const bytes = base64ToBytes(base64)
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4)
}

export function decodeBytes(base64: string): Uint8Array {
  return base64ToBytes(base64)
}

type RawModel = {
  source: string
  testAccuracy: number
  testCount: number
  layers: {
    name: string
    shape: number[]
    kernel?: number
    pad?: number
    weights: string
    bias: string
  }[]
}

export function parseModel(raw: unknown): MnistModel {
  const data = raw as RawModel
  if (!data || !Array.isArray(data.layers)) {
    throw new Error("Model file has no layers")
  }
  const byName = new Map(data.layers.map((layer) => [layer.name, layer]))
  const conv1 = byName.get("conv1")
  const conv2 = byName.get("conv2")
  const dense = byName.get("dense")
  if (!conv1 || !conv2 || !dense) {
    throw new Error("Model is missing conv1/conv2/dense")
  }

  const model: MnistModel = {
    source: data.source,
    testAccuracy: data.testAccuracy,
    testCount: data.testCount,
    conv1: {
      weights: decodeFloats(conv1.weights),
      bias: decodeFloats(conv1.bias),
      kernel: conv1.kernel ?? 5,
      pad: conv1.pad ?? 2,
      inChannels: conv1.shape[1],
      outChannels: conv1.shape[0],
    },
    conv2: {
      weights: decodeFloats(conv2.weights),
      bias: decodeFloats(conv2.bias),
      kernel: conv2.kernel ?? 5,
      pad: conv2.pad ?? 2,
      inChannels: conv2.shape[1],
      outChannels: conv2.shape[0],
    },
    dense: {
      weights: decodeFloats(dense.weights),
      bias: decodeFloats(dense.bias),
      inputs: dense.shape[0],
      units: dense.shape[1],
    },
    params: 0,
  }

  model.params =
    model.conv1.outChannels * (model.conv1.inChannels * model.conv1.kernel ** 2 + 1) +
    model.conv2.outChannels * (model.conv2.inChannels * model.conv2.kernel ** 2 + 1) +
    model.dense.inputs * model.dense.units +
    model.dense.units
  return model
}

export function parseSamples(raw: unknown): SampleDigit[] {
  const list = raw as { label: number; image: string; check: SampleDigit["check"] }[]
  if (!Array.isArray(list)) throw new Error("Samples file is not an array")
  return list.map((entry) => ({
    label: entry.label,
    image: decodeBytes(entry.image),
    check: entry.check,
  }))
}

export function sampleGrid(sample: SampleDigit): Float32Array {
  const grid = new Float32Array(sample.image.length)
  for (let index = 0; index < sample.image.length; index += 1) {
    grid[index] = sample.image[index] / 255
  }
  return grid
}

function stageStats(maps: FloatMap[]): StageStats {
  const perChannel: ChannelStats[] = maps.map((map) => {
    let min = Number.POSITIVE_INFINITY
    let max = Number.NEGATIVE_INFINITY
    let sum = 0
    let zeros = 0
    for (const value of map.data) {
      if (value < min) min = value
      if (value > max) max = value
      sum += value
      if (value === 0) zeros += 1
    }
    const count = map.data.length
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
    return { min: 0, max: 0, mean: 0, zeroFraction: 0, bound: 1, perChannel }
  }
  return {
    min,
    max,
    mean: sum / count,
    zeroFraction: zeros / count,
    bound: Math.max(1e-6, Math.max(Math.abs(min), Math.abs(max))),
    perChannel,
  }
}

function mapFrom(data: Float32Array, rows: number, cols: number): FloatMap {
  return { data, rows, cols }
}

/** Same-padded, stride-1 convolution. Weights are `[out][in][k][k]`. */
function convolve(
  input: Float32Array,
  inChannels: number,
  rows: number,
  cols: number,
  layer: MnistConvLayer
): Float32Array {
  const { kernel, pad, outChannels } = layer
  const out = new Float32Array(outChannels * rows * cols)
  const plane = rows * cols

  for (let oc = 0; oc < outChannels; oc += 1) {
    const outOffset = oc * plane
    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        let sum = layer.bias[oc]
        for (let ic = 0; ic < inChannels; ic += 1) {
          const weightOffset = (oc * inChannels + ic) * kernel * kernel
          const inOffset = ic * plane
          for (let kr = 0; kr < kernel; kr += 1) {
            const sourceRow = row + kr - pad
            if (sourceRow < 0 || sourceRow >= rows) continue
            for (let kc = 0; kc < kernel; kc += 1) {
              const sourceCol = col + kc - pad
              if (sourceCol < 0 || sourceCol >= cols) continue
              sum +=
                input[inOffset + sourceRow * cols + sourceCol] *
                layer.weights[weightOffset + kr * kernel + kc]
            }
          }
        }
        out[outOffset + row * cols + col] = sum
      }
    }
  }
  return out
}

function relu(input: Float32Array): Float32Array {
  const out = new Float32Array(input.length)
  for (let index = 0; index < input.length; index += 1) {
    const value = input[index]
    out[index] = value > 0 ? value : 0
  }
  return out
}

function maxPool(
  input: Float32Array,
  channels: number,
  rows: number,
  cols: number,
  kernel: number,
  stride: number
): FloatMap {
  const outRows = Math.floor((rows - kernel) / stride) + 1
  const outCols = Math.floor((cols - kernel) / stride) + 1
  const out = new Float32Array(channels * outRows * outCols)
  const inPlane = rows * cols
  const outPlane = outRows * outCols

  for (let channel = 0; channel < channels; channel += 1) {
    const inOffset = channel * inPlane
    const outOffset = channel * outPlane
    for (let row = 0; row < outRows; row += 1) {
      for (let col = 0; col < outCols; col += 1) {
        let best = Number.NEGATIVE_INFINITY
        for (let kr = 0; kr < kernel; kr += 1) {
          for (let kc = 0; kc < kernel; kc += 1) {
            const value =
              input[inOffset + (row * stride + kr) * cols + col * stride + kc]
            if (value > best) best = value
          }
        }
        out[outOffset + row * outCols + col] = best
      }
    }
  }

  return { data: out, rows: outRows, cols: outCols }
}

function softmax(logits: Float32Array): Float32Array {
  let max = Number.NEGATIVE_INFINITY
  for (const value of logits) if (value > max) max = value
  const out = new Float32Array(logits.length)
  let total = 0
  for (let index = 0; index < logits.length; index += 1) {
    const value = Math.exp(logits[index] - max)
    out[index] = value
    total += value
  }
  for (let index = 0; index < out.length; index += 1) out[index] /= total
  return out
}

function mapStage(
  id: string,
  name: string,
  kind: StageKind,
  data: Float32Array,
  channels: number,
  rows: number,
  cols: number,
  params: number,
  spec?: TraceableSpec
): ClassifierStage {
  const maps = Array.from({ length: channels }, (_, channel) =>
    mapFrom(
      data.subarray(channel * rows * cols, (channel + 1) * rows * cols),
      rows,
      cols
    )
  )
  return { id, name, kind, maps, rows, cols, channels, params, stats: stageStats(maps), spec }
}

/** Full forward pass of the pretrained model with every activation kept. */
export function classify(model: MnistModel, input: Float32Array): ClassifierRun {
  const stages: ClassifierStage[] = []
  const size = MNIST_SIZE

  stages.push(
    mapStage("input", "Input", "input", input, 1, size, size, 0, MODEL_SPECS[0])
  )

  const z1 = convolve(input, model.conv1.inChannels, size, size, model.conv1)
  stages.push(
    mapStage("conv1", "Conv 1", "conv", z1, model.conv1.outChannels, size, size,
      model.conv1.outChannels * (model.conv1.inChannels * model.conv1.kernel ** 2 + 1),
      MODEL_SPECS[1])
  )

  const a1 = relu(z1)
  stages.push(
    mapStage("relu1", "ReLU 1", "relu", a1, model.conv1.outChannels, size, size, 0, MODEL_SPECS[2])
  )

  const p1 = maxPool(a1, model.conv1.outChannels, size, size, 2, 2)
  stages.push(
    mapStage("pool1", "MaxPool 1", "pool", p1.data, model.conv1.outChannels, p1.rows, p1.cols, 0, MODEL_SPECS[3])
  )

  const z2 = convolve(p1.data, model.conv2.inChannels, p1.rows, p1.cols, model.conv2)
  stages.push(
    mapStage("conv2", "Conv 2", "conv", z2, model.conv2.outChannels, p1.rows, p1.cols,
      model.conv2.outChannels * (model.conv2.inChannels * model.conv2.kernel ** 2 + 1),
      MODEL_SPECS[4])
  )

  const a2 = relu(z2)
  stages.push(
    mapStage("relu2", "ReLU 2", "relu", a2, model.conv2.outChannels, p1.rows, p1.cols, 0, MODEL_SPECS[5])
  )

  const p2 = maxPool(a2, model.conv2.outChannels, p1.rows, p1.cols, 3, 3)
  stages.push(
    mapStage("pool2", "MaxPool 2", "pool", p2.data, model.conv2.outChannels, p2.rows, p2.cols, 0, MODEL_SPECS[6])
  )

  stages.push({
    id: "flatten",
    name: "Dense",
    kind: "flatten",
    maps: [mapFrom(p2.data, 16, 16)],
    rows: 16,
    cols: 16,
    channels: 1,
    params: 0,
    stats: stageStats([mapFrom(p2.data, 16, 16)]),
  })

  const logits = new Float32Array(model.dense.units)
  for (let unit = 0; unit < model.dense.units; unit += 1) {
    let sum = model.dense.bias[unit]
    for (let index = 0; index < model.dense.inputs; index += 1) {
      sum += p2.data[index] * model.dense.weights[index * model.dense.units + unit]
    }
    logits[unit] = sum
  }
  const probabilities = softmax(logits)
  let predicted = 0
  for (let unit = 1; unit < probabilities.length; unit += 1) {
    if (probabilities[unit] > probabilities[predicted]) predicted = unit
  }

  stages.push({
    id: "dense",
    name: "Classification",
    kind: "dense",
    maps: [],
    rows: 1,
    cols: MNIST_CLASSES,
    channels: MNIST_CLASSES,
    params: model.dense.inputs * model.dense.units + model.dense.units,
    stats: stageStats([mapFrom(logits, 1, MNIST_CLASSES)]),
  })

  return { stages, logits, probabilities, predicted }
}

/** Dense weight column for one class, reshaped to the pooled 16×4×4 grid. */
export function classTemplate(
  model: MnistModel,
  unit: number
): FloatMap[] {
  const maps: FloatMap[] = []
  for (let channel = 0; channel < 16; channel += 1) {
    const data = new Float32Array(16)
    for (let index = 0; index < 16; index += 1) {
      data[index] = model.dense.weights[(channel * 16 + index) * 10 + unit]
    }
    maps.push({ data, rows: 4, cols: 4 })
  }
  return maps
}

export type RfTrace = (RfRegion | null)[]

/** Receptive-field regions for one cell of a map stage (0…6). */
export function traceCell(
  stages: { spec?: TraceableSpec }[],
  stageIndex: number,
  channel: number,
  row: number,
  col: number
): RfTrace {
  const specs = stages
    .slice(0, TRACE_STAGE_COUNT)
    .map((stage) => stage.spec)
    .filter((spec): spec is TraceableSpec => spec !== undefined)
  if (stageIndex >= specs.length) return specs.map(() => null)
  return traceReceptiveField(
    specs.map((spec) => ({ spec })),
    stageIndex,
    channel,
    row,
    col
  )
}

/** Sample every nth test digit so the strip shows one of each class. */
export function pickStarterSamples(samples: SampleDigit[]): SampleDigit[] {
  const byClass = new Map<number, SampleDigit>()
  for (const sample of samples) {
    if (!byClass.has(sample.label)) byClass.set(sample.label, sample)
  }
  return Array.from({ length: MNIST_CLASSES }, (_, digit) => byClass.get(digit)).filter(
    (sample): sample is SampleDigit => sample !== undefined
  )
}

/**
 * Downsample the drawing pad to a 28×28 grid. With `centered` the ink is
 * scaled to 20×20 and shifted so its centre of mass sits at (14, 14) — the
 * same normalisation the MNIST dataset uses, which is what the pretrained
 * weights were fitted on.
 */
export function gridFromCanvas(
  source: HTMLCanvasElement,
  size: number,
  centered: boolean
): Float32Array {
  const out = new Float32Array(size * size)
  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext("2d", { willReadFrequently: true })
  if (!context) return out

  context.fillStyle = "#000"
  context.fillRect(0, 0, size, size)
  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = "high"

  if (!centered) {
    context.drawImage(source, 0, 0, size, size)
  } else {
    const box = inkBox(source)
    if (!box) {
      context.drawImage(source, 0, 0, size, size)
    } else {
      const side = Math.max(box.width, box.height)
      const scale = (size * (20 / 28)) / Math.max(1, side)
      context.save()
      context.translate(size / 2, size / 2)
      context.scale(scale, scale)
      context.translate(-(box.left + box.right) / 2, -(box.top + box.bottom) / 2)
      context.drawImage(source, 0, 0)
      context.restore()
    }
  }

  const image = context.getImageData(0, 0, size, size)
  for (let index = 0; index < size * size; index += 1) {
    const offset = index * 4
    out[index] =
      Math.max(image.data[offset], image.data[offset + 1], image.data[offset + 2]) / 255
  }
  return out
}

export function inkBox(source: HTMLCanvasElement): {
  left: number
  top: number
  right: number
  bottom: number
  width: number
  height: number
} | null {
  const context = source.getContext("2d", { willReadFrequently: true })
  if (!context) return null
  const { data } = context.getImageData(0, 0, source.width, source.height)
  let minX = source.width
  let minY = source.height
  let maxX = -1
  let maxY = -1

  for (let y = 0; y < source.height; y += 1) {
    for (let x = 0; x < source.width; x += 1) {
      const offset = (y * source.width + x) * 4
      const luminance =
        (data[offset] + data[offset + 1] + data[offset + 2]) / 3 / 255
      if (luminance > 0.1) {
        if (x < minX) minX = x
        if (y < minY) minY = y
        if (x > maxX) maxX = x
        if (y > maxY) maxY = y
      }
    }
  }

  if (maxX < 0) return null
  return {
    left: minX,
    top: minY,
    right: maxX + 1,
    bottom: maxY + 1,
    width: maxX + 1 - minX,
    height: maxY + 1 - minY,
  }
}

/**
 * Compares the browser forward pass of one sample against the reference
 * values computed with NumPy/onnxruntime when the weights were exported.
 */
export function verifySample(
  model: MnistModel,
  sample: SampleDigit
): { maxLogitDiff: number; conv1Diff: number; relu1Diff: number; pool2Diff: number } {
  const run = classify(model, sampleGrid(sample))
  let maxLogitDiff = 0
  for (let unit = 0; unit < run.logits.length; unit += 1) {
    maxLogitDiff = Math.max(
      maxLogitDiff,
      Math.abs(run.logits[unit] - sample.check.logits[unit])
    )
  }
  const conv1 = run.stages[1]
  const relu1 = run.stages[2]
  const pool2 = run.stages[6]
  let conv1Sum = 0
  for (const map of conv1.maps) for (const value of map.data) conv1Sum += value
  let pool2Sum = 0
  for (const map of pool2.maps) for (const value of map.data) pool2Sum += value
  return {
    maxLogitDiff,
    conv1Diff: Math.abs(conv1Sum - sample.check.conv1Sum),
    relu1Diff: Math.abs(relu1.stats.zeroFraction - sample.check.relu1Zeros),
    pool2Diff: Math.abs(pool2Sum - sample.check.pool2Sum),
  }
}
