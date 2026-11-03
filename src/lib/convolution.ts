export type Matrix = number[][]

export type PaddingMode = "valid" | "same" | "zero"

export type ConvolutionOptions = {
  stride?: number
  padding?: PaddingMode
  /** Explicit zero-pad width; only used when padding === "zero". */
  pad?: number
  bias?: number
  /** Rotate the kernel 180° before sliding (true convolution). */
  flip?: boolean
}

export type PoolingMode = "max" | "avg"

/** Fields shared by sliding-window operations (convolution and pooling). */
export type FeatureMapResult = {
  rows: number
  cols: number
  /**
   * Convolution: per input-channel contribution before the bias.
   * Pooling: per input-channel downsampled map (channels stay separate).
   */
  perChannel: Matrix[]
  pad: number
  stride: number
  kernelSize: number
  valid: boolean
}

export type ConvolutionResult = FeatureMapResult & {
  /** Channel-summed output, bias included. */
  values: Matrix
}

export type PoolingResult = FeatureMapResult & {
  mode: PoolingMode
}

export const KERNEL_SIZES = [1, 3, 5] as const

export const KERNEL_PRESETS: { value: string; label: string }[] = [
  { value: "identity", label: "Identity" },
  { value: "box", label: "Box blur" },
  { value: "gaussian", label: "Gaussian blur" },
  { value: "sharpen", label: "Sharpen" },
  { value: "sobel-x", label: "Sobel-X (vertical edges)" },
  { value: "sobel-y", label: "Sobel-Y (horizontal edges)" },
  { value: "laplacian", label: "Laplacian" },
  { value: "emboss", label: "Emboss" },
  { value: "outline", label: "Outline" },
]

export const clampNumber = (value: number, min: number, max: number) =>
  Math.min(max, Math.max(min, value))

export function zeros(rows: number, cols: number, fill = 0): Matrix {
  return Array.from({ length: rows }, () => new Array<number>(cols).fill(fill))
}

function grid(size: number, fill = 0): Matrix {
  return zeros(size, size, fill)
}

function identityKernel(size: number): Matrix {
  const kernel = grid(size)
  const center = Math.floor(size / 2)
  kernel[center][center] = 1
  return kernel
}

function meanKernel(size: number): Matrix {
  return grid(size, 1 / (size * size))
}

/** Binomial approximation of a Gaussian, normalised so the weights sum to 1. */
function gaussianKernel(size: number): Matrix {
  const coefficients = Array.from({ length: size }, (_, index) => {
    let value = 1
    for (let step = 1; step <= index; step += 1) {
      value = (value * (size - step)) / step
    }
    return value
  })
  const rowTotal = coefficients.reduce((sum, value) => sum + value, 0)
  return coefficients.map((rowValue) =>
    coefficients.map((columnValue) => (rowValue * columnValue) / (rowTotal * rowTotal))
  )
}

const FIXED_KERNELS: Record<string, Record<number, Matrix>> = {
  sharpen: {
    3: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0],
    ],
    5: [
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, 25, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
    ],
  },
  "sobel-x": {
    3: [
      [-1, 0, 1],
      [-2, 0, 2],
      [-1, 0, 1],
    ],
    5: [
      [-1, -2, 0, 2, 1],
      [-4, -8, 0, 8, 4],
      [-6, -12, 0, 12, 6],
      [-4, -8, 0, 8, 4],
      [-1, -2, 0, 2, 1],
    ],
  },
  "sobel-y": {
    3: [
      [-1, -2, -1],
      [0, 0, 0],
      [1, 2, 1],
    ],
    5: [
      [-1, -4, -6, -4, -1],
      [-2, -8, -12, -8, -2],
      [0, 0, 0, 0, 0],
      [2, 8, 12, 8, 2],
      [1, 4, 6, 4, 1],
    ],
  },
  laplacian: {
    3: [
      [0, 1, 0],
      [1, -4, 1],
      [0, 1, 0],
    ],
    5: [
      [0, 0, 1, 0, 0],
      [0, 1, 2, 1, 0],
      [1, 2, -16, 2, 1],
      [0, 1, 2, 1, 0],
      [0, 0, 1, 0, 0],
    ],
  },
  emboss: {
    3: [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2],
    ],
    5: [
      [-1, -1, -1, -1, 0],
      [-1, -1, -1, 0, 1],
      [-1, -1, 1, 1, 1],
      [-1, 0, 1, 1, 1],
      [0, 1, 1, 1, 1],
    ],
  },
  outline: {
    3: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1],
    ],
    5: [
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, 24, -1, -1],
      [-1, -1, -1, -1, -1],
      [-1, -1, -1, -1, -1],
    ],
  },
}

export function kernelFromPreset(preset: string, size: number): Matrix {
  if (preset === "identity") return identityKernel(size)
  if (preset === "box") return meanKernel(size)
  if (preset === "gaussian") return gaussianKernel(size)
  if (size === 1) return identityKernel(1)
  return FIXED_KERNELS[preset]?.[size] ?? identityKernel(size)
}

export function rotate180(kernel: Matrix): Matrix {
  const size = kernel.length
  return Array.from({ length: size }, (_, row) =>
    Array.from(
      { length: size },
      (_, column) => kernel[size - 1 - row][size - 1 - column]
    )
  )
}

export function effectiveKernel(kernel: Matrix, flip: boolean): Matrix {
  return flip ? rotate180(kernel) : kernel
}

/** Padding width for a mode; "same" assumes an odd kernel in v1. */
export function paddingFor(
  mode: PaddingMode,
  kernelSize: number,
  pad = 1
): number {
  if (mode === "same") return Math.floor(kernelSize / 2)
  if (mode === "zero") return Math.max(0, Math.trunc(pad))
  return 0
}

/** H_out = floor((H_in + 2p − k) / s) + 1. */
export function outputDimension(
  inputSize: number,
  kernelSize: number,
  stride: number,
  pad: number
): number {
  return Math.floor((inputSize + 2 * pad - kernelSize) / stride) + 1
}

/** Read with zero padding: anything off the grid counts as 0. */
export function readZero(matrix: Matrix, row: number, column: number): number {
  if (row < 0 || column < 0) return 0
  return matrix[row]?.[column] ?? 0
}

/** The k×k patch of input that produces output cell (i, j), zero-padded. */
export function extractWindow(
  input: Matrix,
  i: number,
  j: number,
  kernelSize: number,
  stride: number,
  pad: number
): Matrix {
  return Array.from({ length: kernelSize }, (_, m) =>
    Array.from({ length: kernelSize }, (_, n) =>
      readZero(input, i * stride + m - pad, j * stride + n - pad)
    )
  )
}

/**
 * Cross-correlation / convolution of a channel stack with one kernel per
 * channel. Contributions are summed across channels, then the bias is added
 * once to produce a single output channel.
 */
export function convolveChannels(
  channels: number[][][],
  kernels: number[][][],
  options: ConvolutionOptions = {}
): ConvolutionResult {
  const stride = Math.max(1, Math.trunc(options.stride ?? 1))
  const padding = options.padding ?? "valid"
  const kernelSize = kernels[0]?.length ?? 0
  const pad = paddingFor(padding, kernelSize, options.pad)
  const bias = options.bias ?? 0
  const flip = options.flip ?? false

  const height = channels[0]?.length ?? 0
  const width = channels[0]?.[0]?.length ?? 0
  const rows =
    height > 0 && kernelSize > 0
      ? outputDimension(height, kernelSize, stride, pad)
      : 0
  const cols =
    width > 0 && kernelSize > 0
      ? outputDimension(width, kernelSize, stride, pad)
      : 0

  const perChannel: Matrix[] = channels.map(() => [])
  const valid =
    rows > 0 && cols > 0 && channels.length === kernels.length && kernelSize > 0

  if (!valid) {
    return {
      rows: Math.max(0, rows),
      cols: Math.max(0, cols),
      values: [],
      perChannel,
      pad,
      stride,
      kernelSize,
      valid: false,
    }
  }

  const effective = kernels.map((kernel) => effectiveKernel(kernel, flip))
  const values = zeros(rows, cols)

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const channel = channels[channelIndex]
    const kernel = effective[channelIndex]
    const contribution = zeros(rows, cols)

    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        let sum = 0
        for (let m = 0; m < kernelSize; m += 1) {
          for (let n = 0; n < kernelSize; n += 1) {
            sum +=
              readZero(channel, i * stride + m - pad, j * stride + n - pad) *
              (kernel[m]?.[n] ?? 0)
          }
        }
        contribution[i][j] = sum
        values[i][j] += sum
      }
    }

    perChannel[channelIndex] = contribution
  }

  for (let i = 0; i < rows; i += 1) {
    for (let j = 0; j < cols; j += 1) {
      values[i][j] += bias
    }
  }

  return { rows, cols, values, perChannel, pad, stride, kernelSize, valid: true }
}

export function convolve2d(
  input: Matrix,
  kernel: Matrix,
  options: ConvolutionOptions = {}
): ConvolutionResult {
  return convolveChannels([input], [kernel], options)
}

/** Sum of the kernel weights — 0 means the filter ignores flat regions. */
export function kernelSum(kernel: Matrix): number {
  return kernel.reduce(
    (total, row) => total + row.reduce((sum, value) => sum + value, 0),
    0
  )
}

/**
 * Max / average pooling over a channel stack, one map per channel.
 * Padded zeros take part in both the max and the average (PyTorch's default
 * `count_include_pad=True`), so padding always has a visible effect.
 */
export function poolChannels(
  channels: number[][][],
  windowSize: number,
  options: {
    stride?: number
    padding?: PaddingMode
    pad?: number
    mode?: PoolingMode
  } = {}
): PoolingResult {
  const stride = Math.max(1, Math.trunc(options.stride ?? 1))
  const padding = options.padding ?? "valid"
  const pad = paddingFor(padding, windowSize, options.pad)
  const mode = options.mode ?? "max"

  const height = channels[0]?.length ?? 0
  const width = channels[0]?.[0]?.length ?? 0
  const rows =
    height > 0 && windowSize > 0
      ? outputDimension(height, windowSize, stride, pad)
      : 0
  const cols =
    width > 0 && windowSize > 0
      ? outputDimension(width, windowSize, stride, pad)
      : 0

  const perChannel: Matrix[] = channels.map(() => [])
  const valid = rows > 0 && cols > 0 && windowSize > 0 && channels.length > 0

  if (!valid) {
    return {
      rows: Math.max(0, rows),
      cols: Math.max(0, cols),
      perChannel,
      pad,
      stride,
      kernelSize: windowSize,
      valid: false,
      mode,
    }
  }

  for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
    const channel = channels[channelIndex]
    const pooled = zeros(rows, cols)

    for (let i = 0; i < rows; i += 1) {
      for (let j = 0; j < cols; j += 1) {
        let sum = 0
        let best = Number.NEGATIVE_INFINITY
        for (let m = 0; m < windowSize; m += 1) {
          for (let n = 0; n < windowSize; n += 1) {
            const value = readZero(
              channel,
              i * stride + m - pad,
              j * stride + n - pad
            )
            sum += value
            if (value > best) best = value
          }
        }
        pooled[i][j] =
          mode === "max" ? best : sum / (windowSize * windowSize)
      }
    }

    perChannel[channelIndex] = pooled
  }

  return { rows, cols, perChannel, pad, stride, kernelSize: windowSize, valid: true, mode }
}

/**
 * Which lanes (rows or columns) of a padded grid are touched by at least one
 * window position. `false` lanes are pixels the sliding window never reads —
 * the visual proof that stride skips input.
 */
export function windowCoverage(
  count: number,
  kernelSize: number,
  stride: number,
  outputs: number
): boolean[] {
  const covered = new Array<boolean>(Math.max(0, count)).fill(false)
  for (let position = 0; position < outputs; position += 1) {
    for (let offset = 0; offset < kernelSize; offset += 1) {
      const index = position * stride + offset
      if (index >= 0 && index < covered.length) covered[index] = true
    }
  }
  return covered
}
