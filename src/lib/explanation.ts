import type {
  ClassifierRun,
  ClassifierStage,
  FloatMap,
  MnistModel,
} from "./mnist-cnn"

/**
 * The guided "explain this prediction" forward pass, expressed as a small,
 * framework-free state machine plus helpers that read the *real* activations of
 * a `ClassifierRun`. The React layer (a `useExplanation` hook and the guided
 * overlays) only orchestrates: every number here comes from the pretrained
 * model's current forward pass.
 */

export type ExplanationStage =
  | "input"
  | "conv1"
  | "relu1"
  | "pool1"
  | "conv2"
  | "relu2"
  | "pool2"
  | "dense"
  | "prediction"

/** One beat of the guided story. `layer` indexes the run's stages and the board. */
export type ExplanationStep<Id extends string = ExplanationStage> = {
  id: Id
  title: string
  /** One sentence shown in the floating annotation. */
  caption: string
  layer: number
  /** How long the beat plays before the controller auto-advances. */
  durationMs: number
}

/**
 * Nine beats, ~21 s total: long enough to read each sentence, short enough to
 * stay a story. Durations stay inside the 1.5–3 s per-stage budget.
 */
export const EXPLANATION_STEPS: readonly ExplanationStep[] = [
  {
    id: "input",
    title: "Input",
    caption: "The network receives your drawing as a 28×28 grayscale image.",
    layer: 0,
    durationMs: 1900,
  },
  {
    id: "conv1",
    title: "Conv 1",
    caption:
      "These learned filters respond to local stroke patterns in your drawing.",
    layer: 1,
    durationMs: 2600,
  },
  {
    id: "relu1",
    title: "ReLU 1",
    caption: "ReLU suppresses negative responses and keeps positive activations.",
    layer: 2,
    durationMs: 2300,
  },
  {
    id: "pool1",
    title: "MaxPool 1",
    caption:
      "Max pooling keeps the strongest local response while reducing spatial resolution.",
    layer: 3,
    durationMs: 3000,
  },
  {
    id: "conv2",
    title: "Conv 2",
    caption:
      "The second convolution combines lower-level responses into more complex patterns.",
    layer: 4,
    durationMs: 2600,
  },
  {
    id: "relu2",
    title: "ReLU 2",
    caption: "ReLU again keeps the positive responses and drops the rest.",
    layer: 5,
    durationMs: 1600,
  },
  {
    id: "pool2",
    title: "MaxPool 2",
    caption: "The last pooling step keeps the strongest responses in a 4×4 grid.",
    layer: 6,
    durationMs: 1900,
  },
  {
    id: "dense",
    title: "Dense",
    caption: "The dense layer combines the extracted features into class scores.",
    layer: 8,
    durationMs: 2600,
  },
  {
    id: "prediction",
    title: "Prediction",
    caption: "Softmax turns the class scores into probabilities.",
    layer: 8,
    durationMs: 2700,
  },
]

export type ExplanationState = {
  active: boolean
  index: number
  playing: boolean
  /** 0…1 through the current beat. */
  progress: number
}

export function createExplanationState(): ExplanationState {
  return { active: false, index: 0, playing: false, progress: 0 }
}

export type ExplanationEvent =
  | { type: "start" }
  | { type: "exit" }
  | { type: "next" }
  | { type: "previous" }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "restart" }
  | { type: "select"; index: number }
  | { type: "tick"; deltaMs: number }

function clampIndex<Id extends string>(
  steps: readonly ExplanationStep<Id>[],
  index: number
): number {
  if (steps.length === 0) return 0
  return Math.min(steps.length - 1, Math.max(0, Math.trunc(index)))
}

/** Pure transition function — the whole controller is testable without React. */
export function explanationReducer<Id extends string = ExplanationStage>(
  steps: readonly ExplanationStep<Id>[],
  state: ExplanationState,
  event: ExplanationEvent
): ExplanationState {
  const last = Math.max(0, steps.length - 1)

  switch (event.type) {
    case "start":
      return { active: true, index: 0, playing: steps.length > 1, progress: 0 }
    case "exit":
      return createExplanationState()
    case "next": {
      if (!state.active) return state
      const index = clampIndex(steps, state.index + 1)
      return {
        ...state,
        index,
        playing: index < last ? state.playing : false,
        progress: 0,
      }
    }
    case "previous": {
      if (!state.active) return state
      return { ...state, index: clampIndex(steps, state.index - 1), progress: 0 }
    }
    case "pause":
      return state.active && state.playing ? { ...state, playing: false } : state
    case "resume": {
      if (!state.active || state.playing || state.index >= last) return state
      return { ...state, playing: true }
    }
    case "restart":
      return { active: true, index: 0, playing: steps.length > 1, progress: 0 }
    case "select": {
      if (!state.active) return state
      return {
        ...state,
        index: clampIndex(steps, event.index),
        playing: false,
        progress: 0,
      }
    }
    case "tick": {
      if (!state.active || !state.playing || event.deltaMs <= 0) return state
      const step = steps[state.index]
      if (!step) return state
      const duration = Math.max(1, step.durationMs)
      const progress = state.progress + event.deltaMs / duration
      if (progress < 1) return { ...state, progress }
      if (state.index >= last) return { ...state, progress: 1, playing: false }
      return { ...state, index: state.index + 1, progress: 0 }
    }
    default:
      return state
  }
}

export function currentExplanationStep<Id extends string>(
  steps: readonly ExplanationStep<Id>[],
  state: ExplanationState
): ExplanationStep<Id> | undefined {
  return steps[clampIndex(steps, state.index)]
}

/* ------------------------------------------------------------------ *
 * Real-activation helpers: everything below reads a ClassifierRun.
 * ------------------------------------------------------------------ */

/** Sum of squares — how much a map reacted to this input. */
export function channelEnergy(map: FloatMap): number {
  let energy = 0
  for (let index = 0; index < map.data.length; index += 1) {
    const value = map.data[index]
    energy += value * value
  }
  return energy
}

/** Channels ranked by activation energy, strongest first. */
export function strongestChannels(stage: ClassifierStage, count = 3): number[] {
  return stage.maps
    .map((map, channel) => ({ channel, energy: channelEnergy(map) }))
    .sort((a, b) => b.energy - a.energy)
    .slice(0, Math.max(0, count))
    .map((entry) => entry.channel)
}

export type ReluTransition = {
  /** The stage the gate saw (pre-activation, e.g. Conv 1). */
  pre: ClassifierStage
  post: ClassifierStage
  /** Representative channel: the strongest post-gate map. */
  channel: number
  /** Fraction of the volume the gate zeroed. */
  zeroFraction: number
}

export function reluTransition(
  run: ClassifierRun,
  stageIndex: number
): ReluTransition | null {
  const post = run.stages[stageIndex]
  const pre = run.stages[stageIndex - 1]
  if (!post || !pre || post.kind !== "relu") return null
  return {
    pre,
    post,
    channel: strongestChannels(post, 1)[0] ?? 0,
    zeroFraction: post.stats.zeroFraction,
  }
}

export type PoolSample = {
  /** Post-ReLU stage the window reads from. */
  input: ClassifierStage
  /** Pooled stage the window writes to. */
  output: ClassifierStage
  channel: number
  window: number
  stride: number
  /** Window origin on the input map (cells). */
  row: number
  col: number
  /** Winning cell on the pooled map. */
  outputRow: number
  outputCol: number
  max: number
}

/**
 * The pooling window that produced a stage's largest pooled value — a
 * representative region that genuinely "won" for this input.
 */
export function poolSample(
  run: ClassifierRun,
  stageIndex: number,
  channel: number
): PoolSample | null {
  const output = run.stages[stageIndex]
  const input = run.stages[stageIndex - 1]
  if (!output || !input || output.kind !== "pool") return null
  const spec = output.spec
  if (!spec || spec.kind !== "pool") return null
  const map = output.maps[channel]
  if (!map || map.data.length === 0) return null

  let best = 0
  for (let index = 1; index < map.data.length; index += 1) {
    if (map.data[index] > map.data[best]) best = index
  }
  const outputRow = Math.floor(best / map.cols)
  const outputCol = best % map.cols
  return {
    input,
    output,
    channel,
    window: spec.window,
    stride: spec.stride,
    row: outputRow * spec.stride - spec.pad,
    col: outputCol * spec.stride - spec.pad,
    outputRow,
    outputCol,
    max: map.data[best],
  }
}

export type Contribution = {
  /** Flat index into the 256-value flattened volume (channels first). */
  index: number
  channel: number
  row: number
  col: number
  /** activation × dense weight for the selected class. */
  value: number
}

export type DenseContributions = {
  unit: number
  /** Most positive feature contributions to the class score. */
  positive: Contribution[]
  /** Most negative feature contributions to the class score. */
  negative: Contribution[]
  /** Σ contributions = logit − bias. */
  total: number
}

/**
 * Feature-level contributions to one class score: each pooled activation times
 * the dense weight wired to that class. This is the same arithmetic the dense
 * inspector shows, just ranked.
 */
export function denseContributions(
  model: MnistModel,
  run: ClassifierRun,
  unit: number,
  count = 3
): DenseContributions | null {
  const pooled = run.stages[6]
  if (!pooled || unit < 0 || unit >= model.dense.units) return null

  const contributions: Contribution[] = []
  let total = 0
  for (let channel = 0; channel < pooled.maps.length; channel += 1) {
    const map = pooled.maps[channel]
    for (let cell = 0; cell < map.data.length; cell += 1) {
      const index = channel * map.data.length + cell
      const value =
        map.data[cell] * model.dense.weights[index * model.dense.units + unit]
      total += value
      contributions.push({
        index,
        channel,
        row: Math.floor(cell / map.cols),
        col: cell % map.cols,
        value,
      })
    }
  }

  return {
    unit,
    positive: contributions
      .filter((entry) => entry.value > 0)
      .sort((a, b) => b.value - a.value)
      .slice(0, Math.max(0, count)),
    negative: contributions
      .filter((entry) => entry.value < 0)
      .sort((a, b) => a.value - b.value)
      .slice(0, Math.max(0, count)),
    total,
  }
}

export type RankedClass = { unit: number; probability: number; logit: number }

export function probabilityRanking(run: ClassifierRun): RankedClass[] {
  return Array.from(run.probabilities, (probability, unit) => ({
    unit,
    probability,
    logit: run.logits[unit],
  })).sort((a, b) => b.probability - a.probability)
}

/** Everything the guided UI needs for one beat, read from the current run. */
export type GuideData = {
  /** Strongest channels of the focused stage (empty when it has no maps). */
  strong: number[]
  /** Channel the camera and highlighters should follow. */
  channel: number
  relu: ReluTransition | null
  pool: PoolSample | null
  dense: DenseContributions | null
}

export function buildGuideData(
  model: MnistModel,
  run: ClassifierRun,
  step: ExplanationStep
): GuideData {
  const stage = run.stages[step.layer]
  const strong = stage && stage.maps.length > 0 ? strongestChannels(stage, 3) : []
  const channel = strong[0] ?? 0

  return {
    strong,
    channel,
    relu:
      step.id === "relu1" || step.id === "relu2"
        ? reluTransition(run, step.layer)
        : null,
    pool:
      step.id === "pool1" || step.id === "pool2"
        ? poolSample(run, step.layer, channel)
        : null,
    dense:
      step.id === "dense" || step.id === "prediction"
        ? denseContributions(model, run, run.predicted, 3)
        : null,
  }
}
