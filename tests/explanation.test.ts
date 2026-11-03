import assert from "node:assert/strict"
import { test } from "node:test"

import {
  EXPLANATION_STEPS,
  buildGuideData,
  createExplanationState,
  currentExplanationStep,
  denseContributions,
  explanationReducer,
  poolSample,
  probabilityRanking,
  reluTransition,
  strongestChannels,
  type ExplanationEvent,
  type ExplanationState,
  type ExplanationStep,
} from "../src/lib/explanation.ts"

import type {
  ClassifierRun,
  ClassifierStage,
  FloatMap,
  MnistConvLayer,
  MnistModel,
} from "../src/lib/mnist-cnn.ts"

/* ---------------------------------------------------------------- *
 * Fixtures: tiny real-shaped stages so the helpers are exercised
 * against actual FloatMap math, not mocks.
 * ---------------------------------------------------------------- */

function map(data: number[], rows: number, cols: number): FloatMap {
  return { data: Float32Array.from(data), rows, cols }
}

function stage(
  id: string,
  kind: ClassifierStage["kind"],
  maps: FloatMap[],
  spec?: ClassifierStage["spec"]
): ClassifierStage {
  let min = Number.POSITIVE_INFINITY
  let max = Number.NEGATIVE_INFINITY
  let sum = 0
  let zeros = 0
  let count = 0
  for (const entry of maps) {
    for (const value of entry.data) {
      if (value < min) min = value
      if (value > max) max = value
      sum += value
      if (value === 0) zeros += 1
      count += 1
    }
  }
  const first = maps[0]
  return {
    id,
    name: id,
    kind,
    maps,
    rows: first?.rows ?? 0,
    cols: first?.cols ?? 0,
    channels: maps.length,
    params: 0,
    stats: {
      min: count > 0 ? min : 0,
      max: count > 0 ? max : 0,
      mean: count > 0 ? sum / count : 0,
      zeroFraction: count > 0 ? zeros / count : 0,
      bound: count > 0 ? Math.max(1e-6, Math.abs(min), Math.abs(max)) : 1,
      perChannel: maps.map((entry) => ({
        min: entry.data.length > 0 ? Math.min(...entry.data) : 0,
        max: entry.data.length > 0 ? Math.max(...entry.data) : 0,
        mean:
          entry.data.length > 0
            ? entry.data.reduce((total, value) => total + value, 0) / entry.data.length
            : 0,
        zeros: Array.from(entry.data).filter((value) => value === 0).length,
        count: entry.data.length,
      })),
    },
    spec,
  }
}

function convLayer(outChannels: number, inChannels: number): MnistConvLayer {
  return {
    weights: new Float32Array(outChannels * inChannels * 25),
    bias: new Float32Array(outChannels),
    kernel: 5,
    pad: 2,
    inChannels,
    outChannels,
  }
}

function makeModel(): MnistModel {
  const weights = new Float32Array(256 * 10)
  for (let index = 0; index < 256; index += 1) {
    weights[index * 10 + 7] = 0.01
  }
  return {
    source: "test model",
    testAccuracy: 1,
    testCount: 1,
    conv1: convLayer(1, 1),
    conv2: convLayer(1, 1),
    dense: {
      weights,
      bias: new Float32Array(10),
      inputs: 256,
      units: 10,
    },
    params: 0,
  }
}

function makeRun(): ClassifierRun {
  const probabilities = new Float32Array(10)
  probabilities[7] = 0.9
  const logits = new Float32Array(10)
  logits[7] = 2
  return {
    stages: [
      stage("input", "input", [map(new Array(784).fill(0.5), 28, 28)]),
      stage("conv1", "conv", [map(new Array(784).fill(1), 28, 28)]),
      stage("relu1", "relu", [map(new Array(784).fill(0.5), 28, 28)]),
      stage("pool1", "pool", [map(new Array(196).fill(1), 14, 14)], {
        kind: "pool",
        window: 2,
        stride: 2,
        pad: 0,
      }),
      stage("conv2", "conv", [map(new Array(196).fill(1), 14, 14)]),
      stage("relu2", "relu", [map(new Array(196).fill(0.5), 14, 14)]),
      stage(
        "pool2",
        "pool",
        Array.from({ length: 16 }, () => map(new Array(16).fill(1), 4, 4)),
        { kind: "pool", window: 3, stride: 3, pad: 0 }
      ),
      stage("flatten", "flatten", [map(new Array(256).fill(1), 16, 16)]),
      stage("dense", "dense", []),
    ],
    logits,
    probabilities,
    predicted: 7,
  }
}

const STEPS: readonly ExplanationStep[] = EXPLANATION_STEPS

function play(
  state: ExplanationState,
  ...events: ExplanationEvent[]
): ExplanationState {
  return events.reduce(
    (current, event) => explanationReducer(STEPS, current, event),
    state
  )
}

/* ---------------------------------------------------------------- *
 * Stage sequence contract
 * ---------------------------------------------------------------- */

test("the guided story has nine ordered beats inside the time budget", () => {
  assert.deepEqual(
    STEPS.map((step) => step.id),
    ["input", "conv1", "relu1", "pool1", "conv2", "relu2", "pool2", "dense", "prediction"]
  )
  const total = STEPS.reduce((sum, step) => sum + step.durationMs, 0)
  assert.ok(total >= 15000, `sequence is too short: ${total} ms`)
  assert.ok(total <= 25000, `sequence is too long: ${total} ms`)
  for (const step of STEPS) {
    assert.ok(step.durationMs >= 1500 && step.durationMs <= 3000, step.id)
    assert.ok(step.layer >= 0 && step.layer <= 8, step.id)
    assert.ok(step.title.length > 0 && step.caption.length > 0, step.id)
  }
})

/* ---------------------------------------------------------------- *
 * Controller behaviour
 * ---------------------------------------------------------------- */

test("start begins at the first beat and plays", () => {
  assert.deepEqual(createExplanationState(), {
    active: false,
    index: 0,
    playing: false,
    progress: 0,
  })
  assert.deepEqual(play(createExplanationState(), { type: "start" }), {
    active: true,
    index: 0,
    playing: true,
    progress: 0,
  })
})

test("ticks accumulate progress and advance after the beat duration", () => {
  const duration = STEPS[0].durationMs
  const started = play(createExplanationState(), { type: "start" })
  const half = play(started, { type: "tick", deltaMs: duration / 2 })
  assert.equal(half.index, 0)
  assert.ok(Math.abs(half.progress - 0.5) < 1e-9)
  const advanced = play(half, { type: "tick", deltaMs: duration / 2 + 1 })
  assert.equal(advanced.index, 1)
  assert.equal(advanced.progress, 0)
  assert.equal(advanced.playing, true)
})

test("the story stops on the final beat", () => {
  let state = play(createExplanationState(), { type: "start" })
  for (const step of STEPS) {
    state = play(state, { type: "tick", deltaMs: step.durationMs + 5 })
  }
  assert.equal(state.index, STEPS.length - 1)
  assert.equal(state.playing, false)
  assert.equal(state.progress, 1)
})

test("pause freezes progress and resume continues from there", () => {
  const started = play(createExplanationState(), { type: "start" })
  const partway = play(started, { type: "tick", deltaMs: 400 })
  const paused = play(partway, { type: "pause" }, { type: "tick", deltaMs: 5000 })
  assert.equal(paused.playing, false)
  assert.equal(paused.progress, partway.progress)
  const resumed = play(paused, { type: "resume" }, { type: "tick", deltaMs: 100 })
  assert.equal(resumed.playing, true)
  assert.ok(resumed.progress > partway.progress)
  const atEnd = play(createExplanationState(), { type: "start" }, { type: "select", index: STEPS.length - 1 })
  assert.deepEqual(play(atEnd, { type: "resume" }), atEnd)
})

test("next, previous, select and restart move through the beats", () => {
  const started = play(createExplanationState(), { type: "start" })
  assert.equal(play(started, { type: "next" }).index, 1)
  assert.equal(play(started, { type: "previous" }).index, 0)
  const selected = play(started, { type: "select", index: 5 })
  assert.equal(selected.index, 5)
  assert.equal(selected.playing, false)
  assert.equal(selected.progress, 0)
  assert.deepEqual(play(selected, { type: "restart" }), {
    active: true,
    index: 0,
    playing: true,
    progress: 0,
  })
  // Clamped at both ends.
  assert.equal(play(started, { type: "previous" }).index, 0)
  const last = play(started, { type: "select", index: 99 })
  assert.equal(last.index, STEPS.length - 1)
  assert.equal(play(last, { type: "next" }).index, STEPS.length - 1)
})

test("events are ignored while the explanation is inactive", () => {
  const idle = createExplanationState()
  assert.deepEqual(play(idle, { type: "tick", deltaMs: 10000 }), idle)
  assert.deepEqual(play(idle, { type: "next" }), idle)
  assert.deepEqual(play(idle, { type: "pause" }), idle)
  assert.deepEqual(play(idle, { type: "select", index: 3 }), idle)
  const exited = play(play(idle, { type: "start" }), { type: "exit" })
  assert.deepEqual(exited, idle)
})

test("currentExplanationStep follows the index", () => {
  const state = play(createExplanationState(), { type: "start" }, { type: "next" }, {
    type: "next",
  })
  assert.equal(currentExplanationStep(STEPS, state)?.id, "relu1")
})

/* ---------------------------------------------------------------- *
 * Real-activation helpers
 * ---------------------------------------------------------------- */

test("strongestChannels ranks maps by activation energy", () => {
  const conv = stage("conv1", "conv", [
    map([0, 0, 0, 0], 2, 2),
    map([1, 1, 1, 1], 2, 2),
    map([0.5, 0.5, 0.5, 0.5], 2, 2),
  ])
  assert.deepEqual(strongestChannels(conv, 2), [1, 2])
  assert.deepEqual(strongestChannels(conv, 10), [1, 2, 0])
})

test("reluTransition reports the gate's real sparsity", () => {
  const run = makeRun()
  const transition = reluTransition(run, 2)
  assert.ok(transition)
  assert.equal(transition.pre.id, "conv1")
  assert.equal(transition.post.id, "relu1")
  assert.equal(transition.channel, 0)
  assert.equal(transition.zeroFraction, 0)
  assert.equal(reluTransition(run, 1), null)
})

test("poolSample finds the window that won the max", () => {
  const input = stage("relu1", "relu", [
    map([0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0], 4, 4),
  ])
  const output = stage(
    "pool1",
    "pool",
    [map([1, 2, 0, 3], 2, 2)],
    { kind: "pool", window: 2, stride: 2, pad: 0 }
  )
  const run: ClassifierRun = {
    stages: [input, output],
    logits: new Float32Array(10),
    probabilities: new Float32Array(10),
    predicted: 0,
  }
  const sample = poolSample(run, 1, 0)
  assert.ok(sample)
  assert.equal(sample.window, 2)
  assert.equal(sample.row, 2)
  assert.equal(sample.col, 2)
  assert.equal(sample.outputRow, 1)
  assert.equal(sample.outputCol, 1)
  assert.equal(sample.max, 3)
  assert.equal(poolSample(run, 0, 0), null)
})

test("denseContributions ranks signed feature contributions", () => {
  const model = makeModel()
  const run = makeRun()
  const contributions = denseContributions(model, run, 7, 2)
  assert.ok(contributions)
  assert.equal(contributions.positive[0]?.index, 0)
  assert.ok(Math.abs((contributions.positive[0]?.value ?? 0) - 0.01) < 1e-6)
  assert.equal(contributions.negative.length, 0)
  assert.ok(Math.abs(contributions.total - 2.56) < 1e-6)
  assert.equal(denseContributions(model, run, 99, 2), null)
})

test("probabilityRanking sorts the softmax output", () => {
  const run = makeRun()
  const ranking = probabilityRanking(run)
  assert.equal(ranking[0]?.unit, 7)
  assert.ok(Math.abs((ranking[0]?.probability ?? 0) - 0.9) < 1e-6)
  assert.equal(ranking[ranking.length - 1]?.probability, 0)
})

test("buildGuideData reads the focused beat from the current run", () => {
  const model = makeModel()
  const run = makeRun()
  const byId = new Map(STEPS.map((step) => [step.id, step]))

  const conv1 = buildGuideData(model, run, byId.get("conv1")!)
  assert.deepEqual(conv1.strong, [0])
  assert.equal(conv1.channel, 0)
  assert.equal(conv1.relu, null)
  assert.equal(conv1.pool, null)

  const relu1 = buildGuideData(model, run, byId.get("relu1")!)
  assert.ok(relu1.relu)
  assert.equal(relu1.relu.pre.id, "conv1")

  const pool1 = buildGuideData(model, run, byId.get("pool1")!)
  assert.ok(pool1.pool)
  assert.equal(pool1.pool.output.id, "pool1")

  const dense = buildGuideData(model, run, byId.get("dense")!)
  assert.ok(dense.dense)
  assert.equal(dense.dense.unit, 7)

  const prediction = buildGuideData(model, run, byId.get("prediction")!)
  assert.ok(prediction.dense)
  assert.equal(prediction.dense.unit, run.predicted)
})
