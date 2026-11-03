import { clampNumber } from "./convolution"

/** Channel display vocabulary shared by the CNN demos. */
export const CHANNEL_LABEL = ["R", "G", "B"]
export const CHANNEL_COLOR = ["#ef4444", "#22c55e", "#3b82f6"]

/** Small deterministic PRNG so "random" filters and scenes are reproducible. */
export function mulberry32(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = Math.imul(a ^ (a >>> 15), 1 | a)
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

function scalar(
  scene: string,
  channel: number,
  row: number,
  col: number,
  size: number,
  random: () => number
): number {
  const last = size - 1 || 1
  const center = Math.floor(size / 2)

  switch (scene) {
    case "edge": {
      const left = col < size / 2
      if (channel === 0) return left ? 0.1 : 0.9
      if (channel === 1) return left ? 0.9 : 0.1
      return 0.3
    }
    case "diagonal": {
      const falling = Math.abs(row - col) / last
      const rising = Math.abs(row - (last - col)) / last
      if (channel === 0) return clampNumber(1 - falling * 2, 0, 1)
      if (channel === 1) return clampNumber(1 - rising * 2, 0, 1)
      return 0.2
    }
    case "cross": {
      if (channel === 0) return row === center ? 1 : 0.1
      if (channel === 1) return col === center ? 1 : 0.1
      return 0.2
    }
    case "noise":
      return Math.round(random() * 1000) / 1000
    case "smiley": {
      const cx = (size - 1) / 2
      const radius = size / 2 - 0.5
      const distance = Math.hypot(row - cx, col - cx)
      if (distance > radius) return 0
      const eyeRow = Math.round(cx - radius * 0.3)
      const eyeLeft = Math.round(cx - radius * 0.45)
      const eyeRight = Math.round(cx + radius * 0.45)
      const isEye =
        (row === eyeRow || row === eyeRow + 1) &&
        (col === eyeLeft || col === eyeRight)
      const mouthRow = Math.round(cx + radius * 0.35)
      const mouthHalf = Math.max(1, Math.round(radius * 0.45))
      const isMouth = row === mouthRow && Math.abs(col - cx) <= mouthHalf
      const face = isEye || isMouth ? 0.05 : 1
      if (channel === 2) return face * 0.7
      if (channel === 1) return face * 0.85
      return face
    }
    case "gradient":
    default: {
      if (channel === 0) return row / last
      if (channel === 1) return col / last
      return 0.4
    }
  }
}

/** Synthetic `[rows][cols][channels]` scenes, deterministic per scene/size. */
export function buildScene(
  scene: string,
  size: number,
  channels: number
): number[][][] {
  let seed = 2166136261
  for (const character of scene) {
    seed = Math.imul(seed ^ character.charCodeAt(0), 16777619) >>> 0
  }
  const random = mulberry32(seed + size * 7919 + channels * 104729)

  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) =>
      Array.from({ length: channels }, (_, channel) =>
        scalar(scene, channel, row, col, size, random)
      )
    )
  )
}

/** Channels-last grid → channels-first tensor (the CNN lib's layout). */
export function toChannelFirst(input: number[][][]): number[][][] {
  const size = input.length
  const channels = input[0]?.[0]?.length ?? 1
  return Array.from({ length: channels }, (_, channel) =>
    Array.from({ length: size }, (_, row) =>
      Array.from({ length: size }, (_, col) => input[row][col][channel] ?? 0)
    )
  )
}
