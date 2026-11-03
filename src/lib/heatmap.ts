import type { CSSProperties } from "react"

import { clampNumber } from "./convolution"

/** Round to at most two decimals for readouts and cell labels. */
export function formatValue(value: number): string {
  const rounded = Math.round(value * 100) / 100
  if (Number.isInteger(rounded)) return String(rounded)
  return rounded.toFixed(2)
}

/**
 * High-contrast accent for receptive-field overlays drawn on top of heatmaps.
 * Theme `primary` is near-black/near-white and disappears against the red/blue
 * diverging scale; amber with a dark contour reads on every cell, light or
 * dark, in both colour schemes.
 */
export const RF_ACCENT = "#f59e0b"
export const RF_ACCENT_SOFT = "rgba(245, 158, 11, 0.18)"
export const RF_ACCENT_EDGE = "rgba(15, 23, 42, 0.65)"

/** Diverging heatmap: red for positive, blue for negative, near-white at 0. */
export function heatColor(value: number, bound: number): string {  if (bound <= 1e-6) return "hsl(220 12% 93%)"
  const t = clampNumber(value / bound, -1, 1)
  const lightness = 92 - Math.abs(t) * 52
  const hue = t >= 0 ? 8 : 220
  const saturation = Math.abs(t) < 1e-6 ? 12 : 72
  return `hsl(${hue} ${saturation}% ${lightness}%)`
}

export function heatTextColor(value: number, bound: number): string {
  const t = bound <= 1e-6 ? 0 : clampNumber(value / bound, -1, 1)
  return Math.abs(t) > 0.55 ? "#f8fafc" : "#0f172a"
}

export function hslToRgb(
  hue: number,
  saturation: number,
  lightness: number
): [number, number, number] {
  if (saturation === 0) {
    const value = Math.round(lightness * 255)
    return [value, value, value]
  }
  const q =
    lightness < 0.5
      ? lightness * (1 + saturation)
      : lightness + saturation - lightness * saturation
  const p = 2 * lightness - q
  const channel = (t: number) => {
    let temp = t
    if (temp < 0) temp += 1
    if (temp > 1) temp -= 1
    if (temp < 1 / 6) return p + (q - p) * 6 * temp
    if (temp < 1 / 2) return q
    if (temp < 2 / 3) return p + (q - p) * (2 / 3 - temp) * 6
    return p
  }
  return [
    Math.round(channel(hue + 1 / 3) * 255),
    Math.round(channel(hue) * 255),
    Math.round(channel(hue - 1 / 3) * 255),
  ]
}

/** Canvas twin of `heatColor`, returning 8-bit RGB instead of a CSS string. */
export function heatRgb(value: number, bound: number): [number, number, number] {
  if (bound <= 1e-6) return hslToRgb(220 / 360, 0.12, 0.93)
  const t = clampNumber(value / bound, -1, 1)
  const lightness = (92 - Math.abs(t) * 52) / 100
  const saturation = Math.abs(t) < 1e-6 ? 0.12 : 0.72
  return hslToRgb((t >= 0 ? 8 : 220) / 360, saturation, lightness)
}

/**
 * Green for positive, red for negative weights. `scale` is the value that
 * saturates the tint (default 8 suits kernel presets; random filters pass
 * their own largest weight).
 */
export function weightStyle(
  value: number,
  colorWeights: boolean,
  scale = 8
): CSSProperties {
  if (!colorWeights || value === 0) return {}
  const alpha = Math.min(0.85, Math.abs(value) / Math.max(1e-6, scale) + 0.18)
  const rgb = value > 0 ? "34,197,94" : "239,68,68"
  return {
    backgroundColor: `rgba(${rgb},${alpha})`,
    color: Math.abs(value) > scale * 0.375 ? "#f8fafc" : undefined,
  }
}
