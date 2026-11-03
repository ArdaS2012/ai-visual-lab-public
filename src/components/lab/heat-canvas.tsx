"use client"

import * as React from "react"

import { heatRgb } from "@/lib/heatmap"
import type { FloatMap } from "@/lib/mnist-cnn"
import { cn } from "@/lib/utils"

const GREY_RAMP = (value: number) => Math.round(Math.min(1, Math.max(0, value)) * 255)

/** One channel rendered as pixels, 1 device pixel per cell. */
export const HeatCanvas = React.memo(function HeatCanvas({
  map,
  mode = "heat",
  bound = 1,
  className,
}: {
  map: FloatMap
  mode?: "heat" | "grey"
  bound?: number
  className?: string
}) {
  const canvasRef = React.useRef<HTMLCanvasElement>(null)
  const { rows, cols } = map

  React.useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || rows === 0 || cols === 0) return
    const context = canvas.getContext("2d")
    if (!context) return
    const image = context.createImageData(cols, rows)
    const { data } = image
    for (let index = 0; index < rows * cols; index += 1) {
      const offset = index * 4
      if (mode === "grey") {
        const level = GREY_RAMP(map.data[index])
        data[offset] = level
        data[offset + 1] = level
        data[offset + 2] = level
      } else {
        const [r, g, b] = heatRgb(map.data[index], bound)
        data[offset] = r
        data[offset + 1] = g
        data[offset + 2] = b
      }
      data[offset + 3] = 255
    }
    context.putImageData(image, 0, 0)
  }, [map, mode, bound, rows, cols])

  return (
    <canvas
      ref={canvasRef}
      width={cols}
      height={rows}
      aria-hidden
      className={cn("block size-full", className)}
      style={{ imageRendering: "pixelated" }}
    />
  )
})
