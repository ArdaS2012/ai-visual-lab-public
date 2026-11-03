"use client"

import * as React from "react"

import {
  createExplanationState,
  currentExplanationStep,
  explanationReducer,
  type ExplanationEvent,
  type ExplanationState,
  type ExplanationStep,
} from "@/lib/explanation"

/**
 * Drives a step sequence from `src/lib/explanation.ts` with a rAF clock.
 *
 * Stage progress is published through `subscribe` instead of React state, so a
 * progress bar can update every frame without re-rendering the network board;
 * React only re-renders when the active stage or play state actually changes.
 */
export type ExplanationController<Id extends string = string> = {
  state: ExplanationState
  step: ExplanationStep<Id> | undefined
  index: number
  total: number
  start: () => void
  exit: () => void
  next: () => void
  previous: () => void
  pause: () => void
  resume: () => void
  restart: () => void
  select: (index: number) => void
  /** Per-frame stage progress (0…1); returns an unsubscribe function. */
  subscribe: (listener: (progress: number) => void) => () => void
}

export function useExplanation<Id extends string>(
  steps: readonly ExplanationStep<Id>[]
): ExplanationController<Id> {
  const [state, setState] = React.useState<ExplanationState>(createExplanationState)
  const stateRef = React.useRef<ExplanationState>(state)
  const listeners = React.useRef(new Set<(progress: number) => void>())

  const publish = React.useCallback((next: ExplanationState) => {
    const previous = stateRef.current
    stateRef.current = next
    for (const listener of listeners.current) listener(next.progress)
    if (
      previous.active !== next.active ||
      previous.index !== next.index ||
      previous.playing !== next.playing
    ) {
      setState(next)
    }
  }, [])

  const dispatch = React.useCallback(
    (event: ExplanationEvent) => {
      publish(explanationReducer(steps, stateRef.current, event))
    },
    [steps, publish]
  )

  const playing = state.active && state.playing

  React.useEffect(() => {
    if (!playing) return
    let frame = 0
    let last = performance.now()
    const loop = (now: number) => {
      // Cap the delta so a backgrounded tab does not skip several beats at once.
      const deltaMs = Math.min(120, now - last)
      last = now
      publish(explanationReducer(steps, stateRef.current, { type: "tick", deltaMs }))
      frame = window.requestAnimationFrame(loop)
    }
    frame = window.requestAnimationFrame(loop)
    return () => window.cancelAnimationFrame(frame)
  }, [playing, steps, publish])

  const subscribe = React.useCallback((listener: (progress: number) => void) => {
    listeners.current.add(listener)
    listener(stateRef.current.progress)
    return () => {
      listeners.current.delete(listener)
    }
  }, [])

  const start = React.useCallback(() => dispatch({ type: "start" }), [dispatch])
  const exit = React.useCallback(() => dispatch({ type: "exit" }), [dispatch])
  const next = React.useCallback(() => dispatch({ type: "next" }), [dispatch])
  const previous = React.useCallback(() => dispatch({ type: "previous" }), [dispatch])
  const pause = React.useCallback(() => dispatch({ type: "pause" }), [dispatch])
  const resume = React.useCallback(() => dispatch({ type: "resume" }), [dispatch])
  const restart = React.useCallback(() => dispatch({ type: "restart" }), [dispatch])
  const select = React.useCallback(
    (index: number) => dispatch({ type: "select", index }),
    [dispatch]
  )

  return {
    state,
    step: currentExplanationStep(steps, state),
    index: state.index,
    total: steps.length,
    start,
    exit,
    next,
    previous,
    pause,
    resume,
    restart,
    select,
    subscribe,
  }
}
