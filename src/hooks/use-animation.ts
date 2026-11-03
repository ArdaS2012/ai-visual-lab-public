"use client"

import * as React from "react"

/** Layout effect on the client, plain effect during SSR/prerender. */
export const useIsomorphicLayoutEffect =
  typeof window === "undefined" ? React.useEffect : React.useLayoutEffect

const REDUCED_MOTION_QUERY = "(prefers-reduced-motion: reduce)"

/**
 * Tracks the user's motion preference. Animations in the lab check this and
 * fall back to instant state changes, never to hidden content.
 */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = React.useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia(REDUCED_MOTION_QUERY).matches
  )

  React.useEffect(() => {
    const query = window.matchMedia(REDUCED_MOTION_QUERY)
    const update = () => setReduced(query.matches)
    update()
    query.addEventListener("change", update)
    return () => query.removeEventListener("change", update)
  }, [])

  return reduced
}
