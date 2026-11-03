import Link from "next/link"
import type { Metadata } from "next"
import { ArrowRightIcon, MousePointerClickIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import {
  THEMES,
  readyLessons,
  themeLessons,
  totalExperiments,
  totalReadyExperiments,
  type Theme,
} from "@/lib/themes"
import { cn } from "@/lib/utils"

export const metadata: Metadata = {
  title: "AI Visual Lab",
  description:
    "Interactive, slider-driven explanations of computer vision, language models and generative AI.",
}

export default function LabOverviewPage() {
  return (
    <div className="grid gap-6">
      <section className="relative overflow-hidden rounded-xl bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-6 ring-1 ring-foreground/10">
        <h1 className="font-heading text-3xl font-semibold tracking-tight">
          AI Visual Lab
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-muted-foreground">
          A hands-on lab for explaining machine learning. Every concept is a
          interactive experiment: change controls, inspect the maths and compare results.
        </p>
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <Badge variant="secondary">{THEMES.length} themes</Badge>
          <Badge variant="secondary">{totalExperiments()} experiments</Badge>
          <Badge variant="secondary">
            {totalReadyExperiments()} live now
          </Badge>
        </div>
      </section>

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {THEMES.map((theme) => (
          <ThemeCard key={theme.id} theme={theme} />
        ))}
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <MousePointerClickIcon className="size-4" />
            How to use the lab
          </CardTitle>
          <CardDescription>
            Pick a theme on the left, open the Playground tab and move a control
            — inspect browser calculations, recorded model results, or optional local inference.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-4 text-sm text-muted-foreground sm:grid-cols-3">
          <div className="grid gap-1">
            <span className="font-medium text-foreground">1. Choose a theme</span>
            <span>
              Computer Vision, LLMs and Generative AI, each with its own
              playground.
            </span>
          </div>
          <div className="grid gap-1">
            <span className="font-medium text-foreground">
              2. Move the controls
            </span>
            <span>
              Sliders, switches and presets map straight onto model parameters.
            </span>
          </div>
          <div className="grid gap-1">
            <span className="font-medium text-foreground">3. Read the maths</span>
            <span>
              Distributions, schedules and normalised readouts update as you
              drag.
            </span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

function ThemeCard({ theme }: { theme: Theme }) {
  const Icon = theme.icon
  const ready = readyLessons(theme)

  return (
    <Card className="transition-shadow hover:shadow-md">
      <CardHeader>
        <span
          className={cn(
            "mb-1 flex size-9 items-center justify-center rounded-lg ring-1",
            theme.accent
          )}
        >
          <Icon className="size-4" />
        </span>
        <CardTitle>{theme.label}</CardTitle>
        <CardAction>
          <Badge variant="outline">
            {ready.length}/{themeLessons(theme).length} live
          </Badge>
        </CardAction>
        <CardDescription>{theme.tagline}</CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <p className="text-sm text-muted-foreground">{theme.summary}</p>
        <div className="flex flex-wrap gap-1.5">
          {theme.concepts.map((concept) => (
            <Badge key={concept} variant="ghost">
              {concept}
            </Badge>
          ))}
        </div>
      </CardContent>
      <CardFooter>
        <Button asChild size="sm" className="w-full">
          <Link href={`/lab/${theme.id}`}>
            Open {theme.short}
            <ArrowRightIcon data-icon="inline-end" />
          </Link>
        </Button>
      </CardFooter>
    </Card>
  )
}
