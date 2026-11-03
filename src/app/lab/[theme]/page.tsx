import Link from "next/link"
import { notFound } from "next/navigation"
import type { Metadata } from "next"
import {
  ArrowRightIcon,
  ClockIcon,
  LayersIcon,
  ZapIcon,
} from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { LessonCard, TrackCard } from "@/components/lab/experiment-card"
import {
  getTheme,
  nodeHref,
  readyLessons,
  themeLessons,
  type Difficulty,
  type LabNode,
} from "@/lib/themes"
import { cn } from "@/lib/utils"

const DIFFICULTY_LABEL: Record<Difficulty, string> = {
  intro: "Intro",
  intermediate: "Intermediate",
  advanced: "Advanced",
}

export async function generateMetadata(
  props: PageProps<"/lab/[theme]">
): Promise<Metadata> {
  const { theme } = await props.params
  const found = getTheme(theme)

  if (!found) return { title: "Theme not found · AI Visual Lab" }

  return {
    title: `${found.label} · AI Visual Lab`,
    description: found.tagline,
  }
}

function collectLessons(
  nodes: LabNode[],
  trail: LabNode[] = []
): { node: LabNode; trail: LabNode[] }[] {
  return nodes.flatMap((node) =>
    node.kind === "lesson"
      ? [{ node, trail }]
      : collectLessons(node.children ?? [], [...trail, node])
  )
}

export default async function ThemePage(props: PageProps<"/lab/[theme]">) {
  const { theme: themeId } = await props.params
  const theme = getTheme(themeId)
  if (!theme) notFound()

  const lessons = themeLessons(theme)
  const ready = readyLessons(theme)
  const entries = collectLessons(theme.nodes)
  const planned = entries.filter((entry) => entry.node.status === "planned")
  const Icon = theme.icon

  return (
    <div className="grid gap-6">
      <section
        className={cn(
          "relative overflow-hidden rounded-xl bg-gradient-to-br p-6 ring-1 ring-foreground/10",
          theme.gradient
        )}
      >
        <div className="flex flex-wrap items-start gap-4">
          <span
            className={cn(
              "flex size-11 shrink-0 items-center justify-center rounded-xl ring-1",
              theme.accent
            )}
          >
            <Icon className="size-5" />
          </span>
          <div className="grid min-w-0 flex-1 gap-1.5">
            <h1 className="font-heading text-2xl font-semibold tracking-tight">
              {theme.label}
            </h1>
            <p className="text-sm text-muted-foreground">{theme.tagline}</p>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <Badge variant="secondary">
                <LayersIcon />
                {lessons.length} lessons
              </Badge>
              <Badge variant="secondary">
                <ZapIcon />
                {ready.length} interactive now
              </Badge>
              {theme.concepts.map((concept) => (
                <Badge key={concept} variant="outline">
                  {concept}
                </Badge>
              ))}
            </div>
          </div>
        </div>
        <p className="mt-5 max-w-3xl text-sm text-muted-foreground">
          {theme.summary}
        </p>
      </section>

      <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {theme.nodes.map((node) =>
          node.kind === "track" ? (
            <TrackCard key={node.slug} themeId={theme.id} node={node} />
          ) : (
            <LessonCard
              key={node.slug}
              themeId={theme.id}
              path={[]}
              node={node}
            />
          )
        )}
      </section>

      <Card>
        <CardHeader>
          <CardTitle>Roadmap</CardTitle>
          <CardDescription>
            Planned lessons for {theme.label}. Everything is driven by the theme
            registry, so adding one is a data change plus a component.
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          {planned.map((entry) => (
            <div
              key={`${entry.trail.map((node) => node.slug).join("/")}/${entry.node.slug}`}
              className="flex flex-wrap items-start justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2.5"
            >
              <div className="grid gap-0.5">
                <div className="flex flex-wrap items-center gap-2">
                  <Link
                    href={nodeHref(theme.id, [
                      ...entry.trail.map((node) => node.slug),
                      entry.node.slug,
                    ])}
                    className="text-sm font-medium hover:underline"
                  >
                    {entry.node.title}
                  </Link>
                  {entry.trail.length > 0 ? (
                    <Badge variant="ghost" className="text-[10px]">
                      {entry.trail
                        .map((node) => node.title)
                        .join(" / ")}
                    </Badge>
                  ) : null}
                </div>
                <span className="max-w-xl text-xs text-muted-foreground">
                  {entry.node.summary}
                </span>
              </div>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                {entry.node.minutes ? (
                  <>
                    <ClockIcon className="size-3" />
                    {entry.node.minutes} min
                  </>
                ) : null}
                {entry.node.difficulty ? (
                  <Badge variant="ghost">
                    {DIFFICULTY_LABEL[entry.node.difficulty]}
                  </Badge>
                ) : null}
              </div>
            </div>
          ))}
          {planned.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              Nothing queued — every lesson for this theme is live.
            </p>
          ) : null}
        </CardContent>
      </Card>

      <Button asChild variant="ghost" size="sm" className="w-fit">
        <Link href="/lab">
          <ArrowRightIcon className="rotate-180" data-icon="inline-start" />
          All themes
        </Link>
      </Button>
    </div>
  )
}
