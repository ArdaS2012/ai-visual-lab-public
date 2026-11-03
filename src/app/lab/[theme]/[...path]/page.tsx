import Link from "next/link"
import { notFound } from "next/navigation"
import type { Metadata } from "next"
import {
  ArrowLeftIcon,
  ClockIcon,
  LayersIcon,
  SparklesIcon,
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
import { Separator } from "@/components/ui/separator"
import { LessonCard } from "@/components/lab/experiment-card"
import { LessonPlayground } from "@/components/lab/theme-playground"
import {
  findNode,
  flattenLessons,
  getTheme,
  nodeHref,
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
  props: PageProps<"/lab/[theme]/[...path]">
): Promise<Metadata> {
  const { theme: themeId, path } = await props.params
  const theme = getTheme(themeId)
  if (!theme) return { title: "Not found · AI Visual Lab" }

  const node = findNode(theme.nodes, path)
  if (!node) return { title: "Not found · AI Visual Lab" }

  return {
    title: `${node.title} · ${theme.label} · AI Visual Lab`,
    description: node.summary,
  }
}

export default async function LabNodePage(
  props: PageProps<"/lab/[theme]/[...path]">
) {
  const { theme: themeId, path } = await props.params
  const theme = getTheme(themeId)
  if (!theme) notFound()

  const node = findNode(theme.nodes, path)
  if (!node) notFound()

  const parentPath = path.slice(0, -1)
  const parent = findNode(theme.nodes, parentPath)
  const backHref = nodeHref(theme.id, parentPath)
  const backLabel = parent ? parent.title : theme.label
  const Icon = theme.icon

  const children = node.children ?? []
  const lessons =
    node.kind === "track" ? flattenLessons(children) : [node]
  const ready = lessons.filter((lesson) => lesson.status === "ready").length

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
            <p className="text-xs text-muted-foreground">
              {theme.label}
              {parent ? ` · ${parent.title}` : ""}
            </p>
            <h1 className="font-heading text-2xl font-semibold tracking-tight">
              {node.title}
            </h1>
            <div className="mt-1 flex flex-wrap items-center gap-2">
              <Badge variant="outline">
                {node.kind === "track"
                  ? "Track"
                  : "Lesson"}
              </Badge>
              <Badge variant={node.status === "ready" ? "secondary" : "outline"}>
                {node.status === "ready" ? "Ready" : "Planned"}
              </Badge>
              {node.difficulty ? (
                <Badge variant="ghost">
                  {DIFFICULTY_LABEL[node.difficulty]}
                </Badge>
              ) : null}
              {node.minutes ? (
                <Badge variant="ghost">
                  <ClockIcon />
                  {node.minutes} min
                </Badge>
              ) : null}
              {node.kind === "track" ? (
                <Badge variant="ghost">
                  <LayersIcon />
                  {ready}/{lessons.length} live
                </Badge>
              ) : null}
            </div>
          </div>
        </div>
        <p className="mt-5 max-w-3xl text-sm text-muted-foreground">
          {node.summary}
        </p>
      </section>

      {node.kind === "track" ? (
        <>
          <section className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {children.map((child) => (
              <LessonCard
                key={child.slug}
                themeId={theme.id}
                path={path}
                node={child}
              />
            ))}
          </section>
          <RoadmapCard themeId={theme.id} path={path} nodes={children} />
        </>
      ) : node.status === "ready" ? (
        <LessonPlayground path={path} />
      ) : (
        <PlannedCard node={node} />
      )}

      <Button asChild variant="ghost" size="sm" className="w-fit">
        <Link href={backHref}>
          <ArrowLeftIcon data-icon="inline-start" />
          Back to {backLabel}
        </Link>
      </Button>
    </div>
  )
}

function PlannedCard({ node }: { node: LabNode }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <SparklesIcon className="size-4" />
          On the roadmap
        </CardTitle>
        <CardDescription>
          {node.title} is planned but not built yet. The registry entry and the
          demo component are all that is missing.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3 text-sm text-muted-foreground">
        <p>{node.summary}</p>
        <Separator />
        <div className="flex flex-wrap items-center gap-2">
          {node.difficulty ? (
            <Badge variant="ghost">{DIFFICULTY_LABEL[node.difficulty]}</Badge>
          ) : null}
          {node.minutes ? (
            <Badge variant="ghost">
              <ClockIcon />
              {node.minutes} min
            </Badge>
          ) : null}
        </div>
      </CardContent>
    </Card>
  )
}

function RoadmapCard({
  themeId,
  path,
  nodes,
}: {
  themeId: string
  path: string[]
  nodes: LabNode[]
}) {
  const planned = flattenLessons(nodes).filter(
    (lesson) => lesson.status === "planned"
  )

  return (
    <Card>
      <CardHeader>
        <CardTitle>Planned lessons</CardTitle>
        <CardDescription>
          The backlog for this track, driven entirely by the theme registry.
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        {planned.map((lesson) => (
          <div
            key={lesson.slug}
            className="flex flex-wrap items-start justify-between gap-3 rounded-lg bg-muted/50 px-3 py-2.5"
          >
            <div className="grid gap-0.5">
              <Link
                href={nodeHref(themeId, [...path, lesson.slug])}
                className="text-sm font-medium hover:underline"
              >
                {lesson.title}
              </Link>
              <span className="max-w-xl text-xs text-muted-foreground">
                {lesson.summary}
              </span>
            </div>
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              {lesson.minutes ? (
                <>
                  <ClockIcon className="size-3" />
                  {lesson.minutes} min
                </>
              ) : null}
              {lesson.difficulty ? (
                <Badge variant="ghost">
                  {DIFFICULTY_LABEL[lesson.difficulty]}
                </Badge>
              ) : null}
            </div>
          </div>
        ))}
        {planned.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Nothing queued — every lesson in this track is live.
          </p>
        ) : null}
      </CardContent>
    </Card>
  )
}
