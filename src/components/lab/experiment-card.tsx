import Link from "next/link"
import { ArrowUpRightIcon, ClockIcon, LayersIcon } from "lucide-react"

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
  flattenLessons,
  nodeHref,
  type Difficulty,
  type LabNode,
} from "@/lib/themes"

const DIFFICULTY_LABEL: Record<Difficulty, string> = {
  intro: "Intro",
  intermediate: "Intermediate",
  advanced: "Advanced",
}

export function TrackCard({
  themeId,
  node,
}: {
  themeId: string
  node: LabNode
}) {
  const lessons = flattenLessons(node.children ?? [])
  const ready = lessons.filter((lesson) => lesson.status === "ready").length

  return (
    <Card size="sm" className={ready === 0 ? "opacity-75" : undefined}>
      <CardHeader>
        <CardTitle>{node.title}</CardTitle>
        <CardAction>
          <Badge variant={ready > 0 ? "secondary" : "outline"}>
            {ready > 0 ? `${ready}/${lessons.length} live` : "Planned"}
          </Badge>
        </CardAction>
        <CardDescription>{node.summary}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <Badge variant="ghost">
          <LayersIcon />
          {lessons.length} lessons
        </Badge>
      </CardContent>
      <CardFooter>
        <Button size="sm" asChild>
          <Link href={nodeHref(themeId, [node.slug])}>
            Open track
            <ArrowUpRightIcon data-icon="inline-end" />
          </Link>
        </Button>
      </CardFooter>
    </Card>
  )
}

export function LessonCard({
  themeId,
  path,
  node,
}: {
  themeId: string
  path: string[]
  node: LabNode
}) {
  const planned = node.status === "planned"

  return (
    <Card size="sm" className={planned ? "opacity-75" : undefined}>
      <CardHeader>
        <CardTitle>{node.title}</CardTitle>
        <CardAction>
          <Badge variant={planned ? "outline" : "secondary"}>
            {planned ? "Planned" : "Ready"}
          </Badge>
        </CardAction>
        <CardDescription>{node.summary}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        {node.difficulty ? (
          <Badge variant="ghost">{DIFFICULTY_LABEL[node.difficulty]}</Badge>
        ) : null}
        {node.minutes ? (
          <span className="inline-flex items-center gap-1">
            <ClockIcon className="size-3" />
            {node.minutes} min
          </span>
        ) : null}
      </CardContent>
      <CardFooter>
        {planned ? (
          <Button size="sm" variant="outline" disabled>
            On the roadmap
          </Button>
        ) : (
          <Button size="sm" asChild>
            <Link href={nodeHref(themeId, [...path, node.slug])}>
              Open lesson
              <ArrowUpRightIcon data-icon="inline-end" />
            </Link>
          </Button>
        )}
      </CardFooter>
    </Card>
  )
}
