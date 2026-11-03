"use client"

import Link from "next/link"
import { usePathname } from "next/navigation"
import { ChevronRightIcon, FlaskConicalIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible"
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuAction,
  SidebarMenuBadge,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarMenuSub,
  SidebarMenuSubButton,
  SidebarMenuSubItem,
  SidebarRail,
  SidebarSeparator,
} from "@/components/ui/sidebar"
import { ThemeToggle } from "@/components/theme-toggle"
import {
  flattenLessons,
  nodeHref,
  THEMES,
  totalReadyExperiments,
  type LabNode,
} from "@/lib/themes"

function isPathActive(pathname: string, href: string) {
  return pathname === href || pathname.startsWith(`${href}/`)
}

function countPlanned(nodes: LabNode[]) {
  return flattenLessons(nodes).filter((node) => node.status === "planned")
    .length
}

function SidebarNode({
  themeId,
  node,
  path,
}: {
  themeId: string
  node: LabNode
  path: string[]
}) {
  const pathname = usePathname()
  const nodePath = [...path, node.slug]
  const href = nodeHref(themeId, nodePath)
  const isActive = isPathActive(pathname, href)

  if (node.kind === "lesson") {
    return (
      <SidebarMenuSubItem>
        <SidebarMenuSubButton
          asChild
          isActive={isActive}
          className={
            node.status === "planned" ? "text-muted-foreground" : undefined
          }
        >
          <Link href={href}>
            <span>{node.title}</span>
          </Link>
        </SidebarMenuSubButton>
      </SidebarMenuSubItem>
    )
  }

  return (
    <SidebarTrack
      themeId={themeId}
      node={node}
      path={path}
      isActive={isActive}
      href={href}
    />
  )
}

function SidebarTrack({
  themeId,
  node,
  path,
  isActive,
  href,
}: {
  themeId: string
  node: LabNode
  path: string[]
  isActive: boolean
  href: string
}) {
  const planned = countPlanned(node.children ?? [])
  const childPath = [...path, node.slug]

  return (
    <Collapsible
      asChild
      key={isActive ? "active" : "idle"}
      defaultOpen={isActive}
      className="group/collapsible"
    >
      <SidebarMenuSubItem>
        <SidebarMenuSubButton
          asChild
          isActive={isActive}
          className="pr-11"
        >
          <Link href={href}>
            <span>{node.title}</span>
          </Link>
        </SidebarMenuSubButton>
        {planned > 0 ? (
          <span className="pointer-events-none absolute top-1/2 right-7 -translate-y-1/2 text-[10px] font-medium text-muted-foreground tabular-nums">
            {planned}
          </span>
        ) : null}
        <CollapsibleTrigger asChild>
          <button
            type="button"
            className="absolute top-1/2 right-1 flex size-5 -translate-y-1/2 items-center justify-center rounded-md text-sidebar-foreground/70 ring-sidebar-ring outline-hidden transition-transform hover:bg-sidebar-accent focus-visible:ring-2 data-[state=open]:rotate-90 [&_svg]:size-4 [&_svg]:shrink-0"
          >
            <ChevronRightIcon />
            <span className="sr-only">Toggle {node.title}</span>
          </button>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <SidebarMenuSub>
            {(node.children ?? []).map((child) => (
              <SidebarNode
                key={child.slug}
                themeId={themeId}
                node={child}
                path={childPath}
              />
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuSubItem>
    </Collapsible>
  )
}

function ThemeSection({ theme }: { theme: (typeof THEMES)[number] }) {
  const pathname = usePathname()
  const href = `/lab/${theme.id}`
  const isActive = isPathActive(pathname, href)
  const planned = countPlanned(theme.nodes)
  const Icon = theme.icon

  return (
    <Collapsible
      asChild
      key={isActive ? "active" : "idle"}
      defaultOpen={isActive}
      className="group/collapsible"
    >
      <SidebarMenuItem>
        <SidebarMenuButton asChild isActive={isActive} tooltip={theme.label}>
          <Link href={href}>
            <Icon />
            <span>{theme.label}</span>
          </Link>
        </SidebarMenuButton>
        {planned > 0 ? (
          <SidebarMenuBadge>{planned}</SidebarMenuBadge>
        ) : null}
        <CollapsibleTrigger asChild>
          <SidebarMenuAction className="transition-transform data-[state=open]:rotate-90">
            <ChevronRightIcon />
            <span className="sr-only">Toggle {theme.label} lessons</span>
          </SidebarMenuAction>
        </CollapsibleTrigger>
        <CollapsibleContent>
          <SidebarMenuSub>
            {theme.nodes.map((node) => (
              <SidebarNode
                key={node.slug}
                themeId={theme.id}
                node={node}
                path={[]}
              />
            ))}
          </SidebarMenuSub>
        </CollapsibleContent>
      </SidebarMenuItem>
    </Collapsible>
  )
}

export function AppSidebar() {
  return (
    <Sidebar variant="inset" collapsible="icon">
      <SidebarHeader>
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton size="lg" asChild tooltip="AI Visual Lab">
              <Link href="/lab">
                <span className="flex size-8 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                  <FlaskConicalIcon className="size-4" />
                </span>
                <span className="flex min-w-0 flex-col gap-0.5 leading-none">
                  <span className="truncate text-sm font-medium">
                    AI Visual Lab
                  </span>
                  <span className="truncate text-xs text-muted-foreground">
                    {totalReadyExperiments()} interactive experiments
                  </span>
                </span>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarSeparator />
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Themes</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {THEMES.map((theme) => (
                <ThemeSection key={theme.id} theme={theme} />
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarSeparator />
      <SidebarFooter>
        <SidebarMenu>
          <SidebarMenuItem>
            <div className="flex items-center gap-2 rounded-lg px-2 py-1.5 group-data-[collapsible=icon]:justify-center group-data-[collapsible=icon]:px-0">
              <ThemeToggle />
              <div className="flex min-w-0 flex-1 flex-col leading-none group-data-[collapsible=icon]:hidden">
                <span className="truncate text-xs font-medium">Appearance</span>
                <span className="truncate text-[11px] text-muted-foreground">
                  Light, dark or system
                </span>
              </div>
            </div>
          </SidebarMenuItem>
          <SidebarMenuItem>
            <div className="px-2 pb-1 group-data-[collapsible=icon]:hidden">
              <Badge variant="outline">Prototype</Badge>
            </div>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  )
}
