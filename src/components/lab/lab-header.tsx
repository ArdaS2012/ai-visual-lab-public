"use client"

import { Fragment } from "react"
import Link from "next/link"
import { usePathname } from "next/navigation"

import {
  Breadcrumb,
  BreadcrumbItem,
  BreadcrumbLink,
  BreadcrumbList,
  BreadcrumbPage,
  BreadcrumbSeparator,
} from "@/components/ui/breadcrumb"
import { Separator } from "@/components/ui/separator"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { ThemeToggle } from "@/components/theme-toggle"
import { getTheme, nodeHref, nodeTrail } from "@/lib/themes"

export function LabHeader() {
  const pathname = usePathname()
  const segments = pathname.split("/").filter(Boolean)
  const [, themeId, ...path] = segments
  const theme = themeId ? getTheme(themeId) : undefined
  const trail = theme ? nodeTrail(theme.nodes, path) : []

  return (
    <header className="sticky top-0 z-20 flex h-14 shrink-0 items-center gap-2 rounded-t-xl bg-background/80 px-4 backdrop-blur-md md:px-6">
      <SidebarTrigger className="-ml-1" />
      <Separator orientation="vertical" className="mr-1 h-4" />
      <Breadcrumb className="min-w-0 flex-1">
        <BreadcrumbList className="flex-nowrap overflow-hidden">
          <BreadcrumbItem className={theme ? "hidden shrink-0 md:block" : "min-w-0"}>
            <BreadcrumbLink asChild>
              <Link href="/lab">Visual Lab</Link>
            </BreadcrumbLink>
          </BreadcrumbItem>
          {theme ? (
            <>
              <BreadcrumbSeparator className="hidden md:block" />
              <BreadcrumbItem className={trail.length > 0 ? "hidden min-w-0 md:block" : "min-w-0"}>
                {trail.length > 0 ? (
                  <BreadcrumbLink asChild>
                    <Link href={nodeHref(theme.id, [])} className="block truncate">
                      {theme.label}
                    </Link>
                  </BreadcrumbLink>
                ) : (
                  <BreadcrumbPage className="block truncate">
                    {theme.label}
                  </BreadcrumbPage>
                )}
              </BreadcrumbItem>
            </>
          ) : null}
          {theme
            ? trail.map((node, index) => {
                const isLast = index === trail.length - 1
                return (
                  <Fragment key={`${node.slug}-${index}`}>
                    <BreadcrumbSeparator className="hidden md:block" />
                    <BreadcrumbItem className={isLast ? "min-w-0" : "hidden min-w-0 md:block"}>
                      {isLast ? (
                        <BreadcrumbPage className="block truncate">
                          {node.title}
                        </BreadcrumbPage>
                      ) : (
                        <BreadcrumbLink asChild>
                          <Link
                            href={nodeHref(
                              theme.id,
                              trail
                                .slice(0, index + 1)
                                .map((entry) => entry.slug)
                            )}
                            className="block truncate"
                          >
                            {node.title}
                          </Link>
                        </BreadcrumbLink>
                      )}
                    </BreadcrumbItem>
                  </Fragment>
                )
              })
            : null}
        </BreadcrumbList>
      </Breadcrumb>
      <ThemeToggle />
    </header>
  )
}
