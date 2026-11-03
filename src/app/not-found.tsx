import Link from "next/link"
import { CompassIcon } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export default function NotFound() {
  return (
    <div className="flex min-h-svh items-center justify-center p-6">
      <Card className="w-full max-w-md">
        <CardHeader>
          <span className="mb-1 flex size-9 items-center justify-center rounded-lg bg-muted ring-1 ring-foreground/10">
            <CompassIcon className="size-4" />
          </span>
          <CardTitle>This lab bench is empty</CardTitle>
          <CardDescription>
            The theme or experiment you asked for does not exist yet.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Button asChild>
            <Link href="/lab">Back to all themes</Link>
          </Button>
        </CardContent>
      </Card>
    </div>
  )
}
