"use client";
import type { ReactNode, KeyboardEvent } from "react";
import { ChevronDownIcon, ScanSearchIcon } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  Collapsible,
  CollapsibleTrigger,
  CollapsibleContent,
} from "@/components/ui/collapsible";
import {
  Dialog,
  DialogContent,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";
import { InfoTip } from "@/components/lab/controls";

export function StageTitle({
  number,
  title,
  metric,
  info,
}: {
  number: number;
  title: string;
  metric?: string;
  info: string;
}) {
  return (
    <header className="flex flex-wrap items-center justify-between gap-2 text-sm font-medium">
      <span className="flex items-center gap-2">
        <Badge variant="outline" className="font-mono">
          {String(number).padStart(2, "0")}
        </Badge>
        {title}
        <InfoTip label={`About ${title}`}>{info}</InfoTip>
      </span>
      <span className="font-mono text-xs text-muted-foreground">{metric}</span>
    </header>
  );
}
export function EvidenceDetails({
  title,
  children,
}: {
  title: string;
  children: ReactNode;
}) {
  return (
    <Collapsible>
      <Card>
        <CardHeader>
          <CollapsibleTrigger asChild>
            <button className="group flex w-full items-center justify-between text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring">
              <CardTitle className="flex items-center gap-2">
                <ScanSearchIcon className="size-4" />
                {title}
              </CardTitle>
              <ChevronDownIcon className="size-4 transition-transform group-data-[state=open]:rotate-180" />
            </button>
          </CollapsibleTrigger>
        </CardHeader>
        <CollapsibleContent>
          <CardContent className="grid gap-4 text-sm leading-relaxed text-muted-foreground">
            {children}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  );
}
export function EvidenceInspector({
  title,
  description,
  open,
  onClose,
  children,
}: {
  title: string;
  description: string;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  return (
    <Dialog
      open={open}
      onOpenChange={(v) => {
        if (!v) onClose();
      }}
    >
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-4xl">
        <DialogTitle>{title}</DialogTitle>
        <DialogDescription>{description}</DialogDescription>
        <div className="grid min-w-0 gap-5">{children}</div>
      </DialogContent>
    </Dialog>
  );
}
export function JsonEvidence({ value }: { value: unknown }) {
  return (
    <pre className="max-h-96 overflow-auto rounded-lg border bg-muted/40 p-4 font-mono text-xs leading-relaxed whitespace-pre-wrap break-all">
      {JSON.stringify(value, null, 2)}
    </pre>
  );
}

export function evidenceGridKey(
  event: KeyboardEvent<HTMLDivElement>,
  columns: number,
) {
  const moves: Record<string, number> = {
    ArrowLeft: -1,
    ArrowRight: 1,
    ArrowUp: -columns,
    ArrowDown: columns,
  };
  if (!(event.key in moves)) return;
  const cells = Array.from(
    event.currentTarget.querySelectorAll<HTMLButtonElement>("button"),
  );
  const index = cells.indexOf(event.target as HTMLButtonElement);
  if (index < 0) return;
  const next =
    cells[Math.max(0, Math.min(cells.length - 1, index + moves[event.key]))];
  if (!next?.disabled) {
    event.preventDefault();
    next.focus();
  }
}
