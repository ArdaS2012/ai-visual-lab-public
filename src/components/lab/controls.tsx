"use client"

import * as React from "react"
import { InfoIcon } from "lucide-react"

import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

/** Small hover/focus target that keeps explanations out of the layout. */
export function InfoTip({
  children,
  label = "More info",
  className,
}: {
  children: React.ReactNode
  label?: string
  className?: string
}) {
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          aria-label={label}
          className={cn(
            "inline-flex size-4 shrink-0 items-center justify-center rounded-full text-muted-foreground/70 transition-colors hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
            className
          )}
        >
          <InfoIcon className="size-3.5" />
        </button>
      </TooltipTrigger>
      <TooltipContent
        side="top"
        align="center"
        className="max-w-72 text-pretty leading-relaxed"
      >
        {children}
      </TooltipContent>
    </Tooltip>
  )
}

function LabelWithInfo({ htmlFor, label, info }: {
  htmlFor?: string
  label: string
  info?: string
}) {
  return (
    <span className="flex items-center gap-1.5">
      <Label htmlFor={htmlFor} className="text-sm font-medium">
        {label}
      </Label>
      {info ? <InfoTip>{info}</InfoTip> : null}
    </span>
  )
}

type SliderFieldProps = {
  label: string
  value: number
  onValueChange: (value: number) => void
  min: number
  max: number
  step?: number
  format?: (value: number) => string
  hint?: string
  info?: string
  disabled?: boolean
  className?: string
}

export function SliderField({
  label,
  value,
  onValueChange,
  min,
  max,
  step = 1,
  format,
  hint,
  info,
  disabled,
  className,
}: SliderFieldProps) {
  const id = React.useId()

  return (
    <div className={cn("grid gap-2", className)}>
      <div className="flex items-center justify-between gap-3">
        <LabelWithInfo htmlFor={id} label={label} info={info} />
        <Badge variant="secondary" className="font-mono tabular-nums">
          {format ? format(value) : value}
        </Badge>
      </div>
      <Slider
        id={id}
        aria-label={label}
        value={[value]}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onValueChange={(next) => onValueChange(next[0] ?? value)}
      />
      {hint ? (
        <p className="text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}

type SwitchFieldProps = {
  label: string
  checked: boolean
  onCheckedChange: (checked: boolean) => void
  hint?: string
  info?: string
  disabled?: boolean
  className?: string
}

export function SwitchField({
  label,
  checked,
  onCheckedChange,
  hint,
  info,
  disabled,
  className,
}: SwitchFieldProps) {
  const id = React.useId()

  return (
    <div className={cn("flex items-start justify-between gap-4", className)}>
      <div className="grid gap-1">
        <LabelWithInfo htmlFor={id} label={label} info={info} />
        {hint ? (
          <p className="text-xs text-muted-foreground">{hint}</p>
        ) : null}
      </div>
      <Switch
        id={id}
        checked={checked}
        onCheckedChange={onCheckedChange}
        disabled={disabled}
      />
    </div>
  )
}

type SelectFieldProps = {
  label: string
  value: string
  onValueChange: (value: string) => void
  options: { value: string; label: string }[]
  hint?: string
  info?: string
  disabled?: boolean
  className?: string
}

export function SelectField({
  label,
  value,
  onValueChange,
  options,
  hint,
  info,
  disabled,
  className,
}: SelectFieldProps) {
  const id = React.useId()

  return (
    <div className={cn("grid gap-2", className)}>
      <LabelWithInfo htmlFor={id} label={label} info={info} />
      <Select value={value} onValueChange={onValueChange} disabled={disabled}>
        <SelectTrigger id={id} className="w-full">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
      {hint ? (
        <p className="text-xs text-muted-foreground">{hint}</p>
      ) : null}
    </div>
  )
}

/** Segmented button picker for small, exclusive option sets. */
export function ChoiceField<T extends string>({
  label,
  value,
  options,
  onChange,
  info,
  disabled,
}: {
  label: string
  value: T
  options: { value: T; label: string }[]
  onChange: (value: T) => void
  info?: string
  disabled?: boolean
}) {
  return (
    <div className="grid gap-2">
      <span className="flex items-center gap-1.5 text-sm font-medium">
        {label}
        {info ? <InfoTip>{info}</InfoTip> : null}
      </span>
      <div className="flex flex-wrap gap-1.5">
        {options.map((option) => (
          <Button
            key={option.value}
            type="button"
            size="sm"
            variant={value === option.value ? "default" : "outline"}
            disabled={disabled}
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
          >
            {option.label}
          </Button>
        ))}
      </div>
    </div>
  )
}

/** Muted group label + fields, the unit the controls rail is built from. */
export function ControlGroup({
  label,
  children,
}: {
  label: string
  children: React.ReactNode
}) {
  return (
    <div className="grid gap-4">
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      {children}
    </div>
  )
}

export function MetricRow({
  label,
  value,
  children,
}: {
  label: string
  value?: React.ReactNode
  children?: React.ReactNode
}) {
  return (
    <div className="grid gap-1.5">
      <div className="flex items-center justify-between gap-3 text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono tabular-nums">{value}</span>
      </div>
      {children}
    </div>
  )
}
