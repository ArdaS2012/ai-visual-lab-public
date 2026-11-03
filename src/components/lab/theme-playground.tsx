import { CvSegmentation } from "@/components/lab/demos/cv-segmentation"
import { CvVisionPrompts } from "@/components/lab/demos/cv-vision-prompts"
import { CvDetection } from "@/components/lab/demos/cv-detection"
import { CvProjection } from "@/components/lab/demos/cv-projection"
import { EvaluationInspector } from "@/components/lab/demos/evaluation-inspector"
import type { ComponentType } from "react"
import { LightbulbIcon } from "lucide-react"

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { CnnClassifier } from "@/components/lab/demos/cnn-classifier"
import { CnnConvolution } from "@/components/lab/demos/cnn-convolution"
import { CnnPipeline } from "@/components/lab/demos/cnn-pipeline"
import { CvFilterStudio } from "@/components/lab/demos/cv-filter-studio"
import { GenaiDiffusionStudio } from "@/components/lab/demos/genai-diffusion-studio"
import { LlmSamplingLab } from "@/components/lab/demos/llm-sampling-lab"

import { RetrievalReranking } from "@/components/lab/demos/retrieval-reranking"
import { BoundedResearchAssistant } from "@/components/lab/demos/bounded-research-assistant"

/**
 * One entry per ready lesson, keyed by its path below the theme
 * (e.g. "cnn/convolution").
 */
const PLAYGROUNDS: Record<string, ComponentType> = {
  "beyond-cnns/segmentation": CvSegmentation,
  "beyond-cnns/vision-prompts": CvVisionPrompts,
  "beyond-cnns/detection-heads": CvDetection,
  "unsupervised/projection-reconstruction": CvProjection,
  "evaluation/metric-inspector": EvaluationInspector,
  "image-foundations/filter-studio": CvFilterStudio,
  "cnn/convolution": CnnConvolution,
  "cnn/pipeline": CnnPipeline,
  "cnn/classifier": CnnClassifier,
  "sampling": LlmSamplingLab,
  "rag/retrieval-reranking": RetrievalReranking,
  "agents/bounded-research-assistant": BoundedResearchAssistant,
  "diffusion": GenaiDiffusionStudio,
}

export function LessonPlayground({ path }: { path: string[] }) {
  const key = path.join("/")
  const Demo = PLAYGROUNDS[key]

  if (!Demo) {
    return (
      <Alert>
        <LightbulbIcon />
        <AlertTitle>No playground wired up yet</AlertTitle>
        <AlertDescription>
          Register <code>{key || "this lesson"}</code> in{" "}
          <code>src/components/lab/theme-playground.tsx</code> and add a demo
          component to <code>src/components/lab/demos</code>.
        </AlertDescription>
      </Alert>
    )
  }

  return <Demo />
}
