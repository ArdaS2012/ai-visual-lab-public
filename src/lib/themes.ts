import { BrainCircuit, Eye, Sparkles, type LucideIcon } from "lucide-react"

export type Difficulty = "intro" | "intermediate" | "advanced"

export type ExperimentStatus = "ready" | "planned"

export type NodeKind = "track" | "lesson"

export type LabNode = {
  slug: string
  title: string
  summary: string
  kind: NodeKind
  status: ExperimentStatus
  difficulty?: Difficulty
  minutes?: number
  children?: LabNode[]
}

export type Theme = {
  id: string
  label: string
  short: string
  tagline: string
  summary: string
  icon: LucideIcon
  /** Chip styling for the theme icon. */
  accent: string
  /** Hero backdrop gradient. */
  gradient: string
  concepts: string[]
  nodes: LabNode[]
}

export const DEFAULT_THEME_ID = "computer-vision"

export const THEMES: Theme[] = [
  {
    id: "computer-vision",
    label: "Computer Vision",
    short: "Vision",
    tagline: "Teach machines to see, filter and segment images.",
    summary:
      "Start with pixels and work your way up: convolutions, filters, feature maps and detection heads. Every parameter is wired to a live preview so the maths stays visible.",
    icon: Eye,
    accent:
      "bg-sky-500/10 text-sky-600 ring-sky-500/20 dark:text-sky-300 dark:ring-sky-400/25",
    gradient: "from-sky-500/20 via-sky-500/5 to-transparent",
    concepts: ["Convolution", "Filters & kernels", "Feature maps", "Detection"],
    nodes: [
      { slug: "unsupervised", title: "Unsupervised Vision", summary: "Explore image features through projection and reconstruction.", kind: "track", status: "ready", children: [
        { slug: "projection-reconstruction", title: "Projection & Reconstruction", summary: "Recompute PCA rank and image error; compare real precomputed t-SNE/UMAP neighborhood views of the same image features.", kind: "lesson", status: "ready", difficulty: "intermediate", minutes: 10,
        },
      ] },
      {
        slug: "image-foundations",
        title: "Image Foundations",
        summary:
          "What an image really is: a grid of numbers per channel. Filters, colour spaces and histograms before any network shows up.",
        kind: "track",
        status: "ready",
        children: [
          {
            slug: "filter-studio",
            title: "Filter Studio",
            summary:
              "Blur, sharpen and re-colour an image while watching the equivalent CSS filter chain build up.",
            kind: "lesson",
            difficulty: "intro",
            minutes: 8,
            status: "ready",
          },
          {
            slug: "pixels-and-channels",
            title: "Pixels & Colour Channels",
            summary:
              "Split an image into R, G and B planes and see how one pixel becomes three numbers.",
            kind: "lesson",
            difficulty: "intro",
            minutes: 8,
            status: "planned",
          },
          {
            slug: "histograms",
            title: "Histograms",
            summary:
              "Count how often each intensity occurs and reason about exposure, contrast and clipping.",
            kind: "lesson",
            difficulty: "intro",
            minutes: 10,
            status: "planned",
          },
        ],
      },
      {
        slug: "cnn",
        title: "CNN",
        summary:
          "From one sliding kernel to a real forward pass: convolution, stride, padding and pooling live in the playground, a sandbox stacks them into a network, and a pretrained classifier lets you unfold the whole thing in 3D and read every activation.",
        kind: "track",
        status: "ready",
        children: [
          {
            slug: "convolution",
            title: "Convolution Playground",
            summary:
              "Drag a kernel across a grid and inspect how each receptive field produces one output pixel. Stride, padding and pooling are live controls in the same playground.",
            kind: "lesson",
            difficulty: "intermediate",
            minutes: 15,
            status: "ready",
          },
          {
            slug: "pipeline",
            title: "CNN Pipeline",
            summary:
              "Chain conv → ReLU → pooling into a network, run the forward pass stage by stage, and read every activation and hidden layer: feature maps, sparsity, parameters and the receptive field of any cell.",
            kind: "lesson",
            difficulty: "intermediate",
            minutes: 15,
            status: "ready",
          },
          {
            slug: "classifier",
            title: "Trained CNN Classifier",
            summary:
              "A real pretrained MNIST network (98.9% test accuracy) you can draw into: unfold it layer by layer from a 3D board, inspect every activation map, filter and receptive field, and end in the fully connected head that turns pooled features into class scores.",
            kind: "lesson",
            difficulty: "advanced",
            minutes: 18,
            status: "ready",
          },
        ],
      },
      {
        slug: "beyond-cnns",
        title: "Beyond CNNs",
        summary:
          "Where the convolutional backbone turns into decisions: detection heads, segmentation masks and attention-based vision.",
        kind: "track",
        status: "planned",
        children: [
          {
            slug: "detection-heads",
            title: "Detection Inspector",
            summary:
              "Compare measured YOLO11n and SSDLite320 outputs on shared photographs; recompute confidence, class-aware NMS and reference-box IoU.",
            kind: "lesson",
            difficulty: "advanced",
            minutes: 10,
            status: "ready",
          },
          {
            slug: "segmentation",
            title: "Segmentation Inspector",
            summary: "Compare real pretrained tiny U-Net class/instance masks, retained color clusters and authored references; inspect touching-object failures and recompute overlap.",
            kind: "lesson",
            difficulty: "advanced",
            minutes: 10,
            status: "ready",
          },
          {
            slug: "vision-prompts",
            title: "Vision Prompt Playground",
            summary: "Compare genuine CLIP candidate scores and SAM point/box masks on fixed photos; inspect failed prompts, ambiguity and model revisions.",
            kind: "lesson",
            difficulty: "advanced",
            minutes: 15,
            status: "ready",
          },
          {
            slug: "vision-transformers",
            title: "Vision Transformers",
            summary:
              "Cut the image into patches and let self-attention learn the spatial relationships.",
            kind: "lesson",
            difficulty: "advanced",
            minutes: 20,
            status: "planned",
          },
        ],
      },
    ],
  },
  {
    id: "llm",
    label: "Large Language Models",
    short: "LLMs",
    tagline: "From next-token scores to evidence and bounded actions.",
    summary:
      "Explore next-token probabilities, retrieved evidence, tool calls and evaluation through interactive experiments.",
    icon: BrainCircuit,
    accent:
      "bg-violet-500/10 text-violet-600 ring-violet-500/20 dark:text-violet-300 dark:ring-violet-400/25",
    gradient: "from-violet-500/20 via-violet-500/5 to-transparent",
    concepts: ["Tokenisation", "Softmax", "Temperature", "Top-k / top-p"],
    nodes: [
      {
        slug: "sampling",
        title: "Next-token sampling",
        summary:
          "Real model scores show how temperature, top-k and top-p shape the next token and its continuation.",
        kind: "lesson",
        difficulty: "intro",
        minutes: 15,
        status: "ready",
      },
      {
        slug: "rag",
        title: "Retrieval-Augmented Generation",
        summary: "Find evidence before generating an answer.",
        kind: "track",
        status: "ready",
        children: [
          {
            slug: "retrieval-reranking",
            title: "RAG: retrieval and reranking",
            summary:
              "Compare question and passage vectors, follow reranking, and read the passages chosen for an answer model.",
            kind: "lesson",
            status: "ready",
            difficulty: "intermediate",
            minutes: 10,
          },
        ],
      },
      {
        slug: "evaluation",
        title: "Evaluation",
        summary: "Compare task scores and inspect disagreements.",
        kind: "track",
        status: "ready",
        children: [
          {
            slug: "metric-inspector",
            title: "Evaluation: answer quality and retrieval",
            summary:
              "Run real model answers with and without source evidence; inspect failures, score a human rating and challenge a model judge.",
            kind: "lesson",
            status: "ready",
            difficulty: "intermediate",
            minutes: 5,
          },
        ],
      },
      {
        slug: "agents",
        title: "Agent Systems",
        summary: "Inspect proposals, execution and enforced boundaries.",
        kind: "track",
        status: "ready",
        children: [
          {
            slug: "bounded-research-assistant",
            title: "Agents: tool calls and limits",
            summary:
              "A real local model proposes search and calculation calls; the application validates, executes and enforces stopping limits.",
            kind: "lesson",
            status: "ready",
            difficulty: "intermediate",
            minutes: 10,
          },
        ],
      },
      {
        slug: "tokenizer",
        title: "Tokenizer Explorer",
        summary:
          "See how raw text is chopped into sub-word tokens and why token counts drive cost and context.",
        kind: "lesson",
        difficulty: "intro",
        minutes: 9,
        status: "planned",
      },
      {
        slug: "attention",
        title: "Attention Heatmap",
        summary:
          "Follow the query/key dot products and see which earlier tokens each position attends to.",
        kind: "lesson",
        difficulty: "advanced",
        minutes: 20,
        status: "planned",
      },
      {
        slug: "embeddings",
        title: "Embedding Space",
        summary:
          "Project word vectors into 2D, then hunt for nearest neighbours and semantic directions.",
        kind: "lesson",
        difficulty: "intermediate",
        minutes: 15,
        status: "planned",
      },
    ],
  },
  {
    id: "genai",
    label: "Generative AI",
    short: "GenAI",
    tagline: "From pure noise to a finished image.",
    summary:
      "Diffusion models learn to reverse a noising process. Scrub through the steps, change the guidance scale and watch structure crystallise out of static.",
    icon: Sparkles,
    accent:
      "bg-amber-500/10 text-amber-600 ring-amber-500/20 dark:text-amber-300 dark:ring-amber-400/25",
    gradient: "from-amber-500/20 via-amber-500/5 to-transparent",
    concepts: ["Latent space", "Noise schedule", "Guidance", "Denoising"],
    nodes: [
      {
        slug: "diffusion",
        title: "Diffusion Studio",
        summary:
          "Scrub the denoising trajectory, compare noise schedules and dial the classifier-free guidance scale.",
        kind: "lesson",
        difficulty: "intermediate",
        minutes: 12,
        status: "ready",
      },
      {
        slug: "latent-space",
        title: "Latent Space Walk",
        summary:
          "Interpolate between two latents and follow how semantic attributes blend along the path.",
        kind: "lesson",
        difficulty: "intermediate",
        minutes: 14,
        status: "planned",
      },
      {
        slug: "style-transfer",
        title: "Style Transfer",
        summary:
          "Separate content from style and recombine them with a controllable style strength.",
        kind: "lesson",
        difficulty: "advanced",
        minutes: 16,
        status: "planned",
      },
      {
        slug: "prompt-control",
        title: "Prompt Conditioning",
        summary:
          "See how prompt embeddings and conditioning strength steer the generated result.",
        kind: "lesson",
        difficulty: "intro",
        minutes: 10,
        status: "planned",
      },
    ],
  },
]

export function getTheme(id: string): Theme | undefined {
  return THEMES.find((theme) => theme.id === id)
}

/** Every lesson in the tree, depth-first, tracks flattened away. */
export function flattenLessons(nodes: LabNode[]): LabNode[] {
  return nodes.flatMap((node) =>
    node.kind === "lesson" ? [node] : flattenLessons(node.children ?? [])
  )
}

/** Resolve a path like ["cnn", "convolution"] to its node. */
export function findNode(
  nodes: LabNode[],
  path: string[]
): LabNode | undefined {
  let current = nodes
  let match: LabNode | undefined

  for (const slug of path) {
    match = current.find((node) => node.slug === slug)
    if (!match) return undefined
    current = match.children ?? []
  }

  return match
}

/** Resolve a path to the chain of nodes leading to it, for breadcrumbs. */
export function nodeTrail(nodes: LabNode[], path: string[]): LabNode[] {
  const trail: LabNode[] = []
  let current = nodes

  for (const slug of path) {
    const match = current.find((node) => node.slug === slug)
    if (!match) break
    trail.push(match)
    current = match.children ?? []
  }

  return trail
}

export function nodeHref(themeId: string, path: string[]): string {
  return path.length === 0
    ? `/lab/${themeId}`
    : `/lab/${themeId}/${path.join("/")}`
}

export function themeLessons(theme: Theme): LabNode[] {
  return flattenLessons(theme.nodes)
}

export function readyLessons(theme: Theme): LabNode[] {
  return themeLessons(theme).filter((node) => node.status === "ready")
}

export function plannedLessons(theme: Theme): LabNode[] {
  return themeLessons(theme).filter((node) => node.status === "planned")
}

export function totalReadyExperiments(): number {
  return THEMES.reduce(
    (total, theme) => total + readyLessons(theme).length,
    0
  )
}

export function totalExperiments(): number {
  return THEMES.reduce(
    (total, theme) => total + themeLessons(theme).length,
    0
  )
}
