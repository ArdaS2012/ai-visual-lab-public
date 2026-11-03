export type CnnImage = {
  id: string
  label: string
  src: string
  alt: string
  credit: string
  license: string
  sourceUrl: string
  /** Why this picture is worth convolving — shown next to the preview. */
  note: string
}

/**
 * Five real photographs picked so that the convolution playground stays
 * readable after it is downsampled to a 12×12 grid: each one has a clear
 * silhouette, high contrast or a strong frequency structure.
 *
 * All files are Wikimedia Commons assets in the public domain or under a
 * CC licence; see `credit`/`license` for attribution.
 */
export const CNN_IMAGES: CnnImage[] = [
  {
    id: "zone-plate",
    label: "Zone plate",
    src: "/images/cnn/zone-plate.png",
    alt: "Zone plate test pattern of concentric rings",
    credit: "Vierge Marie · Wikimedia Commons",
    license: "Public domain",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Zone_Plate_128.png",
    note: "Rings whose spacing tightens towards the edge — the cleanest way to watch blur and edge filters react to frequency, and to see aliasing when it is downsampled.",
  },
  {
    id: "usaf-1951",
    label: "USAF 1951 target",
    src: "/images/cnn/usaf-1951.png",
    alt: "USAF 1951 resolving power test target",
    credit: "Alemily · Wikimedia Commons",
    license: "CC BY-SA 2.5",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:1951usaf_test_target.jpg",
    note: "Bars of increasing frequency in three orientations — a direct read-out for directional kernels such as Sobel-X and Sobel-Y.",
  },
  {
    id: "face",
    label: "Portrait",
    src: "/images/cnn/face.png",
    alt: "Close-up portrait of a boy's face",
    credit: "Wilfredor · Wikimedia Commons",
    license: "CC0",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Boy_Face_from_Venezuela.jpg",
    note: "Smooth skin gradients plus fine edges along the eyes, brow and lips — the kind of input an image classifier actually sees.",
  },
  {
    id: "cat-on-snow",
    label: "Cat on snow",
    src: "/images/cnn/cat-on-snow.png",
    alt: "Tabby cat standing on snow",
    credit: "Von.grzanka · Wikimedia Commons",
    license: "CC BY-SA 3.0",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg",
    note: "A dark animal silhouette against a bright, low-contrast background — good for separating object edges from surface texture.",
  },
  {
    id: "stop-sign",
    label: "Stop sign",
    src: "/images/cnn/stop-sign.png",
    alt: "Red stop sign among green foliage",
    credit: "International Driving Authority · Wikimedia Commons",
    license: "CC BY-SA 4.0",
    sourceUrl: "https://commons.wikimedia.org/wiki/File:Stop_road_sign_in_Italy.jpg",
    note: "Saturated red against green foliage with white lettering — shows how the R, G and B channels and colour-contrast filters behave on real text.",
  },
]

export const DEFAULT_CNN_IMAGE_ID = CNN_IMAGES[0].id

export function findCnnImage(id: string): CnnImage | undefined {
  return CNN_IMAGES.find((image) => image.id === id)
}

/** Decode an image from `public/` and resolve once its bitmap is ready. */
export function loadImageElement(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new window.Image()
    image.decoding = "async"
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error(`Could not load image: ${src}`))
    image.src = src
  })
}

/**
 * Downsample with repeated halving before the final draw. A single
 * `drawImage` from 256 px to 12 px lets the browser skip most source pixels,
 * which turns fine detail into noise; halving keeps the averages honest.
 */
function halfScale(
  image: HTMLImageElement,
  size: number
): { source: CanvasImageSource; width: number; height: number } {
  let source: CanvasImageSource = image
  let width = image.naturalWidth
  let height = image.naturalHeight

  while (width > size * 2 && height > size * 2) {
    const nextWidth = Math.max(size, Math.floor(width / 2))
    const nextHeight = Math.max(size, Math.floor(height / 2))
    const canvas = document.createElement("canvas")
    canvas.width = nextWidth
    canvas.height = nextHeight
    const context = canvas.getContext("2d")
    if (!context) break
    context.imageSmoothingEnabled = true
    context.imageSmoothingQuality = "high"
    context.drawImage(source, 0, 0, width, height, 0, 0, nextWidth, nextHeight)
    source = canvas
    width = nextWidth
    height = nextHeight
  }

  return { source, width, height }
}

/**
 * Sample an image into the `[rows][cols][channels]` grid the convolution
 * playground works on. Photo values are normalised to 0…1; greyscale uses
 * Rec. 709 luminance so it matches the rest of the demo.
 */
export function imageToGrid(
  image: HTMLImageElement,
  size: number,
  channels: 1 | 3
): number[][][] {
  const { source, width, height } = halfScale(image, size)

  const canvas = document.createElement("canvas")
  canvas.width = size
  canvas.height = size
  const context = canvas.getContext("2d", { willReadFrequently: true })
  if (!context) return []

  context.imageSmoothingEnabled = true
  context.imageSmoothingQuality = "high"
  context.drawImage(source, 0, 0, width, height, 0, 0, size, size)
  const { data } = context.getImageData(0, 0, size, size)

  return Array.from({ length: size }, (_, row) =>
    Array.from({ length: size }, (_, col) => {
      const offset = (row * size + col) * 4
      const r = data[offset] / 255
      const g = data[offset + 1] / 255
      const b = data[offset + 2] / 255
      if (channels === 1) {
        return [0.2126 * r + 0.7152 * g + 0.0722 * b]
      }
      return [r, g, b]
    })
  )
}
