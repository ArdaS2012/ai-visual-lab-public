# Third-party notices

The root MIT license applies to original application code and documentation. It does not relicense the assets below. Dependency licenses remain in their installed packages. Preserve upstream copyright and attribution notices when redistributing.

## CNN example images

Files in `public/images/cnn/` were converted to PNG for this lab. Credits and source links also appear next to image selection in the app (`src/lib/cnn-images.ts`).

| File | Creator / source | Terms |
| --- | --- | --- |
| `zone-plate.png` | Vierge Marie, [Zone Plate 128](https://commons.wikimedia.org/wiki/File:Zone_Plate_128.png) | Public domain |
| `usaf-1951.png` | Alemily, [1951 USAF target](https://commons.wikimedia.org/wiki/File:1951usaf_test_target.jpg) | CC BY-SA 2.5 |
| `face.png` | Wilfredor, [Boy Face from Venezuela](https://commons.wikimedia.org/wiki/File:Boy_Face_from_Venezuela.jpg) | CC0 |
| `cat-on-snow.png` | Von.grzanka, [Felis catus, cat on snow](https://commons.wikimedia.org/wiki/File:Felis_catus-cat_on_snow.jpg) | CC BY-SA 3.0 |
| `stop-sign.png` | International Driving Authority, [Stop road sign in Italy](https://commons.wikimedia.org/wiki/File:Stop_road_sign_in_Italy.jpg) | CC BY-SA 4.0 |

Corresponding license texts are in `LICENSES/`. Share-alike terms apply to adaptations of those images, not a replacement MIT grant. CC0/public-domain copyright status does not grant personality or trademark rights.

## Detection and vision-prompt photographs

`public/data/cv-day2/{bus,zidane}.jpg` and their Day 3 copies originate from the [Ultralytics assets directory](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/assets). They are redistributed with the upstream repository's [AGPL-3.0 license](LICENSES/AGPL-3.0-only.txt), not this project's MIT license. The Day 3 `bus-*.png` and `zidane-*.png` overlays are lab-produced adaptations of those photographs and retain that license. The complete editable app source and stored inference data are in this repository. Preserve this notice and the upstream terms with copies. No endorsement by depicted people is implied.

YOLO11n detections are recorded outputs; the public lab does not bundle or execute Ultralytics code or YOLO model weights. SSDLite320 detections are also recorded outputs. Model identity, settings and provenance are retained in `src/lib/data/cv-detection.json` and `public/data/cv-day2/detections.json`. Upstream references: [YOLO11](https://docs.ultralytics.com/models/yolo11/), [torchvision SSDLite](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.ssdlite320_mobilenet_v3_large.html).

## Bundled numeric assets and weights

- `public/models/mnist-cnn.json`: converted [ONNX Model Zoo MNIST-12](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist), originally trained with CNTK. Its model card specifies MIT. Copyright remains with the upstream contributors; the MIT permission and disclaimer are reproduced in the root license, with attribution here. Conversion stores float32 weights as base64; this public release corrects an earlier Apache label to the model-specific MIT declaration.
- `public/models/mnist-samples.json`: handwritten digit examples from MNIST (Yann LeCun, Corinna Cortes and Christopher J. C. Burges), with lab-generated inference checks. [Original dataset](https://yann.lecun.com/exdb/mnist/). Dataset rights are separate from application code.
- `src/lib/data/cv-projection.json`: normalized digit data from scikit-learn `load_digits` (UCI Optical Recognition of Handwritten Digits; E. Alpaydin and C. Kaynak). PCA/t-SNE/UMAP fits were computed for this lab. [scikit-learn dataset documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html). Scikit-learn's BSD notice is retained in `LICENSES/BSD-3-Clause.txt`.
- Synthetic geometric scenes, workshop catalogues, references and experiment code were authored for this lab. Recorded model answers are fallible evidence, not reference truth.

## Models downloaded separately

These weights are not committed here. Their own model licenses apply when downloaded:

| Model | Use | Upstream license |
| --- | --- | --- |
| [HuggingFaceTB/SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Browser sampling | Apache-2.0 |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Local retrieval | Apache-2.0 |
| [mixedbread-ai/mxbai-rerank-xsmall-v1](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1) | Local reranking | Apache-2.0 |
| [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) | Local agent/evaluation | Apache-2.0 |

Recorded vision-prompt results were computed with [OpenAI CLIP](https://github.com/openai/CLIP) (MIT) and [Segment Anything](https://github.com/facebookresearch/segment-anything) (Apache-2.0); no CLIP/SAM weights are bundled. Exact model revisions are retained in `public/data/cv-day3/prompts.json`.

Geist fonts are fetched by Next.js at build time and distributed under the SIL Open Font License. Framework, icon, UI and inference dependencies retain their package licenses.

`src/app/shadcn.css` is the unmodified Tailwind helper stylesheet from shadcn 4.21.0 (copyright shadcn), vendored to avoid installing the unrelated CLI in the app. Its MIT notice is in `LICENSES/shadcn-MIT.txt`.
