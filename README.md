# AI Visual Lab

Interactive experiments for learning computer vision and language models. Change a control, inspect the actual computation, and compare the evidence. Built by Arda Sener.

## Run locally

Install **Node.js 22.18+ or 24**, and **pnpm 10.33.2**. A current Chrome, Edge or Firefox desktop browser is recommended. Mobile layouts are supported, but local language models need desktop-class memory.

```sh
git clone https://github.com/ArdaS2012/ai-visual-lab-public.git
cd ai-visual-lab-public
npm install --global pnpm@10.33.2
pnpm install --frozen-lockfile
pnpm dev
```

Open **http://127.0.0.1:3001/lab**. No account or paid API key is needed. Stop with Ctrl+C. Keep the app bound to your own machine; a downloadable public repository is different from a public multi-user inference service.

To use a production build locally:

```sh
pnpm build
pnpm start
```

## What works where

| Experiment | Computation | Extra setup |
| --- | --- | --- |
| Image filters, convolution and CNN pipeline | Browser calculations | None |
| Trained CNN classifier | Actual bundled MNIST weights, browser inference | None |
| Projection & reconstruction | Live PCA; recorded genuine t-SNE/UMAP fits | None |
| Detection, segmentation and vision prompts | Recorded genuine inference; live thresholds/overlap and inspection | None |
| Next-token sampling | Actual SmolLM2 inference in a browser worker | Click Load; approximately 365 MB first download |
| RAG retrieval and reranking | Actual local embedding and reranker models | Optional Python runtime below |
| Bounded research assistant | Actual local Qwen generation and guarded tools | Optional Python runtime below; saved trace also available |
| Evaluation | Actual local Qwen answers and judging | Optional Python runtime for live runs; recorded run and fixture calculator work without it |

Navigation marks unfinished lessons **Planned**. Recorded neural outputs are labelled; changing their inspection controls does not rerun the model. A schematic architecture diagram does not claim to show measured activations.

### Optional local Python runtime

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). From the repository root:

```sh
uv sync --project runtime --locked
uv run --project runtime --locked python runtime/preload.py
uv run --project runtime --locked python runtime/server.py
```

Keep that terminal open and run `pnpm dev` in another terminal. uv provisions Python 3.12. Preloading downloads approximately **3.2 GB** of public model weights; allow several GB of disk space for packages and about **6.5 GB RAM** for the agent model service, in addition to your browser. CPU runs can be slow. Linux CPU is the verified runtime platform; Windows/macOS runtime support is not yet verified. The browser-only experiments do not require Python.

The service listens on `127.0.0.1:8010`; the app accesses it through its same-origin API bridge. Only one model experiment runs at once. See [runtime details](runtime/README.md) for pinned models, cancellation and budgets.

## Privacy and network use

There is no application analytics or user account system. Browser sampling prompts stay in the browser; live Python experiments send their inputs to the local runtime. Traces leave your device only when you choose to share/export them. Initial model and runtime downloads contact Hugging Face/CDN providers; package installation contacts package registries. Building fetches Google Fonts. These providers receive ordinary download request metadata. First use needs an internet connection; fully offline operation is not promised.

Keep confidential information out of examples and exported traces. Generated answers can be inaccurate. This is an educational lab, not a production decision system.

## Development and contribution

```sh
pnpm check
uv run --project runtime --locked python -m unittest discover -s runtime -p 'test_*.py' -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md), [SECURITY.md](SECURITY.md) and [release verification](docs/verification.md). Model preloading is intentionally separate from ordinary CI. Projection-data regeneration has its own locked environment in `tools/projection/`.

## License and attribution

Original application code and documentation are **MIT licensed**; see [LICENSE](LICENSE). Third-party photographs, datasets, converted weights and downloaded models retain their own terms. In particular, the Ultralytics-sourced example images are not covered by this project's MIT grant. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the bundled license texts before redistributing assets.
