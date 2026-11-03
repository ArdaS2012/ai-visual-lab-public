# Day 2 local model runtime

This separate CPU service powers the retrieval, agent and evaluation Visual Lab lessons. It uses its own
uv environment and lockfile; it does not depend on the student exercises.

From the Visual Lab root, preload once (downloads public model assets):

```bash
uv sync --project runtime --locked
HF_HUB_DISABLE_XET=1 uv run --project runtime --locked python runtime/preload.py
```

Start the cached-only service, then the lab in another terminal:

```bash
uv run --project runtime --locked python runtime/server.py
pnpm dev --hostname 127.0.0.1 --port 3001
```

The service binds only to `127.0.0.1:8010`. The Next.js same-origin bridge is
`/api/day2/{health,retrieval,agent,evaluation,evaluation-judge}`. Keep the local user app loopback-bound: this
classroom service has no multi-user authentication, and concurrent model runs
return 429. For online teaching, screen-share the local user browser. Public hosting
and participant-device access are separate work.

Inference never downloads or calls a remote model provider. Preload fetches pinned
public files from Hugging Face; later runs use cached weights. Queries and traces
stay on the local user host/browser until the user exports a JSON file. First startup
needs approximately 3.2 GB of model files and several GB of CPU RAM, plus the Python
environment. The model/cache directories are ignored. All three model cards
specify Apache-2.0. Exact revisions are in `models.json`, packages in `uv.lock`.

## Mechanisms

Retrieval: real MiniLM mask-aware mean pooling, L2 normalization, exact 384D cosine,
then real `mixedbread-ai/mxbai-rerank-xsmall-v1` query/passage pair logits, pinned
to revision `b5c6e9da73abc3711f593f705371cdbe9e0fe422`. Inputs are truncated at 256/512 model
tokens, with original token counts exposed. No generation is claimed by this demo.
Fixed whitespace-word chunks preserve half-open word offsets and corpus hash.

The reranker upgrade was compared with the previous TinyBERT and two MiniLM
candidates using identical retrieved passages. The selected model improves some workshop retrieval cases; relevance is not proof of factual support. Its cached weight
file is approximately 135 MiB (70.8 million parameters); inference loads float32
on the CPU, so file size is not a RAM estimate. Current model cards and revisions
remain available in the lab's closed provenance section. Earlier verification
records explicitly retain their original TinyBERT results.

Agent: Qwen2.5-1.5B-Instruct uses its native tool-aware chat template and real greedy
CPU float32 decoding. Naive dynamic-int8 quantization was rejected after an actual
failed run; the selected model uses its original weights converted to float32.
Allow approximately 6.5 GB of model-service RAM on this host. A LangChain BaseChatModel adapter exposes AIMessage tool calls; two
registered StructuredTools implement scoped lexical search and direct arithmetic.
The application owns a serial loop rather than a generic create_agent runtime,
so every admission, denial and execution boundary is visible. Live permissions
are phase-dependent: search first, calculator after a search observation, then
no further tools after a successful calculation. The model still chooses whether
to call a permitted tool or answer, and extracts its arguments; this is a narrow
guided task rather than open-ended autonomous planning. LlamaIndex stays a
teaching comparison; there is no second implementation. The task uses explicit evidence-first instructions, including recorded post-search
and post-calculation instructions. The model still selects arguments and can generate
invalid proposals, skip a requested tool or give inaccurate answers. Such failures stay visible.

Live model requests reserve the complete formatted prompt plus 160 output tokens;
actual prompt and generated token IDs are charged after inference. Model attempts,
all tool attempts (including denied ones), wall time and token budgets are enforced.
A 4,096-token classroom context cap is separate from model capacity. Cancellation
and deadlines are checked between generated tokens and around tool execution;
loading and a single CPU forward pass cannot be interrupted immediately. Late
results are discarded. A cancelled browser retains its partial trace with an
explicit client status, rather than fabricating a server terminal event.

Controller-test modes do not invoke a model and charge no invented model tokens.
Their simulated decision attempts are counted explicitly. The bundled `public/traces/day2-study-a.json` is a witnessed real run; the UI
labels saved loading/replay separately and preserves original model metadata.
Replay walks recorded events only. Pinning a trace preserves its original settings and immutable events.

## Verification

```bash
uv run --project runtime --locked python -m unittest discover -s runtime -p 'test_*.py' -v
pnpm test
pnpm exec tsc --noEmit
pnpm build
```

See `../docs/verification.md` for public-release verification and limitations.

## Real evaluation

Evaluation reuses the cached pinned Qwen model with its ordinary chat template and greedy CPU generation. Four fixed fictional workshop questions are answered with and without source context (eight outputs); references never enter answer prompts. B can omit Cedar, and both variants can request JSON or plain text. The benchmark, source context, messages, raw outputs, token counts, elapsed seconds, revision and decoding are retained. Output is capped at 100 new tokens per answer, 160 per judge; the complete evaluation has a 240-second cooperative deadline, judge 90 seconds. Cancellation/deadlines are checked between tokens; model loading and a CPU forward pass are not immediately interruptible. Truncated answers remain labelled and fail scoring.

A separate request judges one selected response against the catalogue/reference and rubric. It uses the same model, so is not an independent ground truth. The UI requires a prior human rating and retains invalid judge output. The bundled `public/traces/evaluation-workshops.json` is a recorded real run, explicitly distinct from live generation. The earlier fixture-only calculator remains available without this service. See the public release verification record.
