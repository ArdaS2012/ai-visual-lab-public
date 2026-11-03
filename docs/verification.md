# Public release verification — 3 November 2026

This edition preserves all 22 original lesson entries (13 ready experiments and 9 planned lessons). Private course-slide mappings, local development skills, feedback journals and historical machine logs are omitted. Numerical demo code, model identities and prepared results are retained; the bundled MNIST provenance label is corrected to its model-specific MIT license.

## Verified locally on Linux

- Fresh dependency installation from the public edition's lockfiles.
- `pnpm check`: 40 Node tests, route/type generation, TypeScript, ESLint and production build pass.
- Python runtime: all 13 unit tests pass under Transformers 5.22.1 and the updated locked dependencies.
- `pnpm audit --prod`: zero advisories at verification time.
- Exported locked runtime packages checked with pip-audit: zero known vulnerabilities at verification time. Databases change; these are dated checks, not an ongoing guarantee.
- Production browser: SmolLM2 q8 model loaded from its pinned revision, actual logits visible, one continuation token generated, and 1,000 fixed-prefix draws observed. At that checked prefix the leading token had 42.76% probability and appeared 451/1,000 times.
- Trained MNIST classifier: loaded a bundled zero; actual browser inference predicted zero, displayed as 100% after rounding. Desktop light/dark views and a 390px-wide layout were inspected.
- Retrieval through the production web bridge: actual MiniLM embeddings and Mixedbread reranking completed, selecting the Cedar workshop for the registration-deadline question.
- Live Qwen agent through the production bridge: searched the report, requested ratio(18,24), then returned 75% with the source ID. The run completed with three model attempts and two tool attempts.
- Same-model judge through the production bridge: completed a real judgement of a catalogue answer. The judge remains fallible.
- Evaluation runtime: all eight live A/B answers to the four catalogue cases completed. Generated answers remain fallible; successful execution does not mean all answers are correct.
- Cross-origin experiment POST returned HTTP 403 in the production build. Unit tests also verify JSON-only submissions and the 16 KiB streaming body limit.

## Boundaries

This verifies the local educational application, not a hosted multi-user service. Windows/macOS runtime installation, every model failure mode, full offline operation and every possible interactive path have not been verified. Large model downloads are intentional and are documented in the README. GitHub CI is configured to rerun checks without downloading model weights; its run status is separate from these local observations.

## Local browser origin correction — 3 November 2026

A real browser POST exposed a gap in the initial verification: Next.js can reconstruct the request URL with an internal hostname, so comparing Origin against that URL rejected legitimate local users. The bridge now compares against the browser-addressed Host authority and still rejects foreign origins, protocol/port mismatches and forged forwarded-host headers. Added three regression tests; all 43 web tests, type checking, lint and the production build pass. The original command-line inference checks lacked an Origin header and therefore did not verify this browser boundary.
