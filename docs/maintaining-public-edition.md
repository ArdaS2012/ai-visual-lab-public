# Maintaining the public edition

Keep this repository's Git history independent from private working material. Transfer only reviewed application, test, runtime and public documentation changes. Never merge an unrelated private repository history into this public repository.

Before a release, inspect the full diff for secrets, personal data, local paths, course-only links and missing asset notices. Preserve all recorded-versus-live labels and model provenance. Update lockfiles deliberately, run `pnpm check` and runtime unit tests, audit dependencies, and witness the affected real model path. Update `docs/verification.md` with dated observations and honest limitations. Use a GitHub noreply address for commits. Publish changes through a pull request once the repository has its initial public snapshot.
