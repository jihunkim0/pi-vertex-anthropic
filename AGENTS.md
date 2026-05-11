# AGENTS.md

Cross-agent configuration for `pi-vertex-anthropic`. Anything in this file is
expected to be read by every agent (Claude Code, OpenCode, Codex, Cursor,
Aider, Continue, etc.) before it acts on this repo.

This is a small TypeScript extension for the Pi coding agent that routes
Claude requests through Google Cloud Vertex AI.

## Agent skills

### Issue tracker

GitHub issues on `jihunkim0/pi-vertex-anthropic` (origin). Use the `gh` CLI;
it auto-targets `origin` when run inside this clone. See `docs/agents/issue-tracker.md`.

### Triage labels

Five canonical roles, all using the default label strings (`needs-triage`,
`needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`). See
`docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` and one `docs/adr/` at the repo root, both
created lazily by `grill-with-docs` when terminology or decisions actually
get pinned down. See `docs/agents/domain.md`.
