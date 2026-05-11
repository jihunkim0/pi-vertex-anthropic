# Issue tracker: GitHub

Issues and PRDs for this repo live as GitHub issues on
`jihunkim0/pi-vertex-anthropic` (the `origin` remote). Use the `gh` CLI for
all operations.

## Conventions

- **Create an issue**: `gh issue create --title "..." --body "..."`. Use a
  heredoc for multi-line bodies.
- **Read an issue**: `gh issue view <number> --comments`, filtering comments
  by `jq` and also fetching labels.
- **List issues**: `gh issue list --state open --json number,title,body,labels,comments --jq '[.[] | {number, title, body, labels: [.labels[].name], comments: [.comments[].body]}]'`
  with appropriate `--label` and `--state` filters.
- **Comment on an issue**: `gh issue comment <number> --body "..."`
- **Apply / remove labels**: `gh issue edit <number> --add-label "..."` /
  `--remove-label "..."`
- **Close**: `gh issue close <number> --comment "..."`

`gh` auto-targets `origin` (i.e. `jihunkim0/pi-vertex-anthropic`) when run
inside this clone.

## Note on the upstream fork

This repo is a fork of `skyfallsin/pi-vertex-anthropic`. To file or read
issues against the canonical upstream instead of this fork, pass
`-R skyfallsin/pi-vertex-anthropic` to every `gh` call. By default, all
skill operations target this fork.

## When a skill says "publish to the issue tracker"

Create a GitHub issue on `jihunkim0/pi-vertex-anthropic`.

## When a skill says "fetch the relevant ticket"

Run `gh issue view <number> --comments`.
