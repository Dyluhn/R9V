# PR template

Copy this into the PR body and fill every section. Leave a section as "none" rather than deleting it; the acceptance tooling expects the headings.

```markdown
## Card
<id> — <card title>   (kind: API | implementation; GPU: yes | no; size: S | M | L)

## Spec sections implemented
- spec <n> §<x>: <one line on what this PR does for it>
- ...

## Deliverables
- [ ] <each deliverable from the card, checked when present>

## Done-when tests
| test | location | tier | status |
|---|---|---|---|
| <name from the card> | <path> | cpu-only / gpu | green / pending runner |

## Decisions
<every `DECISION(<id>)` comment in the diff, one per line, with the file:line>
- none

## SPEC-ISSUES filed
- SI-<n>: <one line>
- none

## New dependencies
- <crate>: <why the workspace didn't already have this>
- none

## Hardware
Tests run here: <hosted cpu-only | stub device | compile-only | runner>
Tests pending on the runner: <list or none>
Measured numbers (if any): fingerprint <…>, command <…>, receipt <path or "none">

## Checklist
Walked `references/acceptance-checklist.md`. Items answered "no" and why:
- none

## Size check
Diff size vs card size class: <lines> vs <S|M|L> — <ok | explanation>
```
