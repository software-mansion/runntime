---
name: pr-review
description: How to review a change in this repo, as a reviewer or before opening a pull request. Use whenever asked to review a diff, branch or pull request, to address review comments, or before finishing work that will be reviewed.
---

# Review

Find what breaks. Each finding names the line that breaks and the input
that breaks it.

## Rules

1. **Evidence is `file:line`.** Open the code before you say anything about
   it. "`trim()` probably leaks" is a guess from a name. "`bufferPool.ts:88`
   returns before `release`" is a finding. A claim with no line is not a
   finding.
2. **Test your assumptions.** If a finding depends on a caller, a route
   threshold or a feature check, read that code or run it, and say which
   you did. If you can't check it, label it `question`.
3. **Reproduce a bug before you report it, when that is cheap.** If a unit
   test, a `node` one-liner or a small edit plus a test run can show the
   bug in a few minutes, do it, and put the command and its output in the
   finding. If it doesn't reproduce, drop the finding. A `bug` you could
   have run and didn't is not a finding. If reproducing needs a GPU you
   don't have, a model download or a long run, say so and why.
4. **Say how likely the input is.** Every `bug` gets one of: `common` (real
   use hits it), `rare` (a real but unusual input: an odd shape, a slow
   network, a device without a feature), `practically never` (only a typo,
   a malformed file nobody ships, a buffer past what any model uses). Put
   it on the finding's first line, so a `practically never` can be skipped
   at a glance.
5. **Hold the author to the house rules.** Anything that breaks `AGENTS.md`,
   `code-style` or `testing` is a finding. Quote the rule. "It matches the
   file around it" doesn't excuse breaking one.
6. **Label nits.** They go after the real findings and are labelled
   `nit`. Skip anything Prettier, ESLint or tsc already catch.
7. **Run what CI runs:** `pnpm typecheck`, `pnpm lint`, `pnpm format:check`,
   `pnpm --filter runntime test:unit`. For changes in `src/core/kernels` or
   `src/core/gpu`, also run `test:gpu`. CI has no GPU, so nobody else will.
8. **Flag assumptions about the data format.** Code that only works for
   one tensor layout, dtype, key name or file format, with no check that
   the data matches, is a finding. So is any `core` API that makes the
   user know how the weights are laid out: if a `core` user has to write
   their own `transformStateDict` to load a normal checkpoint, the layer
   should adapt the layout itself.

## What matters, by area

Read the section for each folder the diff touches.

**`src/core/graph`**: ops validate shapes and return a `Value`.
Failure: a `createBuffer`, an encode or a submit inside `graph/`.

**`src/core/gpu`**: pipeline keys hold op, dtype and load-time config only.
Pooled buffers are never destroyed by hand. Every `tensor()` / `uploadF32`
has an owner that destroys it.
Failure: a shader compiles for every new input length. A "destroyed buffer
used in submit" validation error. Memory grows across calls.

**`src/core/kernels`**: `device.features` is checked before `subgroups` /
`shader-f16` are used, and the fallback route exists. New elementwise work
goes into an existing epilogue. The PR carries an A/B table.
Failure: a crash on a device without subgroups. Wrong output past 2^24
elements. A dispatch count that goes up with no reason in the PR. A
nonzero last A/B column on a kernel the change didn't mean to touch.

**`src/core/nn`**: built-in layers load the checkpoint's own layout. Any
reshape, transpose, split or fuse happens in the layer's
`transformStateDict`, not in the caller.
Failure: a docs sample or test that has to rewrite the state dict before
`loadStateDict` works.

**`src/core/weights`**: data that arrives as `unknown` is checked, not cast.
Failure: a bad header throws a `TypeError` instead of `CHECKPOINT_MISMATCH`.

**`src/zoo`**: one executor per model. Calls go through the promise queue.
`dispose()` frees everything, even after a half-finished build
(`createResourceScope`). Variable-length runners call `trim()`. User-facing
throws are `RunntimeError` with a code. A new model has a parity manifest.
Module and field names match the checkpoint's tensor names, so keys load
as they are, with no renaming.
Failure: two overlapping calls corrupt each other. A key-renaming map or
`transformStateDict` that only exists because a module is named
differently from the checkpoint. Buffers survive
`dispose()`. A plain `Error` reaches the app. `coverage.test.ts` fails.

**`test/`**: expected values come from the math or PyTorch, never from our
own output. A bug fix adds a case that fails without it. Each route is
covered on both sides of its threshold, with the route comment. Missing
features skip with a reason.
Failure: a golden value copied from the kernel's output. A test that passes
silently on a machine without f16.

**`apps/docs`**: samples are twoslash. Benchmark numbers come from measured
runs. Pages show only what users act on. A new error code goes in the
error table.
Failure: the docs build fails. A number that no run produced.

**Repo-wide**: every export in `core` and `zoo` has a doc comment. A changed
signature or default in either package says so in the PR and in the commit
(`!` or `BREAKING CHANGE`). The lockfile follows dependency changes. The
commit type and scope match the diff.
Failure: `--frozen-lockfile` fails in CI. A `fix` that also renames things.

## Walkthrough

Before the findings, write the walkthrough. Treat `core` as a public
package, just like `zoo`: an export a user can import is API.

1. **What it introduces.** Two or three sentences in plain words: what a
   user or a model can do after this PR that they couldn't before.
   Name the new or changed exports with `file:line`.
2. **Design decisions.** Each choice the author made where another option
   existed: what they picked, what they could have picked, and why, if the
   PR or the code says. If the reason isn't written anywhere, say so.
3. **Effect on users.** Answer separately for each package:
   - `core`: new or changed exports, signatures, defaults, errors thrown,
     anything that breaks existing callers.
   - `zoo`: what changes for someone calling `create<Task>()`: load time,
     speed, memory, output, errors, devices that stop or start working.

   Say "no change" when there is none. Don't leave a package out.

4. **Where it falls short.** What the PR doesn't cover: missing cases,
   routes or devices it doesn't handle, docs not updated, numbers not
   measured. These are gaps, not bugs. A gap that breaks something is
   also a finding.

Use evidence from rule 1 here too: every claim points to a `file:line`
or a command you ran.

## Report

First the walkthrough, then one block per finding, most severe first:

- `bug` / `rule` / `question` / `nit`, and for a `bug` how likely it is
- `file:line`
- what goes wrong, and on which input
- the rule broken, if any
- how you checked it: read `x.ts:40`, or ran a command, with its output
  for a reproduced `bug`

Findings go to whoever asked. Nothing is posted to GitHub.

## Addressing a review

Fix the code, or reply in one sentence with the input that shows the
comment is wrong. One new commit per round, no amending.
