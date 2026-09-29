---
name: code-style
description: How to write and review code, comments and docs in this repo. Use whenever writing, editing or reviewing TypeScript in packages/runntime, code samples or prose in apps/docs, or before finishing any change.
---

# Code style

Write the least code that solves the task, in the style of the code around
it. How the engine works is in `AGENTS.md`; this file is about how to write.
Short bad/good pairs are in [examples.md](examples.md).

## Scope

- Do what was asked. The smallest diff that solves it is the best one.
- Simple means no extras, not a weaker fix. When the problem needs a
  harder solution (an edge case that can happen, a new kernel, a fix in
  the right place instead of a patch at the symptom), write it.
- No abstraction, option, parameter or generic type with one caller.
  Repeating similar code is fine; make a helper when it is needed in a
  third place.
- Validate at the public API (`create<Task>()`, runner methods). Inside the
  library, trust the types and skip checks for cases that cannot happen.
- No fallbacks, "just in case" config, or code that keeps an old name or
  old behaviour working after a change. If it is not needed now, it is
  not written.
- Leave unrelated code alone: no refactors, renames or reformatting
  outside the task.
- Reuse what exists before adding: a helper in `core`, a pattern from a
  sibling task or model. A new file only when nothing fits.
- Delete dead code. Do not comment it out.

## Structure

- A new file starts with a doc comment: what the file is, one or two lines.
- A function does one thing and its name says what. Return early instead
  of nesting ifs.
- Numbers get a named constant with its unit or meaning in a doc comment,
  like `TICK_SAMPLES` or `MAX_SEGMENT_SECONDS`.
- Keep logic in pure functions: data in, result out. The code that runs
  the GPU, fetches files or holds state calls them.
- Match the file: naming, error messages, how options and defaults are
  destructured. If you see a better way, do not mix two styles: change
  every place in its own PR, or suggest it and leave it.
- Keep a concept in one place. Two copies of the same logic drift apart.

## Comments

- Describe the code as it is now. No history: "changed", "previously",
  "now uses", "new", "fixed", "no longer". History goes in the commit.
- Say why, not what. A comment that repeats the code is deleted.
- Short: one line if it fits. Plain words, and the exact term when it is the
  right one (EOS, KV cache, subgroup, letterbox).
- Every export gets a doc comment: what it does, units, defaults, what it
  returns or throws. Internal code only where the reason is not obvious.
- No filler: "Note:", "IMPORTANT", "This function...", "Helper to...",
  "Simply", emojis.
- Do not over-explain. Obvious code gets no comment.

## Docs (apps/docs)

- Same rules for prose: short sentences, plain words, present tense.
- Say what the user gets and how to use it. Implementation details only when
  they change what the user does.
- Code samples are the shortest code that runs. Steps done in order are
  numbered (`// 1. Load the model once.`). Comments say what matters in the step,
  not the syntax.
- Show the expected output in a comment: `// 0.68, related`.
- Give formats, units and numbers, not vague words: "mono samples at
  16 kHz", "every half second", not "audio data", "regularly".
- Benchmark numbers come from measured runs only.

## Before finishing

Read your own diff and check:

1. Can any line, parameter, option or file be deleted and the task still
   works?
2. Does any new helper or type have one caller?
3. Does any comment talk about the past or repeat the code?
4. Does it look like the code next to it?

Fix what you find, then finish.
