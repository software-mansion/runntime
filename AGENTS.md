# AGENTS.md

## What this is

ruNNtime runs neural networks on the user's GPU through WebGPU, in
TypeScript on [TypeGPU](https://docs.swmansion.com/TypeGPU/). pnpm monorepo,
Node 22 or newer.

- `packages/runntime/src/zoo` is what users call: one `create<Task>()` runner
  per task, the Hub weight registry in `models.ts`, and the transformers.js
  plugin in `transformers/`.
- `packages/runntime/src/core` is the engine. Treat it as a public API, like
  `src/zoo`: its exports get doc comments, and a changed signature or default
  is a breaking change.
- `apps/docs` is the only user-facing documentation. Read `apps/docs/AGENTS.md`
  before working there.
- Edit the root `README.md`, never the package copy:
  `scripts/sync-readme.mjs` generates it on prepack.

A change is verified when its tests pass. Before writing, running or
reviewing a test, adding a kernel route, fixing a bug or benchmarking, read
[.claude/skills/testing/SKILL.md](.claude/skills/testing/SKILL.md).

Before writing or reviewing code, comments or docs, read
[.claude/skills/code-style/SKILL.md](.claude/skills/code-style/SKILL.md).

## Why it is built this way

- Ops never touch the GPU. An op like `matmul(a, b)` validates shapes and
  returns a pending `Value`. `toArray` / `evalValues` walk the graph, encode
  every dispatch into one pass and submit once.
- Never put an input dim in a pipeline key. Key by op, dtype and load-time
  config, and pass runtime-variable dims in uniforms, so a new input length
  never compiles a shader.
- Never destroy a pooled buffer by hand: intermediates go back to the
  executor's pool after their last consumer is dispatched. Uploads made with
  `tensor()` or `uploadF32` are outside the pool, and whoever made them
  destroys them.
- A model whose input length varies gets its own executor and calls `trim()`
  after each call. The pool keeps free buffers per exact size, so without
  `trim()` it holds a set for every length it has seen.
- Create one `gpuExecutor` per model, once. Pooled buffers and memoized bind
  groups only pay off across calls.
- Check `device.features` before using `subgroups` or `shader-f16`. The user's
  `tgpu.init` (and the plugin's `initRunntimeBackend`) requests both as
  optional. Without `subgroups` the fast small-m matmul and attention routes
  are gone, without `shader-f16` the conv-based tasks refuse to load.
- Put a new op into an existing kernel's epilogue before giving it its own
  dispatch: dispatch count is the cost model. Elementwise ops fuse into their
  consumer, matmul carries bias, activation and addend in its epilogue, decode
  runs bursts of tokens per submit and readback rides the same submit.
- Task runners serialize calls on a promise queue and free their GPU
  resources in `dispose()`.

## How to build

- `pnpm install` installs deps and the Git hooks (Lefthook: Prettier and
  ESLint on staged files, `tsc` on the library, commitlint on the message).
- `pnpm typecheck`, `pnpm lint`, `pnpm format`.
- `pnpm --filter runntime build`: tsdown, output in `dist`. Kernel files carry
  `'use gpu'` and `unplugin-typegpu` embeds their WGSL, so consumers need no
  build plugin. `typegpu`, `@huggingface/transformers`,
  `@huggingface/tokenizers` and `js-tiktoken` stay external.
- Change dependencies with `pnpm add` / `pnpm install` so `pnpm-lock.yaml`
  follows. CI installs with `--frozen-lockfile` and fails on a stale lockfile.
- Commits follow Conventional Commits (`feat`, `fix`, `perf`, `refactor`,
  `docs`, `chore`, scopes `core`, `zoo`, `docs`), header up to 120
  characters. CI runs the docs build on every pull request that touches
  `apps/docs` or `packages/runntime`.

## How to run tests

- `pnpm --filter runntime test:unit` runs the Node tests,
  `pnpm --filter runntime test:gpu` the GPU tests (local only, CI has no GPU),
  `pnpm test` both. Tests live in `packages/runntime/test`, mirroring `src`.
- Once per machine, before the GPU tests: `pnpm --filter runntime exec
playwright install chromium`. Without it Vitest cannot start the browser.
