# AGENTS.md

## What this is

ruNNtime runs neural networks on the user's own GPU through WebGPU. It is
TypeScript on [TypeGPU](https://docs.swmansion.com/TypeGPU/): kernels are
TypeGPU code compiled to WGSL at build time, models are composed from those
kernels in plain TypeScript. No ONNX, no WASM, no export step. Published to
npm as `runntime`, documented at
[docs.swmansion.com/runntime](https://docs.swmansion.com/runntime).

pnpm monorepo, Node 22 or newer:

- `packages/runntime`: the library. `src/zoo` is what users call, one
  `create<Task>()` runner per task and the Hub weight registry in `models.ts`.
  `src/zoo/transformers` is the transformers.js plugin, one loader per model.
  `src/core` is the engine (graph ops, kernels, dispatch, buffer pool, weight
  loading). Only `runntime/zoo` and `runntime/zoo/transformers` are published.
- `apps/docs`: the Starlight docs site, the only user-facing documentation.
  See `apps/docs/AGENTS.md` before working there.
- `scripts/sync-readme.mjs`: copies the root `README.md` into the package on
  prepack. Edit the root file, the package copy is generated and gitignored.

Verification is typecheck, lint, unit tests and the docs build.

Before writing or reviewing code, comments or docs, read
[.claude/skills/code-style/SKILL.md](.claude/skills/code-style/SKILL.md).

## Why it is built this way

- Ops never touch the GPU. `matmul(a, b)` validates shapes and returns a
  pending `Value`. `toArray` / `evalValues` walk the graph, encode every
  dispatch into one pass and submit once.
- Pipelines are keyed by op, dtype and load-time config, never by input
  shape. Runtime-variable dims travel in uniforms, so a new input length never
  compiles a shader.
- Intermediate buffers come from the executor's pool and go back when their
  last consumer is dispatched. Uploads made with `tensor()` or `uploadF32`
  are outside the pool and must be destroyed by whoever made them.
- One `gpuExecutor` per model, created once. Pooled buffers and memoized bind
  groups only pay off across calls.
- The user's `tgpu.init` (and the plugin's `initRunntimeBackend`) requests
  `subgroups` and `shader-f16` as optional features. Without `subgroups` the
  fast small-m matmul and attention routes are gone, without `shader-f16` the
  conv-based tasks refuse to load. Code checks `device.features` before using
  either.
- Dispatch count is the cost model. Elementwise ops fuse into their consumer,
  matmul carries bias, activation and addend in its epilogue, decode runs
  bursts of tokens per submit and readback rides the same submit. A new op
  goes into an existing kernel's epilogue before it gets its own dispatch.
- Task runners serialize calls on a promise queue and free their GPU
  resources in `dispose()`.

## How to build and verify

- `pnpm install` installs deps and the Git hooks (Lefthook: Prettier and
  ESLint on staged files, `tsc` on the library, commitlint on the message).
- `pnpm typecheck`, `pnpm lint`, `pnpm format`.
- Once per machine, before the GPU tests: `pnpm --filter runntime exec
playwright install chromium`. Without it Vitest cannot start the browser.
- `pnpm --filter runntime build`: tsdown, output in `dist`. Kernel files carry
  `'use gpu'` and `unplugin-typegpu` embeds their WGSL, so consumers need no
  build plugin. `typegpu`, `@huggingface/transformers` and `js-tiktoken` stay
  external.
- `pnpm dev:docs` / `pnpm build:docs`. The docs and their live demos import
  `runntime/zoo` from the library's `dist`, so build the library first or run
  `pnpm --filter docs... run build`. The docs dev server does not watch the
  library: after editing `packages/runntime/src`, rebuild it.
- Docs code samples are ` ```ts twoslash `: the docs build type-checks
  them against the built library and fails on an error. Setup the reader does
  not need (`declare const canvas: HTMLCanvasElement;`) goes above
  `// ---cut---`. The API reference under `/api/` is generated from the
  exports and their doc comments by `starlight-typedoc`; do not edit it.
- Dependency changes go through `pnpm add` / `pnpm install` so `pnpm-lock.yaml`
  follows. CI installs with `--frozen-lockfile` and fails on a stale lockfile.
- Commits follow Conventional Commits (`feat`, `fix`, `perf`, `refactor`,
  `docs`, `chore`, scopes `core`, `zoo`, `docs`), header up to 120
  characters. CI runs the docs build on every pull request that touches
  `apps/docs` or `packages/runntime`. The npm publish is a manual workflow.
- The docs site is published from the `docs` branch, not from `main`. Work
  lands on `main`; merging `main` into `docs` deploys it to Pages, usually
  right after an npm release. A fix for the live site goes in a pull request
  against `docs`, then `docs` is merged back into `main`. Both of these merges
  use a merge commit, never squash, or the two branches drift apart.
- Benchmark numbers on the docs pages come from measured runs. Never edit or
  derive one by hand.

## How to write tests

- Vitest, configured in `packages/runntime/vitest.config.ts`. `pnpm test`
  runs everything, `pnpm --filter runntime test:unit` only the Node tests,
  `pnpm --filter runntime test:gpu` only the GPU tests.
- Tests live in `packages/runntime/test`, never in `src`, and the test path
  mirrors the source path: `src/core/weights/convert.ts` is tested in
  `test/core/weights/convert.test.ts`. Import the source with its `.ts`
  extension.
- `*.test.ts` runs in Node with no GPU. `*.gpu.test.ts` runs in headless
  Chromium on the machine's real GPU (Vitest browser mode, Playwright).
  `test/gpu.ts` gives it the device (`gpu()`), uploads freed when the test
  ends (`upload`), feature skips (`requireFeature`), the float check
  (`expectClose`), seeded inputs (`randomF32`) and f16 rounding for
  references (`roundF16`). CI runners have no GPU, so the GPU tests run
  locally: run them before pushing a change to a kernel or its dispatch.
- Write the test once and put the cases in a list that `it.each` runs.
  Adding a case should be one line.
- Expected values come from the math, worked out by hand or from Hugging
  Face / PyTorch, never from running our code and copying its output. A CPU
  reference for a kernel is a plain loop written from the formula, not one
  that reuses the kernel's helpers, or a bug in them passes in both.
- A bug fix adds a case that fails without the fix.
- Compare floats with a tolerance (`|got - want| <= atol + rtol * |want|`)
  and let NaN fail. For f16, round the inputs to f16 before computing the
  reference.
- No snapshots of generated WGSL. They break on every refactor and catch
  nothing.
- A test that needs a device feature the machine lacks (`shader-f16`,
  `subgroups`) skips with a reason. It never passes silently.
- A known bug that is not fixed yet is an `it.fails` case with a comment
  naming the bug. The fix turns it into a plain `it`.

### Kernels

- Every kernel has a `*.gpu.test.ts` suite whose path mirrors the kernel's
  file under `src/core/kernels/`: `kernels/matmul/matmulGemv.ts` is tested in
  `test/core/kernels/matmul/matmulGemv.gpu.test.ts`. It runs the kernel
  through its op on a real device and checks the output against golden values
  from a trusted source (PyTorch, Hugging Face) or, where none fits, a plain
  CPU loop written from the formula.
- The cases cover, where the kernel allows them: very large inputs (buffers
  past 2^24 elements, near `maxStorageBufferBindingSize`), empty inputs, a
  dtype other than the declared one, sizes past a kernel's limit (it throws,
  it does not return garbage), and unusual shapes (1×1, odd sizes, sizes that
  are not a multiple of the tile or vec4 width).
- Cover every route a kernel can take: matmul alone has plain, gemv, smallm,
  tiled and tiledF16, each picked by shape and device features.
- A new route in a kernel lands with the cases that cover it fully: every
  dtype it runs, every option it handles, and shapes on both sides of each
  threshold that picks it, so the cases show which inputs take the route and
  which fall back.
- A case written for one route says so in a comment: the route it takes and
  the condition that picks it (`// smallm: m % 4 == 0, m <= 32, subgroups`).
  When the condition changes, the comment shows which cases to move.
- Run the suite after every meaningful change while tuning a kernel. A change
  to a kernel already on `main` also gets an A/B run against `main`'s output:
  any difference that was not intended means the suite is missing a case, so
  add it.

### Zoo

- Each model is tested against its PyTorch reference on the same inputs,
  corner cases included. Where our output differs on purpose, the test says
  why.

### Performance

- A change to kernels, `core` or the zoo is benchmarked on the models it
  touches, before and after. The pull request description carries the A/B
  report. Performance stays the same or gets better.
- Benchmarks always run on the same machine, and every run is saved to a file,
  so a slow regression over many PRs shows up.
