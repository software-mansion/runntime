---
name: testing
description: How to write and run tests and benchmarks in this repo. Use whenever adding, changing or reviewing a test in packages/runntime/test, adding or changing a kernel or a kernel route, fixing a bug, or benchmarking a change.
---

# Testing

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

## Kernels

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
- Cover every route a kernel can take: matmul alone has plain, gemv, gemvVec4, smallm,
  tiled and tiledVec4, each picked by shape and device features.
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

## Zoo

- Each model is tested against its PyTorch reference on the same inputs,
  corner cases included. Where our output differs on purpose, the test says
  why.

## Performance

- A change to kernels, `core` or the zoo is benchmarked on the models it
  touches, before and after. The pull request description carries the A/B
  report. Performance stays the same or gets better.
- Benchmarks always run on the same machine, and every run is saved to a file,
  so a slow regression over many PRs shows up.
- A change to a kernel gets a kernel A/B against `main`:
  `pnpm --filter runntime bench:ab kernels [-b <ref>] [-k matmul,reduce]`.
  It copies the base ref's `src` to `.bench/base`, runs both copies in one
  headless Chromium page on the same inputs with their rounds interleaved,
  and prints a table, with a markdown copy for the pull request in
  `.bench/results/report.md`. A row says `faster` or `slower` only past the
  noise: both versions' interquartile spread, at least 3%.
- The table's last column is the largest difference between A's and B's
  outputs. Anything but 0 on a kernel the change did not mean to touch is a
  bug, or a case the GPU tests are missing.
- Cases live in `bench/kernels/<kernel>.ab.ts`, one per route the dispatch
  can pick, with shapes from the zoo models. A new route gets a case.
- Close the GPU-heavy apps before a run, and run on power. A row that flips
  between runs is noise, not a result.
