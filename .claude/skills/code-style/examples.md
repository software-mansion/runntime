# Examples

Short bad/good pairs. The good side is in the style of this repo.

## Comment about the past

```ts
// Bad
// Switched accumulators to f32, f16 was drifting.
export const F16_ELEM: Elem = { key: 'f16', scalar: d.f16, vec4: d.vec4h as unknown as d.Vec4f };

// Good
/** Only storage varies: accumulators stay f32 everywhere, since an f16
 *  accumulator over a long reduction drifts percent-level. */
export const F16_ELEM: Elem = { key: 'f16', scalar: d.f16, vec4: d.vec4h as unknown as d.Vec4f };
```

## Comment that repeats the code

```ts
// Bad
// Create a queue promise and chain the run onto it
let queue: Promise<unknown> = Promise.resolve();

// Good
// Calls run one after another: the next preprocess overwrites the scratch
// array only after the previous run took it.
let queue: Promise<unknown> = Promise.resolve();
```

## Abstraction with one caller

```ts
// Bad
interface QueueOptions {
  concurrency?: number;
  onError?: (err: unknown) => void;
}
function createTaskQueue(opts: QueueOptions = {}) {
  /* 30 lines */
}
const queue = createTaskQueue();

// Good
let queue: Promise<unknown> = Promise.resolve();
const run = queue.then(() => detector.run(input));
queue = run.catch(() => undefined);
```

## Checks inside the library

```ts
// Bad: the types and the caller already guarantee all of this
function concat(chunks: Float32Array[], length: number): Float32Array {
  if (!Array.isArray(chunks)) throw new Error('chunks must be an array');
  if (length < 0) throw new Error('length must be positive');
  if (chunks.length === 0) return new Float32Array(0);
  // ...
}

// Good: validate at the public method, trust it inside
streamInsert(samples) {
  if (!(samples instanceof Float32Array)) {
    throw new RunntimeError('INVALID_ARGUMENT', 'streamInsert: samples must be a Float32Array');
  }
  // ...
}
```

## Type assertion instead of a check

```ts
// Bad: the cast trusts the config file, a missing key becomes NaN later
const dModel = json.hidden_size as number;

// Good: typeof narrows, and a bad config is a CHECKPOINT_MISMATCH
function num(json: Json, key: string): number {
  const v = json[key];
  if (typeof v !== 'number')
    throw checkpointMismatch('moonshine config', `'${key}' missing or not a number`);
  return v;
}
const dModel = num(json, 'hidden_size');
```

A caught error is `unknown`: anything can be thrown, not only an `Error`.

```ts
// Bad
} catch (err) {
  if ((err as RunntimeError).code) throw err;
  throw new RunntimeError('LOAD_FAILED', `${where}: ${(err as Error).message}`, { cause: err });
}

// Good: isRunntimeError is a type guard, messageOf narrows with instanceof
} catch (err) {
  if (isRunntimeError(err)) throw err;
  throw new RunntimeError('LOAD_FAILED', `${where}: ${messageOf(err)}`, { cause: err });
}
```

## Magic number

```ts
// Bad
if (live.fresh >= 8000) live.wake();

// Good
/** New audio between two transcriptions of the sentence being spoken. */
const TICK_SAMPLES = SPEECH_SAMPLE_RATE / 2;
if (live.fresh >= TICK_SAMPLES) live.wake();
```

## Doc comment on an export

```ts
// Bad
/**
 * This is a helper function that is used to check the abort signal.
 * @param signal - The abort signal to check.
 * @returns void
 */
export function throwIfAborted(signal?: AbortSignal): void;

// Good
/** Throws when the signal is already aborted. */
export function throwIfAborted(signal?: AbortSignal): void;
```

Options: say the unit, the range and the default.

```ts
/** Drops objects scored below this, 0 to 1. Default 0.3. */
readonly confidenceThreshold?: number;
```

## Docs intro

```md
<!-- Bad -->

The `createTextEmbedder` function leverages the powerful MiniLM architecture
to seamlessly provide robust, state-of-the-art embedding capabilities
directly within your browser environment.

<!-- Good -->

Text embedding turns text into a vector of numbers that captures its
meaning. Texts about the same thing get similar vectors, which makes the
vectors useful for semantic search, text classification, clustering and
finding duplicates.
```

## Docs code sample

```ts
// Bad
// Import the functions
import { createTextEmbedder, models, similarity } from 'runntime/zoo';

// Create the embedder
const embedder = await createTextEmbedder(models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT);
// Embed the texts
const a = await embedder.embed('how do I reset my password');
const b = await embedder.embed('Click "Forgot password" on the login page.');
// Calculate the similarity
console.log(similarity(a, b));

// Good
import { createTextEmbedder, models, similarity } from 'runntime/zoo';

// 1. Load the model once. The weights download from the Hugging Face Hub.
const embedder = await createTextEmbedder(models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT);

// 2. Turn text into vectors.
const query = await embedder.embed('how do I reset my password');
const answers = await embedder.embedBatch([
  'Click "Forgot password" on the login page.',
  'Our office is open from 9 to 5.',
]);

// 3. Compare them.
similarity(query, answers[0]!); // 0.68, related
similarity(query, answers[1]!); // 0.05, unrelated

// 4. Free the GPU memory when done.
embedder.dispose();
```
