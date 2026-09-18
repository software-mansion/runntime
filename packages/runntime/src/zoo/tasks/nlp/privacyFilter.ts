/** Privacy filter task: text in, the personal data it contains out, as
 *  labeled spans. Runs OpenAI's privacy-filter on the initRunntime() device. */

import { createResourceScope, RunntimeError } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import { models } from '../../models.ts';
import { createDetector } from '../../privacy-filter/detector.ts';

/** What to load. With no config, the int8 export from the Hugging Face
 *  Hub, models.privacyFilter.PRIVACY_FILTER.DEFAULT. */
export interface PrivacyFilterModel {
  /** The safetensors checkpoint: the hosted int8 export, or your own int8,
   *  f16 or bf16 export of the official file. The weight format comes from
   *  the file. */
  readonly modelPath?: ModelPath;
}

/** One piece of personal data found in the text. */
export interface PrivacySpan {
  /** The kind of data: `private_person`, `private_email`, `private_phone`,
   *  `private_address`, `private_date`, `private_url`, `account_number`
   *  or `secret`. */
  readonly label: string;
  /** Where it sits in the text, as string indices. `end` is exclusive. */
  readonly start: number;
  readonly end: number;
  /** The matched text, `text.slice(start, end)`. */
  readonly text: string;
  /** A tag to put in its place, like `<PRIVATE_EMAIL>`. */
  readonly placeholder: string;
}

export interface PrivacyFilter {
  /** Text in, every span of personal data in it out, in text order.
   *  Empty text gives an empty list. */
  detect(text: string): Promise<PrivacySpan[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner. */
export async function createPrivacyFilter(
  config: PrivacyFilterModel = {},
  opts: LoadOptions = {},
): Promise<PrivacyFilter> {
  const scope = createResourceScope();
  try {
    const { modelPath = models.privacyFilter.PRIVACY_FILTER.DEFAULT.modelPath } = config;

    // Cache key. The whole path, so same-named files of two formats stay
    // apart. Bump the version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `privacy-filter-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const detector = scope.track(await createDetector(sd, { onProgress: opts.onProgress }));

    let disposed = false;
    // Calls run one after another, one forward pass at a time on the GPU.
    let queue: Promise<unknown> = Promise.resolve();

    return {
      async detect(text) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'privacy filter is disposed');
        if (typeof text !== 'string') {
          throw new RunntimeError('INVALID_ARGUMENT', 'detect: text must be a string');
        }
        const run = queue.then(() => detector.detect(text));
        queue = run.catch(() => undefined);
        return (await run.catch(rethrowRunError)).spans;
      },
      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err);
  }
}
