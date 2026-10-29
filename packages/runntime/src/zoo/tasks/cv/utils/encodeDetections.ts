/** GPU in, GPU out for the YOLO26 runners: the detector's input written
 *  from a texture or a buffer, its head decoded on the GPU into a list of
 *  detections in the target buffer. */

import { d } from 'typegpu';
import {
  createDecodeDetections,
  createImageToChw,
  defaultRoot,
  detectionList,
  RunntimeError,
  type DetectionList,
  type U32Buffer,
} from '../../../../core/index.ts';
import { rethrowRunError } from '../../../errors.ts';
import type { Detector, Yolo26Input } from '../../../yolo26/detector.ts';
import {
  encodeWith,
  unwrapSource,
  unwrapStorageTarget,
  type EncodeOptions,
  type GpuImage,
  type GpuTensor,
} from '../../gpu.ts';
import type { ImagePreprocessor } from './imagePreprocessor.ts';

/** What every YOLO26 encode<Verb> method takes. */
export interface EncodeDetectionsOptions<Target extends d.BaseData> extends EncodeOptions<
  GpuImage | GpuTensor<Yolo26Input>,
  GpuTensor<Target>
> {
  readonly confidenceThreshold?: number;
  readonly maxDetections?: number;
}

/** Returns the list layout and what records one forward and its decode for
 *  an encode<Verb> method. `detection` is one list entry, `numExtra` the
 *  head's landmarks or mask prototypes, `maskSize` the mask side. */
export function createDetectionEncoder<T extends d.AnyWgslData>(
  detector: Detector,
  preprocessor: ImagePreprocessor,
  detection: T,
  extra: { readonly numExtra: number; readonly maskSize: number },
) {
  const root = defaultRoot();
  const { inputSize, numClasses } = detector;
  const imageToChw = createImageToChw(root);
  const decode = createDecodeDetections(root, detector.task, inputSize);
  // A buffer source is the model input itself, so the image is that size.
  const inputImage: U32Buffer = root
    .createBuffer(d.arrayOf(d.u32, 2), [inputSize, inputSize])
    .$usage('storage');
  const inputBytes = d.sizeOf(detector.schemas.input);
  const detectionBytes = d.sizeOf(detection);
  // `count`, then the 4-byte-aligned entries.
  const headerBytes = d.sizeOf(d.u32);
  const { mode, cropFraction } = preprocessor.gpuArgs;
  return {
    /** The list layout with room for `maxDetections`. */
    schema: (maxDetections: number): DetectionList<T> => detectionList(detection, maxDetections),
    encode(options: EncodeDetectionsOptions<DetectionList<T>>): void {
      const { source, target, encoder, confidenceThreshold = 0.3 } = options;
      if (
        typeof confidenceThreshold !== 'number' ||
        !(confidenceThreshold >= 0 && confidenceThreshold <= 1)
      ) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `confidenceThreshold: expected a number from 0 to 1, got ${confidenceThreshold}`,
        );
      }
      const src = unwrapSource(source, inputBytes);
      const out = unwrapStorageTarget(target, headerBytes + detectionBytes);
      const capacity = Math.floor((out.size - headerBytes) / detectionBytes);
      const { maxDetections = capacity } = options;
      if (!Number.isInteger(maxDetections) || maxDetections < 1 || maxDetections > capacity) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `maxDetections: expected an integer from 1 to ${capacity}, what the target holds, got ${maxDetections}`,
        );
      }
      try {
        encodeWith(encoder, (enc) =>
          detector.encode(
            enc,
            (input) => {
              if ('tensor' in src) {
                enc.copyBufferToBuffer(src.tensor, 0, root.unwrap(input), 0, inputBytes);
              } else {
                imageToChw.encode(enc, src.image, input, preprocessor.gpuArgs);
              }
            },
            ([p3, p4, p5, proto]) =>
              decode.encode(
                enc,
                [p3!, p4!, p5!],
                proto,
                'tensor' in src ? inputImage : imageToChw.size,
                out,
                {
                  numClasses,
                  numExtra: extra.numExtra,
                  confidenceThreshold,
                  maxDetections,
                  mode: 'tensor' in src ? 'stretch' : mode,
                  cropFraction,
                  maskSize: extra.maskSize,
                },
              ),
          ),
        );
      } catch (err) {
        rethrowRunError(err);
      }
    },
    dispose(): void {
      imageToChw.dispose();
      decode.dispose();
      inputImage.destroy();
    },
  };
}
