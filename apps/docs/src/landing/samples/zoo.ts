/** The runntime/zoo sample on the landing page, kept in a .ts file so it
 *  type-checks. The page shows only what follows the cut line. */

declare const canvas: HTMLCanvasElement;
// ---cut---
import {
  createObjectDetector,
  imageBufferFromImageData,
  models,
} from 'runntime/zoo';

const model = models.objectDetection.YOLO26.DEFAULT;
const detector = await createObjectDetector(model);

const ctx = canvas.getContext('2d')!;
const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
const image = imageBufferFromImageData(pixels);
const objects = await detector.detectObjects(image);
console.log(objects);
// [{ label, confidence, box }, ...]
