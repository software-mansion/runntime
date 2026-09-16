<div align="center">
  <h1>ruNNtime</h1>
  <p><strong>AI models on the web, built with TypeGPU.</strong></p>
</div>

<br />

**ruNNtime** is a framework for running AI models on the user's own GPU,
implemented using TypeGPU. It composes GPU kernels into the layers a model is
made of, attention, convolutions, norms, matmuls and builds common
architectures out of them. The kernels are plain TypeGPU code compiled to
WGSL at build time, so there is no WASM binary to download at runtime, and
the same code runs anywhere WebGPU does: the browser, React Native and
Electron.

It ships as one package, `runntime`, with two entry points:

- **`runntime/zoo`** - ready-to-run model implementations for commonly used tasks: speech, vision, and natural language processing. You call a single function, and get a running model.
- **`runntime/core`** - this package is not yet exposed to the public. It is a set of pre-built kernels for building neural networks on the Web, mirroring PyTorch semantics.

## Table of Contents

- [Key Features](#key-features)
- [Quickstart](#quickstart)
  - [1. Installation](#1-installation)
  - [2. Set up the engine](#2-set-up-the-engine)
  - [3. Run a model](#3-run-a-model)
- [Documentation](#documentation)
- [Migrating from transformers.js](#coming-from-transformers.js)
- [Created by Software Mansion](#created-by-software-mansion)

## Key Features

- **One function per task** - `createObjectDetector()` hands you a loaded
  model. Call `detectObjects(image)`, get boxes back. The `zoo` api is meant to be straight-forward and just work.
- **No WASM to download** - the kernels are TypeGPU code compiled
  to WGSL at build time, so there is no WASM binary shipped alongside your
  app.
- **Same code everywhere WebGPU runs** - the browser, React Native (through
  [react-native-webgpu](https://github.com/wcandillon/react-native-webgpu))
  and Electron.
- **Weights come from the Hub** - every model points at a
  [software-mansion repo](https://huggingface.co/software-mansion). Turn on the OPFS cache and the second load skips the download.
- **Drop-in for transformers.js** - keep your `pipeline()` code, swap the
  backend underneath.

## Quickstart

### 1. Installation

```bash
npm install runntime
# or
pnpm add runntime
# or
yarn add runntime
```

ruNNtime is built on [TypeGPU](https://docs.swmansion.com/TypeGPU/), so you need it
installed. The kernels are already compiled to shaders in the published package, so
no bundler plugin is required. Follow the
[TypeGPU installation guide](https://docs.swmansion.com/TypeGPU/getting-started/).

> [!IMPORTANT]
> ruNNtime needs WebGPU:
>
> - **Web** - Chrome, Edge, Safari 26 and newer, recent Firefox. See
>   [caniuse](https://caniuse.com/webgpu).
> - **Electron** - ships Chromium, so the web setup works as is.
> - **React Native** - through `react-native-webgpu`. Call its
>   `installWebGPU()`, request the device yourself and wrap it with
>   `tgpu.initFromDevice({ device })`.
>
> Object detection, instance segmentation, pose, depth estimation and image
> classification run in half precision and need the `shader-f16` feature.
> Text embedding, speech to text and the privacy filter can run on
> devices without it.

### 2. Set up the engine

Once per page, before you create any model:

```ts
import { initRunntime } from 'runntime/zoo';
import tgpu from 'typegpu';

initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
```

Both features are optional, so the call succeeds on devices without them.
Many kernels are much faster with `subgroups`. If your app already uses
TypeGPU, pass the root you have.

### 3. Run a model

```ts
import { createObjectDetector, imageBufferFromImageData, models } from 'runntime/zoo';

// Load the model once. The weights download from the Hugging Face Hub.
const detector = await createObjectDetector(models.objectDetection.YOLO26.DEFAULT);

// Get the pixels, here from a canvas the picture is drawn on.
const ctx = canvas.getContext('2d');
const image = imageBufferFromImageData(ctx.getImageData(0, 0, canvas.width, canvas.height));

// Find the objects: one entry per object, best first.
const objects = await detector.detectObjects(image);
for (const { label, confidence, box } of objects) {
  console.log(label, confidence, box);
}
// bus 0.93 { format: 'xyxy', xmin: 6, ymin: 228, xmax: 807, ymax: 749 }
// person 0.92 { format: 'xyxy', xmin: 47, ymin: 399, xmax: 237, ymax: 902 }

// Free the GPU memory when done.
detector.dispose();
```

A video or a camera works the same way: draw each frame on the canvas and
call `detectObjects` again. The detector is loaded once.

## Models

Every task has its own factory function and its own methods. Pass it an entry
from `models`, or spread one and change a field.

| Task                  | Factory                   | Models                                |
| --------------------- | ------------------------- | ------------------------------------- |
| Text embedding        | `createTextEmbedder`      | all-MiniLM-L6-v2                      |
| Privacy filter        | `createPrivacyFilter`     | OpenAI privacy-filter (int8)          |
| Speech to text        | `createSpeechToText`      | Moonshine tiny / base, streaming tiny |
| Object detection      | `createObjectDetector`    | YOLO26 n / m                          |
| Instance segmentation | `createInstanceSegmenter` | YOLO26-seg n / m                      |
| Pose & keypoints      | `createKeypointDetector`  | YOLO26-pose n / m                     |
| Depth estimation      | `createDepthEstimator`    | DepthART s / b                        |
| Image classification  | `createImageClassifier`   | MobileNetV4 small                     |

## Coming from transformers.js

Already on transformers.js and don't want to move your whole codebase? Keep
your `pipeline()` code and swap the engine underneath. By default
transformers.js runs on ONNX Runtime, which pays for its generality in
speed. One `initRunntimeBackend()` call replaces that backend with ruNNtime.
It only changes anything for the models ruNNtime supports, everything else
keeps loading on ONNX Runtime.

```ts
import { initRunntimeBackend } from 'runntime/zoo/transformers';
import { pipeline } from '@huggingface/transformers';

await initRunntimeBackend({ fallbackToOnnx: true });

const pipe = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
```

## Documentation

Full docs live at
[docs.swmansion.com/runntime](https://docs.swmansion.com/runntime) - a guide
per task, and some cool interactive examples you can run in the
browser on your own GPU.

## Created by Software Mansion

Since 2012, [Software Mansion](https://swmansion.com) has been building
mobile and web apps, contributing to open-source software, and dealing with
all kinds of React Native and graphics challenges. We are Core React Native
Contributors and the authors of TypeGPU. We can help you build your next AI
product – [Hire us](https://swmansion.com/contact?utm_source=runntime&utm_medium=readme).

[![swm](https://logo.swmansion.com/logo?color=white&variant=desktop&width=150&tag=runntime-github 'Software Mansion')](https://swmansion.com)
