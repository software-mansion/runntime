<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/software-mansion/runntime/main/apps/docs/src/assets/logo-dark.svg">
    <img alt="ruNNtime" src="https://raw.githubusercontent.com/software-mansion/runntime/main/apps/docs/src/assets/logo-light.svg" width="360">
  </picture>
  <br />
  <br />
  <p>
    <a href="https://docs.swmansion.com/runntime"><img alt="Documentation" src="https://img.shields.io/badge/docs-docs.swmansion.com%2Frunntime-blue?logo=readthedocs&logoColor=white" height="24"></a>
    <a href="https://www.npmjs.com/package/runntime"><img alt="npm" src="https://img.shields.io/npm/v/runntime?logo=npm&color=cb3837" height="24"></a>
    <a href="https://huggingface.co/software-mansion"><img alt="Follow us on Hugging Face" src="https://img.shields.io/badge/Hugging%20Face-software--mansion-ffd21e?logo=huggingface&logoColor=black" height="24"></a>
  </p>
</div>

**ruNNtime is a [high-performance](#performance) framework for running AI
models on the web, in Electron and in React Native.** Everything runs on the
local GPU, and your data can stay there from input to output, with no trip
through the CPU.

It ships as one package, `runntime`, with two entry points:

- **`runntime/zoo`** - ready-to-run models for speech, vision and text. One function call gives you a running model.
- **`runntime/core`** - the building blocks behind them, for composing your own models with PyTorch-like semantics. Not public yet, coming soon.

## Supported models

The zoo package currently supports 19 models across 9 tasks, each loaded with one function call.

| Task                  | Factory                   | Models                                |
| --------------------- | ------------------------- | ------------------------------------- |
| Embedding             | `createEmbedder`          | EmbeddingGemma 2, all-MiniLM-L6-v2    |
| Privacy filter        | `createPrivacyFilter`     | OpenAI privacy-filter (int8)          |
| Speech to text        | `createSpeechToText`      | Moonshine tiny / base, streaming tiny |
| Text to speech        | `createTextToSpeech`      | Reverb-TTS                            |
| Object detection      | `createObjectDetector`    | YOLO26 n / s / m                      |
| Instance segmentation | `createInstanceSegmenter` | YOLO26-seg n / s / m                  |
| Pose & keypoints      | `createKeypointDetector`  | YOLO26-pose n / s / m                 |
| Depth estimation      | `createDepthEstimator`    | DepthART s / b                        |
| Image classification  | `createImageClassifier`   | MobileNetV4 small                     |

## Performance

Against two popular alternative inference runtimes on the same GPU, tensor
in to tensor out (all-MiniLM-L6-v2, Apple M4 Pro, Chrome):

| Input                       | vs runtime A | vs runtime B |
| --------------------------- | ------------ | ------------ |
| 1 sentence (32 tokens)      | **2.25×**    | -            |
| 128 tokens                  | **1.83×**    | **3.8×**     |
| 32 × 128 tokens             | **1.26×**    | **4.2×**     |
| Real text, one at a time    | **2.05×**    | -            |
| Real text, in batches of 32 | **2.36×**    | -            |

For more benchmarks and measuring methodology, see the
[benchmarks page](https://docs.swmansion.com/runntime/zoo/benchmarks/).

Why it's fast:

- **Built for WebGPU only** - no generic backend layer, kernels tuned for
  what models actually run.
- **Fused kernels** - bias, activations and residual adds ride along in the
  kernel before them, so fewer dispatches.
- **One submit per call** - the whole forward pass goes to the GPU at once.
- **Specialized shaders** - TypeGPU generates a lean shader for every
  variant from one source.

## Key Features

- **No model conversion** - models run straight from their PyTorch weights.
  No export step to break, and input shapes are set at runtime, not baked
  in ahead of time.
- **Data stays on the GPU** - feed in a video frame, get results in a GPU
  buffer, ready for your own rendering.
- **One line to a running model** - pick a model, call one function, and it
  downloads and runs. Turn on caching to skip the download next time.
- **Just TypeScript** - no WASM or native binaries. The same code runs on the
  web, in Electron and in React Native.
- **Works with transformers.js** - already using it? One call swaps the engine
  and your code stays as it is, see
  [migrating from transformers.js](https://docs.swmansion.com/runntime/zoo/migrating-from-transformers-js/).

## Documentation

Everything else lives at
[docs.swmansion.com/runntime](https://docs.swmansion.com/runntime): how to
[get started](https://docs.swmansion.com/runntime/zoo/getting-started/), a
guide per task, and interactive examples you can run in the browser on your
own GPU.

```bash
npm install runntime typegpu
```

## License

ruNNtime is [MIT licensed](https://github.com/software-mansion/runntime/blob/main/LICENSE.md). The models it runs are not covered by
that: each has its own license, listed on its page on the
[Hugging Face Hub](https://huggingface.co/software-mansion).

## ruNNtime is created by Software Mansion AI Labs

<a href="https://ai.swmansion.com/?utm_source=runntime&utm_medium=readme">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/software-mansion/runntime/main/apps/docs/src/assets/swm-ai-labs-dark.svg">
    <img src="https://raw.githubusercontent.com/software-mansion/runntime/main/apps/docs/src/assets/swm-ai-labs-light.svg" alt="Software Mansion AI Labs" width="122" height="80" align="left">
  </picture>
</a>

We're an end-to-end AI development lab focused on local and hybrid AI. We build and fine-tune models, deploy them anywhere and turn them into agents that solve real business problems. [Hire us](https://ai.swmansion.com/?utm_source=runntime&utm_medium=readme#contact).
