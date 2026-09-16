/** MobileNetV4-Conv-S architecture table, transcribed from timm's
 *  mobilenetv4_conv_small (the `MobileNetV3` class it instantiates). Each row
 *  is one block; in-channels are derived by walking the table. `expand` is the
 *  UIR expansion ratio applied to the block's INPUT channels (all products are
 *  whole numbers for this variant, so no make_divisible rounding is needed). */

/** ConvBnAct: plain conv + BN + ReLU. */
export interface ConvBlockSpec {
  kind: 'cn';
  out: number;
  kernelSize: number;
  stride: number;
}

/** UniversalInvertedResidual: optional leading depthwise (no act) → 1×1 expand
 *  (ReLU) → optional mid depthwise (ReLU, carries the stride) → 1×1 project
 *  (no act). A kernel size of 0 means that depthwise stage is absent. */
export interface UirBlockSpec {
  kind: 'uir';
  out: number;
  expand: number;
  dwStartK: number;
  dwMidK: number;
  stride: number;
}

export type Mnv4BlockSpec = ConvBlockSpec | UirBlockSpec;

export interface Mnv4Config {
  stemOut: number;
  stages: readonly (readonly Mnv4BlockSpec[])[];
  /** conv_head width (timm num_features). */
  headHidden: number;
  numClasses: number;
  inputSize: number;
}

const cn = (out: number, kernelSize: number, stride: number): ConvBlockSpec => ({
  kind: 'cn',
  out,
  kernelSize,
  stride,
});
const uir = (
  out: number,
  expand: number,
  dwStartK: number,
  dwMidK: number,
  stride: number,
): UirBlockSpec => ({ kind: 'uir', out, expand, dwStartK, dwMidK, stride });

/** timm mobilenetv4_conv_small.e2400_r224_in1k (3.8M params, 224×224 in1k). */
export const MOBILENETV4_CONV_S: Mnv4Config = {
  stemOut: 32,
  stages: [
    // stage 0, 112×112 in
    [cn(32, 3, 2), cn(32, 1, 1)],
    // stage 1, 56×56 in
    [cn(96, 3, 2), cn(64, 1, 1)],
    // stage 2, 28×28 in
    [
      uir(96, 3, 5, 5, 2),
      uir(96, 2, 0, 3, 1),
      uir(96, 2, 0, 3, 1),
      uir(96, 2, 0, 3, 1),
      uir(96, 2, 0, 3, 1),
      uir(96, 4, 3, 0, 1),
    ],
    // stage 3, 14×14 in
    [
      uir(128, 6, 3, 3, 2),
      uir(128, 4, 5, 5, 1),
      uir(128, 4, 0, 5, 1),
      uir(128, 3, 0, 5, 1),
      uir(128, 4, 0, 3, 1),
      uir(128, 4, 0, 3, 1),
    ],
    // stage 4, 7×7 in
    [cn(960, 1, 1)],
  ],
  headHidden: 1280,
  numClasses: 1000,
  inputSize: 224,
};

/** ImageNet preprocessing constants (timm pretrained_cfg for this checkpoint). */
export const IMAGENET_MEAN = [0.485, 0.456, 0.406] as const;
export const IMAGENET_STD = [0.229, 0.224, 0.225] as const;
/** The evaluation crop: 224 out of a 256 short side (timm crop_pct). */
export const MOBILENETV4_CROP_FRACTION = 224 / 256;
