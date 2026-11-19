/** Audio input helpers: an encoded clip or raw microphone samples in, the
 *  mono Float32Array the speech models take out. Both use the
 *  browser's own decoder and resampler through an OfflineAudioContext, so
 *  they run in the browser only. Importing the file is safe anywhere, the
 *  browser objects appear only inside the functions. */

import { RunntimeError } from '../../../core/index.ts';

/** Sample rate the speech models take. */
export const SPEECH_SAMPLE_RATE = 16_000;

/** Decodes any clip the browser can play (wav, mp3, ogg, ...) to mono
 *  samples at `sampleRate`, 16 kHz by default. Stereo is mixed down to one
 *  channel. Throws INVALID_ARGUMENT for a sample rate this browser does not
 *  take. */
export async function decodeAudio(
  data: ArrayBuffer,
  sampleRate = SPEECH_SAMPLE_RATE,
): Promise<Float32Array> {
  if (!Number.isFinite(sampleRate) || sampleRate <= 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `decodeAudio: sampleRate ${sampleRate} must be a positive number`,
    );
  }
  // decodeAudioData resamples to its context's rate, so a context at
  // `sampleRate` does the resample. Its one-frame render buffer is never used.
  let probe: OfflineAudioContext;
  try {
    probe = new OfflineAudioContext(1, 1, sampleRate);
  } catch (err) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `decodeAudio: sampleRate ${sampleRate} is not one this browser takes`,
      { cause: err },
    );
  }
  const decoded = await probe.decodeAudioData(data);
  if (decoded.numberOfChannels === 1) return decoded.getChannelData(0);
  const mix = new OfflineAudioContext(1, decoded.length, sampleRate);
  const src = mix.createBufferSource();
  src.buffer = decoded;
  src.connect(mix.destination);
  src.start();
  return (await mix.startRendering()).getChannelData(0);
}

/** Resamples raw mono samples recorded at `sourceRate` to 16 kHz. Returns
 *  the input as is when it is already at 16 kHz. Use it on microphone
 *  samples, a lossy re-encode is not needed. */
export async function resampleAudio(
  samples: Float32Array,
  sourceRate: number,
): Promise<Float32Array> {
  if (sourceRate === SPEECH_SAMPLE_RATE) return samples;
  const frames = Math.ceil((samples.length * SPEECH_SAMPLE_RATE) / sourceRate);
  const off = new OfflineAudioContext(1, frames, SPEECH_SAMPLE_RATE);
  const buf = off.createBuffer(1, samples.length, sourceRate);
  // The lib type wants a non-shared buffer. Recorded samples always are one.
  buf.copyToChannel(samples as Float32Array<ArrayBuffer>, 0);
  const src = off.createBufferSource();
  src.buffer = buf;
  src.connect(off.destination);
  src.start();
  return (await off.startRendering()).getChannelData(0);
}
