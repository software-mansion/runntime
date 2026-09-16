/** Audio input helpers: an encoded clip or raw microphone samples in, the
 *  16 kHz mono Float32Array the speech models take out. Both use the
 *  browser's own decoder and resampler through an OfflineAudioContext, so
 *  they run in the browser only. Importing the file is safe anywhere, the
 *  browser objects appear only inside the functions. */

/** Sample rate the speech models take. */
export const SPEECH_SAMPLE_RATE = 16_000;

/** Decodes any clip the browser can play (wav, mp3, ogg, ...) to 16 kHz
 *  mono samples. Stereo is mixed down to one channel. */
export async function decodeAudio(data: ArrayBuffer): Promise<Float32Array> {
  // decodeAudioData resamples to its context's rate, so a 16 kHz context
  // does the resample. Its one-frame render buffer is never used.
  const probe = new OfflineAudioContext(1, 1, SPEECH_SAMPLE_RATE);
  const decoded = await probe.decodeAudioData(data);
  if (decoded.numberOfChannels === 1) return decoded.getChannelData(0);
  const mix = new OfflineAudioContext(1, decoded.length, SPEECH_SAMPLE_RATE);
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
