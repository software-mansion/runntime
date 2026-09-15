import { resampleAudio, SPEECH_SAMPLE_RATE } from 'runntime/zoo';

/** Push-to-talk microphone capture: raw PCM through an AudioWorklet rather
 *  than MediaRecorder, because decodeAudioData on a recorded blob is
 *  unreliable on mobile browsers. The context asks for 16 kHz directly;
 *  where the browser refuses that rate, stop() resamples once. */
const CAPTURE_WORKLET = `
  registerProcessor('docs-pcm-capture', class extends AudioWorkletProcessor {
    process(inputs) {
      const channel = inputs[0][0];
      if (channel) this.port.postMessage(channel.slice(0));
      return true;
    }
  });
`;

export interface Recording {
  /** Stops the microphone and returns what was said, mono at 16 kHz. */
  stop(): Promise<Float32Array>;
}

export async function startRecording(): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ sampleRate: SPEECH_SAMPLE_RATE });
  } catch {
    ctx = new AudioContext(); // Device rate; resampled at stop.
  }

  const workletUrl = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: 'text/javascript' }));
  try {
    await ctx.audioWorklet.addModule(workletUrl);
  } finally {
    URL.revokeObjectURL(workletUrl);
  }

  const chunks: Float32Array[] = [];
  const tap = new AudioWorkletNode(ctx, 'docs-pcm-capture');
  tap.port.onmessage = (event) => chunks.push(event.data as Float32Array);
  ctx.createMediaStreamSource(stream).connect(tap);
  await ctx.resume();

  return {
    async stop() {
      const rate = ctx.sampleRate;
      for (const track of stream.getTracks()) track.stop();
      await ctx.close();

      const pcm = new Float32Array(chunks.reduce((n, chunk) => n + chunk.length, 0));
      let offset = 0;
      for (const chunk of chunks) {
        pcm.set(chunk, offset);
        offset += chunk.length;
      }
      return await resampleAudio(pcm, rate);
    },
  };
}
