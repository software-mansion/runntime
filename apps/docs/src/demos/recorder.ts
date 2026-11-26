import { resampleAudio, SPEECH_SAMPLE_RATE } from 'runntime/zoo';

/** Push-to-talk microphone capture: raw PCM through an AudioWorklet rather
 *  than MediaRecorder, because decodeAudioData on a recorded blob is
 *  unreliable on mobile browsers. The context asks for the wanted rate
 *  directly, 16 kHz unless told otherwise; where the browser refuses that
 *  rate, it captures at the device rate and stop() resamples once. */
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
  /** Stops the microphone and returns what was said, mono at the rate it
   *  was captured at, in Hz. */
  stopRaw(): Promise<{ samples: Float32Array; sampleRate: number }>;
}

/** Starts capturing the microphone at `sampleRate` Hz, or at the device rate
 *  where the browser refuses it. */
export async function startRecording(sampleRate = SPEECH_SAMPLE_RATE): Promise<Recording> {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  let ctx: AudioContext;
  try {
    ctx = new AudioContext({ sampleRate });
  } catch {
    ctx = new AudioContext(); // Device rate.
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

  const stopRaw = async () => {
    const rate = ctx.sampleRate;
    for (const track of stream.getTracks()) track.stop();
    await ctx.close();

    const pcm = new Float32Array(chunks.reduce((n, chunk) => n + chunk.length, 0));
    let offset = 0;
    for (const chunk of chunks) {
      pcm.set(chunk, offset);
      offset += chunk.length;
    }
    return { samples: pcm, sampleRate: rate };
  };

  return {
    async stop() {
      const { samples, sampleRate: rate } = await stopRaw();
      return await resampleAudio(samples, rate);
    },
    stopRaw,
  };
}
