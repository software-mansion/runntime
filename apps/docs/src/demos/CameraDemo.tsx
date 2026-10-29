import { imageBufferFromImageData, type ImageBuffer } from 'runntime/zoo';
import { useCallback, useEffect, useRef, useState, type ReactNode } from 'react';
import { initEngine } from './engine.ts';

/** Shipped with the docs, for readers without a camera. `BASE_URL` is what
 *  the site is served under, so the path holds up under /runntime/ too. */
const SAMPLE_IMAGE = `${import.meta.env.BASE_URL}demos/bus.jpg`;

/** Frames are read at this width; the height follows the source. */
const FRAME_WIDTH = 640;

type Facing = 'user' | 'environment';

type CameraDemoProps<T> =
  | {
      run: (frame: ImageBuffer) => Promise<T>;
      /** Paints the results over the frame. Left out by a model whose results
       *  are not a picture, like a list of class names. */
      draw?: (ctx: CanvasRenderingContext2D, results: T) => void;
      /** Rendered under the frame, for results that read better as markup. */
      below?: (results: T) => ReactNode;
      /** Rendered in the footer next to the frame rate. */
      stats: (results: T) => string;
    }
  | {
      /** Records the model from the frame into the canvas texture, with no
       *  CPU copy. Replaces `run` and `draw`. */
      encode: (source: GPUExternalTexture, target: GPUTexture) => void;
      /** Rendered in the footer next to the frame rate. */
      stats: string;
    };

/** A camera loop, or the sample picture, with a model's results drawn over
 *  every frame. `run` calls the model, `draw` paints its results after the
 *  frame; `encode` does both on the GPU. Frames wait for the previous run,
 *  so the loop never queues up. */
export function CameraDemo<T>(props: CameraDemoProps<T>) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  // Off the DOM: the page styles every video element visible.
  const videoRef = useRef<HTMLVideoElement>(null);
  const grabRef = useRef<OffscreenCanvas>(null);
  const contextRef = useRef<GPUCanvasContext>(null);
  const stopRef = useRef<() => void>(null);
  const [source, setSource] = useState<'idle' | 'camera' | 'sample'>('idle');
  const [results, setResults] = useState<T>();
  const [facing, setFacing] = useState<Facing>('user');
  const [summary, setSummary] = useState<string>();
  const [fps, setFps] = useState<number>();
  const [runMs, setRunMs] = useState<number>();
  const [error, setError] = useState<string>();

  const stop = useCallback(() => {
    stopRef.current?.();
    stopRef.current = null;
    setSource('idle');
    setResults(undefined);
    setFps(undefined);
  }, []);

  /** One frame from `image` through the model and onto the canvas. */
  const step = useCallback(
    async (
      image: HTMLVideoElement | VideoFrame,
      width: number,
      height: number,
      mirror: boolean,
    ) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const scale = FRAME_WIDTH / width;
      const w = FRAME_WIDTH;
      const h = Math.round(height * scale);
      if ('encode' in props) {
        const { device } = await initEngine();
        // Resizing drops the canvas texture, so only on a new frame size.
        if (canvas.width !== w || canvas.height !== h) {
          canvas.width = w;
          canvas.height = h;
        }
        if (!contextRef.current) {
          contextRef.current = canvas.getContext('webgpu')!;
          contextRef.current.configure({
            device,
            format: navigator.gpu.getPreferredCanvasFormat(),
          });
        }
        const context = contextRef.current;
        // The map of a mirrored frame is the mirrored map.
        canvas.style.transform = mirror ? 'scaleX(-1)' : '';
        const start = performance.now();
        props.encode(device.importExternalTexture({ source: image }), context.getCurrentTexture());
        await device.queue.onSubmittedWorkDone();
        setRunMs(performance.now() - start);
        setSummary(props.stats);
        return;
      }
      const { run, draw, stats } = props;
      const grab = (grabRef.current ??= new OffscreenCanvas(w, h));
      grab.width = w;
      grab.height = h;
      const gctx = grab.getContext('2d', { willReadFrequently: true })!;
      gctx.save();
      if (mirror) {
        gctx.translate(w, 0);
        gctx.scale(-1, 1);
      }
      gctx.drawImage(image, 0, 0, w, h);
      gctx.restore();
      const start = performance.now();
      const results = await run(imageBufferFromImageData(gctx.getImageData(0, 0, w, h)));
      setRunMs(performance.now() - start);
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(grab, 0, 0);
      draw?.(ctx, results);
      setResults(results);
      setSummary(stats(results));
    },
    [props],
  );

  const startCamera = useCallback(
    async (which: Facing) => {
      stop();
      setError(undefined);
      const video = (videoRef.current ??= document.createElement('video'));
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 1280, height: 720, facingMode: which },
        });
        video.srcObject = stream;
        video.muted = true;
        video.playsInline = true;
        await video.play();
      } catch {
        setError('No camera. Check the permission the browser asked for, or use the sample.');
        return;
      }
      setSource('camera');
      let alive = true;
      let frames = 0;
      let since = performance.now();
      const loop = async () => {
        if (!alive) return;
        try {
          await step(video, video.videoWidth, video.videoHeight, which === 'user');
        } catch (err) {
          setError((err as Error).message);
          stop();
          return;
        }
        frames += 1;
        const now = performance.now();
        if (now - since > 1000) {
          setFps((frames * 1000) / (now - since));
          frames = 0;
          since = now;
        }
        if (alive) requestAnimationFrame(() => void loop());
      };
      stopRef.current = () => {
        alive = false;
        (video.srcObject as MediaStream | null)?.getTracks().forEach((t) => t.stop());
        video.srcObject = null;
      };
      void loop();
    },
    [step, stop],
  );

  const showSample = useCallback(async () => {
    stop();
    setError(undefined);
    try {
      const bitmap = await createImageBitmap(await (await fetch(SAMPLE_IMAGE)).blob());
      // A VideoFrame, so the GPU path can import it like a camera frame.
      const frame = new VideoFrame(bitmap, { timestamp: 0 });
      bitmap.close();
      try {
        await step(frame, frame.displayWidth, frame.displayHeight, false);
      } finally {
        frame.close();
      }
      setSource('sample');
    } catch (err) {
      setError((err as Error).message);
    }
  }, [step, stop]);

  const flip = () => {
    const next: Facing = facing === 'user' ? 'environment' : 'user';
    setFacing(next);
    void startCamera(next);
  };

  // Release the camera when the reader leaves the page.
  useEffect(() => stop, [stop]);

  return (
    <div className="demo not-content">
      <div className="demo-row">
        {source === 'camera' ? (
          <>
            <button className="demo-button" onClick={stop}>
              <span className="demo-glyph">■</span>Stop
            </button>
            <button className="demo-button demo-button-quiet" onClick={flip}>
              Flip camera
            </button>
          </>
        ) : (
          <>
            <button className="demo-button" onClick={() => void startCamera(facing)}>
              <span className="demo-glyph">●</span>Start camera
            </button>
            <button className="demo-button demo-button-quiet" onClick={() => void showSample()}>
              Sample picture
            </button>
          </>
        )}
      </div>

      <canvas ref={canvasRef} className="demo-canvas" hidden={source === 'idle'} />
      {source !== 'idle' && results !== undefined && 'below' in props && props.below?.(results)}
      {source === 'idle' && !error && (
        <p className="demo-note">Start the camera to see live results, or try the sample.</p>
      )}

      {error && <p className="demo-error">{error}</p>}

      <div className="demo-stats">
        <span>{summary ?? '—'}</span>
        <span>run {runMs === undefined ? '—' : `${runMs.toFixed(0)} ms`}</span>
        {fps !== undefined && <span>{fps.toFixed(1)} fps</span>}
      </div>
    </div>
  );
}
