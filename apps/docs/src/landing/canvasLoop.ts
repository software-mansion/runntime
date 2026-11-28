/** The frame loop the landing page canvases share: sized to the element at a
 *  reduced resolution, paused while the tab is hidden or the canvas offscreen. */

export interface CanvasLoop {
  canvas: HTMLCanvasElement;
  /** Canvas pixels per CSS pixel. */
  scale: number;
  /** Draws one frame and stops, for reduced motion. */
  still: boolean;
  /** Seconds on the clock the first frame shows. */
  startTime: number;
  /** Draws a frame at `time` seconds, `dt` seconds after the previous one. */
  draw: (time: number, dt: number) => void;
}

/** Starts the loop and returns a function that stops it. Draws the first
 *  frame before returning, so a shader that does not compile throws here. */
export function runCanvasLoop({ canvas, scale, still, startTime, draw }: CanvasLoop): () => void {
  let stopped = false;
  let frame = 0;
  let visible = document.visibilityState === 'visible';
  let onScreen = true;
  let last = performance.now();
  let time = startTime;

  const resize = () => {
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * scale));
    canvas.height = Math.max(1, Math.round(rect.height * scale));
    draw(time, 0);
  };

  const tick = (now: number) => {
    frame = 0;
    if (stopped || !visible || !onScreen) return;
    const dt = Math.min((now - last) / 1000, 0.1);
    last = now;
    time += dt;
    draw(time, dt);
    frame = requestAnimationFrame(tick);
  };

  const resume = () => {
    if (still || stopped || frame || !visible || !onScreen) return;
    last = performance.now();
    frame = requestAnimationFrame(tick);
  };

  const onVisibility = () => {
    visible = document.visibilityState === 'visible';
    resume();
  };

  const resizeObserver = new ResizeObserver(resize);
  const screenObserver = new IntersectionObserver(([entry]) => {
    onScreen = entry?.isIntersecting ?? true;
    resume();
  });
  resize();
  resizeObserver.observe(canvas);
  screenObserver.observe(canvas);
  document.addEventListener('visibilitychange', onVisibility);
  resume();

  return () => {
    stopped = true;
    cancelAnimationFrame(frame);
    resizeObserver.disconnect();
    screenObserver.disconnect();
    document.removeEventListener('visibilitychange', onVisibility);
  };
}
