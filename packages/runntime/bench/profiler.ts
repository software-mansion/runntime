/** GPU time per command buffer from timestamp queries. It wraps the WebGPU
 *  methods themselves, so both library copies in the A/B are timed the same
 *  way whatever their internals do. */

/** Timestamp pairs between two submits, one per compute pass. The executor
 *  encodes a submit as one pass. A set of 8192 queries reads back all zeros on
 *  Apple GPUs, so keep it small. */
const MAX_PAIRS = 256;

export interface Profiler {
  /** Waits for every submit so far and returns the GPU time of each command
   *  buffer that ran a compute pass, in ns. */
  collect(): Promise<number[]>;
}

/** Times every compute pass encoded on `device`. Call once per device; each
 *  call wraps the previous one and passes other devices through. On Metal, Chrome samples
 *  timestamps only at command-buffer boundaries, so a buffer's time is the
 *  span from its first pass start to its last pass end. */
export function installProfiler(device: GPUDevice): Profiler {
  if (!device.features.has('timestamp-query')) {
    throw new Error('kernel A/B: the device has no timestamp-query');
  }
  const querySet = device.createQuerySet({ type: 'timestamp', count: MAX_PAIRS * 2 });
  let used = 0;
  const encoderPairs = new WeakMap<GPUCommandEncoder, number[]>();
  const bufferPairs = new WeakMap<GPUCommandBuffer, number[]>();
  const ours = new WeakSet<GPUCommandEncoder>();
  const pending: Promise<number[]>[] = [];

  const createEncoder = GPUDevice.prototype.createCommandEncoder;
  GPUDevice.prototype.createCommandEncoder = function (desc) {
    const encoder = createEncoder.call(this, desc);
    if (this === device) ours.add(encoder);
    return encoder;
  };

  const beginPass = GPUCommandEncoder.prototype.beginComputePass;
  GPUCommandEncoder.prototype.beginComputePass = function (desc = {}) {
    if (!ours.has(this) || desc.timestampWrites || used >= MAX_PAIRS) {
      return beginPass.call(this, desc);
    }
    const pair = used++;
    const pairs = encoderPairs.get(this) ?? [];
    pairs.push(pair);
    encoderPairs.set(this, pairs);
    return beginPass.call(this, {
      ...desc,
      timestampWrites: {
        querySet,
        beginningOfPassWriteIndex: pair * 2,
        endOfPassWriteIndex: pair * 2 + 1,
      },
    });
  };

  const finish = GPUCommandEncoder.prototype.finish;
  GPUCommandEncoder.prototype.finish = function (desc) {
    const buffer = finish.call(this, desc);
    const pairs = encoderPairs.get(this);
    if (pairs) bufferPairs.set(buffer, pairs);
    return buffer;
  };

  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function (buffers) {
    const list = [...buffers];
    const timed = list.map((b) => bufferPairs.get(b)).filter((p) => p !== undefined);
    if (this !== device.queue || timed.length === 0) return submit.call(this, list);
    const bytes = used * 16;
    const resolve = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    });
    const staging = device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const encoder = createEncoder.call(device);
    encoder.resolveQuerySet(querySet, 0, used * 2, resolve, 0);
    encoder.copyBufferToBuffer(resolve, 0, staging, 0, bytes);
    submit.call(this, [...list, finish.call(encoder)]);
    used = 0;
    pending.push(
      staging.mapAsync(GPUMapMode.READ).then(() => {
        const ts = new BigUint64Array(staging.getMappedRange().slice(0));
        staging.destroy();
        resolve.destroy();
        return timed.map((pairs) => {
          const begin = pairs.reduce(
            (m, p) => (ts[p * 2]! < m ? ts[p * 2]! : m),
            ts[pairs[0]! * 2]!,
          );
          const end = pairs.reduce((m, p) => (ts[p * 2 + 1]! > m ? ts[p * 2 + 1]! : m), 0n);
          return end > begin ? Number(end - begin) : 0;
        });
      }),
    );
  };

  return {
    async collect() {
      const all = await Promise.all(pending.splice(0));
      return all.flat();
    },
  };
}
