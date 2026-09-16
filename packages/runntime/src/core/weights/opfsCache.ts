import type { WeightCache } from './cache.ts';

/** OPFS-backed cache; returns undefined when OPFS is unavailable. */
export async function createOpfsCache(namespace: string): Promise<WeightCache | undefined> {
  if (!('storage' in navigator) || !navigator.storage.getDirectory) return undefined;
  let dir: FileSystemDirectoryHandle;
  try {
    const opfsRoot = await navigator.storage.getDirectory();
    dir = await opfsRoot.getDirectoryHandle(namespace, { create: true });
  } catch {
    return undefined;
  }
  const fileName = (key: string) => key.replace(/[^a-zA-Z0-9._-]/g, '_');
  // FileSystemDirectoryHandle.keys() is in the spec but missing from the DOM lib.
  const names = () => (dir as unknown as { keys(): AsyncIterable<string> }).keys();

  const get = async (key: string): Promise<ArrayBuffer | undefined> => {
    try {
      const handle = await dir.getFileHandle(fileName(key));
      const buf = await (await handle.getFile()).arrayBuffer();
      // Zero-byte files are crash artifacts: createWritable commits atomically
      // on close(), but a create leaves an empty file if the tab dies first.
      // Nonzero truncation is not detected.
      return buf.byteLength > 0 ? buf : undefined;
    } catch {
      return undefined;
    }
  };

  return {
    get,
    async getOrCompute(key, compute) {
      const cached = await get(key);
      if (cached) return cached;
      const value = await compute();
      const bytes = new Uint8Array(value.byteLength);
      bytes.set(new Uint8Array(value.buffer, value.byteOffset, value.byteLength));
      try {
        const handle = await dir.getFileHandle(fileName(key), { create: true });
        const writable = await handle.createWritable();
        await writable.write(bytes);
        await writable.close();
      } catch {
        // Quota or write failure: serve uncached.
      }
      return bytes.buffer;
    },
    async deletePrefix(prefix) {
      // Names are the mangled keys, so mangle the prefix to match them.
      const want = fileName(prefix);
      for await (const name of names()) {
        if (name.startsWith(want)) await dir.removeEntry(name).catch(() => {});
      }
    },
    async clear() {
      for await (const name of names()) {
        await dir.removeEntry(name).catch(() => {});
      }
    },
  };
}
