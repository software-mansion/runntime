/** A fake Hub, for the tests that check which requests a weights load
 *  sends. */

/** A fetch that serves `file` at every URL, with 206 for a Range request,
 *  and answers a HEAD with `head`. `calls` lists each request as
 *  `METHOD url`. */
export function hubServer(
  file: Uint8Array,
  head: () => Promise<Response> = () => Promise.resolve(new Response(null)),
) {
  const calls: string[] = [];
  const fetchFn: typeof fetch = (input, init) => {
    const method = init?.method ?? 'GET';
    calls.push(`${method} ${String(input)}`);
    if (method === 'HEAD') return head();
    const range = /^bytes=(\d+)-(\d+)$/.exec(new Headers(init?.headers).get('Range') ?? '');
    const body = range ? file.slice(Number(range[1]), Number(range[2]) + 1) : file.slice();
    return Promise.resolve(new Response(body, { status: range ? 206 : 200 }));
  };
  return { fetchFn, calls };
}
