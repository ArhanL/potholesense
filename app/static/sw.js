/* Service worker: makes a survey possible with nothing but the phone.
 *
 * Open the capture page once while the phone can reach the server and
 * everything needed for a drive - the page, the inference runtime, the model
 * weights - is stored on the device. After that the survey runs with the
 * laptop switched off and no signal, which is the point of on-device
 * inference and the reason this file exists.
 *
 * Strategy is deliberately split. The app shell is network-first so a fix to
 * the client reaches the phone the next time it is in range. The runtime and
 * the weights are cache-first, because they are large, immutable, and must
 * never be re-fetched on a road with no coverage.
 */
const CACHE = 'potholesense-v1';

const SHELL = [
  '/', '/static/detector.js', '/static/queue.js',
  '/static/vendor/ort/ort.wasm.min.js',
  '/static/vendor/ort/ort-wasm-simd-threaded.mjs',
  '/static/vendor/ort/ort-wasm-simd-threaded.wasm',
];

const isImmutable = url =>
  url.pathname.startsWith('/static/vendor/') || url.pathname.startsWith('/models/');

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE)
      // Individually, so one missing file cannot fail the whole install.
      .then(c => Promise.all(SHELL.map(u => c.add(u).catch(() => null))))
      .then(() => self.skipWaiting()));
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim()));
});

self.addEventListener('fetch', event => {
  const { request } = event;
  if (request.method !== 'GET') return;              // never cache uploads
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith('/api/')) return;      // always live

  if (isImmutable(url)) {
    event.respondWith(
      caches.match(request).then(hit => hit || fetch(request).then(res => {
        if (res.ok) caches.open(CACHE).then(c => c.put(request, res.clone()));
        return res;
      })));
    return;
  }

  event.respondWith(
    fetch(request)
      .then(res => {
        if (res.ok) {
          const copy = res.clone();
          caches.open(CACHE).then(c => c.put(request, copy));
        }
        return res;
      })
      .catch(() => caches.match(request)));
});
