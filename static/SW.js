/* NexaScholar AI service worker: caches the app shell, never touches /api/* */
const VERSION = 'nexascholar-shell-v1';
const SHELL = ['/', '/static/manifest.json', '/static/favicon.svg', '/static/icon-192.png', '/static/icon-512.png'];

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(VERSION).then((c) => c.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter((k) => k !== VERSION).map((k) => caches.delete(k)))).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const req = event.request, url = new URL(req.url);
  // API calls (analysis, chat streaming, TTS, ...) always go straight to the network
  if (url.pathname.startsWith('/api/')) { event.respondWith(fetch(req)); return; }
  if (req.method !== 'GET' || url.origin !== self.location.origin) return;

  // The page itself: network first (always fresh), cached copy when offline
  if (req.mode === 'navigate' || url.pathname === '/') {
    event.respondWith(
      fetch(req).then((res) => { const copy = res.clone(); caches.open(VERSION).then((c) => c.put('/', copy)); return res; })
        .catch(() => caches.match('/').then((r) => r || Response.error()))
    );
    return;
  }
  // Static files: cached copy first, refreshed in the background
  if (url.pathname.startsWith('/static/') || url.pathname === '/manifest.json') {
    event.respondWith(
      caches.match(req).then((hit) => {
        const refresh = fetch(req).then((res) => { if (res.ok) { const copy = res.clone(); caches.open(VERSION).then((c) => c.put(req, copy)); } return res; }).catch(() => hit);
        return hit || refresh;
      })
    );
  }
});