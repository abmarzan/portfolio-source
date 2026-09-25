// sw.js — Advanced but minimal

const CACHE_VERSION = 'v5';
const CACHE_NAME = `marzan-portfolio-${CACHE_VERSION}`;

// Core assets to precache (no query strings here)
const PRECACHE_URLS = [
  '/', 
  '/index.html',
  '/style.css',
  '/script.js',
  '/media/home/Logo.png'
];

// Normalize requests: strip ?v= for CSS/JS/images so cache hits
function normalizeRequest(req) {
  const url = new URL(req.url);
  if (url.origin === self.location.origin) {
    if (url.pathname.endsWith('.css') || url.pathname.endsWith('.js') || url.pathname.match(/\.(png|jpg|jpeg|svg|gif)$/)) {
      return new Request(url.origin + url.pathname, { method: req.method, headers: req.headers });
    }
  }
  return req;
}

// Install: precache core assets
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_URLS))
  );
  self.skipWaiting();
});

// Activate: clear old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((key) => key !== CACHE_NAME && caches.delete(key)))
    )
  );
  self.clients.claim();
});

// Fetch strategy
self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;

  const req = normalizeRequest(event.request);
  const accept = event.request.headers.get('accept') || '';
  const isHTML = accept.includes('text/html') || event.request.mode === 'navigate';

  if (isHTML) {
    // Network-first for HTML
    event.respondWith(
      fetch(event.request)
        .then((res) => {
          const clone = res.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
          return res;
        })
        .catch(() => caches.match(req).then((cached) => cached || caches.match('/index.html')))
    );
  } else {
    // Stale-while-revalidate for static assets
    event.respondWith(
      caches.match(req).then((cached) => {
        const fetchPromise = fetch(event.request)
          .then((res) => {
            const clone = res.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(req, clone));
            return res;
          })
          .catch(() => cached);
        return cached || fetchPromise;
      })
    );
  }
});
