/* DeskLens HQ service worker — network-first for the app, cache as offline fallback */
const CACHE='dlhq-v7';
const SHELL=['./desklens-hq.html','./manifest.json','./icon-192.png','./icon-512.png','./icon-maskable.png'];
self.addEventListener('install',e=>{
  self.skipWaiting();
  e.waitUntil(caches.open(CACHE).then(c=>c.addAll(SHELL).catch(()=>{})));
});
self.addEventListener('activate',e=>{
  e.waitUntil(caches.keys().then(ks=>Promise.all(ks.filter(k=>k!==CACHE).map(k=>caches.delete(k)))).then(()=>self.clients.claim()));
});
self.addEventListener('fetch',e=>{
  const url=new URL(e.request.url);
  if(e.request.method!=='GET')return;
  // never intercept Firebase / Google APIs — always live
  if(/googleapis|gstatic|firebase|firebaseio|google\.com/.test(url.hostname))return;
  if(url.origin!==self.location.origin)return;
  e.respondWith(
    fetch(e.request).then(r=>{
      const copy=r.clone();
      caches.open(CACHE).then(c=>c.put(e.request,copy)).catch(()=>{});
      return r;
    }).catch(()=>caches.match(e.request).then(r=>r||caches.match('./desklens-hq.html')))
  );
});
