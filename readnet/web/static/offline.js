// ReadNet with no server: everything runs in this browser tab.
//
// The page (app.js) talks to /api/... exactly as it does with the local
// server. This file answers those calls itself:
//   - the scoring code is the project's own Python (readnet/), run by Pyodide;
//   - the Hindi speech model runs with onnxruntime-web, downloaded once and
//     kept in the browser's cache storage for later visits.
// No account, no key, and the recording never leaves the computer.

(() => {
  const CONFIG = window.READNET_CONFIG || {};
  const MODEL_URL = CONFIG.modelUrl || "https://huggingface.co/madhushripatil032003/readnet-onnx/resolve/main/hi.onnx";
  const VOCAB_URL = CONFIG.vocabUrl || "https://huggingface.co/madhushripatil032003/readnet-onnx/resolve/main/vocab_hi.json";
  const MODEL_MB = CONFIG.modelMb || 378;
  const PYODIDE = "https://cdn.jsdelivr.net/pyodide/v0.27.7/full/";
  const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  const CACHE = "readnet-models-v1";

  // -- a small status bar ------------------------------------------------------------
  const bar = document.createElement("div");
  bar.className = "engine-status";
  bar.innerHTML = `<span class="engine-text">Starting…</span><div class="engine-progress"><div></div></div>`;
  const showStatus = (text, fraction) => {
    bar.querySelector(".engine-text").textContent = text;
    bar.querySelector(".engine-progress").hidden = fraction == null;
    if (fraction != null) bar.querySelector(".engine-progress div").style.width = `${Math.round(fraction * 100)}%`;
    bar.hidden = false;
  };
  document.addEventListener("DOMContentLoaded", () => document.body.appendChild(bar));

  const loadScript = (src) => new Promise((resolve, reject) => {
    const s = Object.assign(document.createElement("script"), { src, onload: resolve, onerror: () => reject(new Error(`Could not load ${src}`)) });
    document.head.appendChild(s);
  });

  async function fetchModel(url, onProgress) {
    let cache = null;
    try { cache = await caches.open(CACHE); } catch { /* no cache storage: download each visit */ }
    const hit = cache && await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Model download failed (HTTP ${res.status})`);
    const total = Number(res.headers.get("Content-Length")) || MODEL_MB * 1e6;
    const reader = res.body.getReader();
    const bytes = new Uint8Array(total);
    let got = 0;
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (got + value.length <= bytes.length) bytes.set(value, got); else chunks.push(value);
      got += value.length;
      onProgress(got, total);
    }
    let out = bytes.subarray(0, Math.min(got, bytes.length));
    if (chunks.length) {  // Content-Length was missing or wrong: stitch the overflow
      const full = new Uint8Array(got);
      full.set(out); let at = out.length;
      for (const c of chunks) { full.set(c, at); at += c.length; }
      out = full;
    }
    try { await cache?.put(url, new Response(out.slice())); } catch { /* quota: fine, just not cached */ }
    return out;
  }

  // -- start-up: Python first (the page needs it), then the model ---------------------
  let py, rb;
  const pythonReady = (async () => {
    showStatus("Loading the scoring engine (first visit takes a moment)…");
    await loadScript(`${PYODIDE}pyodide.js`);
    py = await loadPyodide({ indexURL: PYODIDE });
    await py.loadPackage("numpy");
    const zip = await (await fetch("readnet.zip")).arrayBuffer();
    py.unpackArchive(zip, "zip", { extractDir: "/home/pyodide/app" });
    py.runPython("import sys; sys.path.insert(0, '/home/pyodide/app')");
    rb = py.pyimport("readnet.web.browser");
  })();

  let session, vocabJson;
  const modelReady = pythonReady.then(async () => {
    await loadScript(`${ORT}ort.min.js`);
    ort.env.wasm.wasmPaths = ORT;
    vocabJson = await (await fetch(VOCAB_URL)).text();
    const cached = await caches.open(CACHE).then((c) => c.match(MODEL_URL)).catch(() => null);
    showStatus(cached ? "Loading the Hindi speech model from this browser's cache…"
                      : `Downloading the Hindi speech model (${MODEL_MB} MB, once; later visits load it from this browser)…`, cached ? null : 0);
    const bytes = await fetchModel(MODEL_URL, (got, total) =>
      showStatus(`Downloading the Hindi speech model: ${Math.round(got / 1e6)} of ${Math.round(total / 1e6)} MB (once)`, got / total));
    showStatus("Starting the Hindi speech model…");
    session = await ort.InferenceSession.create(bytes, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    showStatus("Ready: record a reading and press Score.");
    setTimeout(() => { bar.hidden = true; }, 4000);
  }).catch((err) => {
    showStatus(`The speech model could not load: ${err.message}. Typed transcripts still work.`);
    throw err;
  });

  // -- audio -> 16 kHz samples -> model -------------------------------------------------
  async function samples16k(b64) {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
    const ctx = new AudioContext();
    const decoded = await ctx.decodeAudioData(bytes.buffer);
    ctx.close();
    const offline = new OfflineAudioContext(1, Math.max(1, Math.ceil(decoded.duration * 16000)), 16000);
    const src = offline.createBufferSource();
    src.buffer = decoded;
    src.connect(offline.destination);
    src.start();
    return (await offline.startRendering()).getChannelData(0);
  }

  function normalise(x) {  // what the model's feature extractor does: zero mean, unit variance
    let mean = 0; for (const v of x) mean += v; mean /= x.length;
    let variance = 0; for (const v of x) variance += (v - mean) ** 2; variance /= x.length;
    const scale = 1 / Math.sqrt(variance + 1e-7);
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = (x[i] - mean) * scale;
    return out;
  }

  async function score(body) {
    await pythonReady;
    if (body.engine === "typed") return JSON.parse(rb.score(JSON.stringify(body)));
    if (!body.audio_b64) throw new Error("Record or upload the child's reading first.");
    if (!session) showStatus("Waiting for the Hindi speech model to finish loading…");
    await modelReady;
    const started = performance.now();
    showStatus("Listening…");
    const audio = await samples16k(body.audio_b64);
    if (audio.length < 16000 * 0.3) throw new Error("The recording is too short: read the text, then press Stop.");
    const out = await session.run({ input_values: new ort.Tensor("float32", normalise(audio), [1, audio.length]) });
    const lp = out.log_probs;
    const [, frames, vocabSize] = lp.dims;
    const elapsed = (performance.now() - started) / 1000;
    const result = rb.score(JSON.stringify(body), lp.data, frames, vocabSize, audio.length / 16000, vocabJson, elapsed);
    bar.hidden = true;
    return JSON.parse(result);
  }

  async function handle(path, body) {
    if (path.startsWith("/api/config")) {
      await pythonReady;
      bar.hidden = !!session;
      if (!session) showStatus("Scoring engine ready. The Hindi speech model is still loading…");
      return JSON.parse(rb.config("hi"));
    }
    if (path.startsWith("/api/score")) return score(body);
    if (path.startsWith("/api/next")) { await pythonReady; return JSON.parse(rb.next_step(JSON.stringify(body.outcomes || {}))); }
    if (path.startsWith("/api/dispute")) return { saved: "nowhere: the online demo stores nothing" };
    throw new Error(`Unknown request ${path}`);
  }

  const realFetch = window.fetch.bind(window);
  window.fetch = async (input, init) => {
    // Pyodide and onnxruntime also call fetch, with URL or Request objects: only
    // this page's own /api/ calls are answered here, everything else passes through.
    const raw = typeof input === "string" ? input : input instanceof URL ? input.href : (input && input.url) || String(input);
    const url = new URL(raw, location.href);
    if (url.origin !== location.origin || !url.pathname.startsWith("/api/")) return realFetch(input, init);
    try {
      const data = await handle(url.pathname + url.search, init && init.body ? JSON.parse(init.body) : null);
      return new Response(JSON.stringify(data), { status: 200, headers: { "Content-Type": "application/json" } });
    } catch (err) {
      return new Response(JSON.stringify({ error: String(err && err.message || err) }), { status: 400, headers: { "Content-Type": "application/json" } });
    }
  };
})();
