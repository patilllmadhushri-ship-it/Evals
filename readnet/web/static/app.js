// ReadNet test bench. Plain JavaScript, no framework.
// Records the child as 16-bit PCM WAV in the browser, sends it to /api/score,
// and walks the ASER order with /api/next.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = {
  config: null,
  language: "hi",
  outcomes: {},    // level -> outcome from /api/score
  results: {},     // level -> full score response
  current: "PARAGRAPH",
  pending: null,   // score response awaiting Accept
  texts: {},       // "hi:PARAGRAPH" -> edited text
};

// -- API -------------------------------------------------------------------------------

async function api(path, body) {
  const res = await fetch(path, body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  const data = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
  if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

// -- recording ---------------------------------------------------------------------------

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const str = (o, s) => [...s].forEach((c, i) => view.setUint8(o + i, c.charCodeAt(0)));
  str(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); str(8, "WAVE");
  str(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  str(36, "data"); view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",")[1]);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// One recorder per input area: record, or upload, or type.
class Recorder {
  constructor(container, { typedPlaceholder }) {
    this.container = container;
    this.typedPlaceholder = typedPlaceholder;
    this.audio = null; // { b64, filename }
    this.render();
  }

  get typed() { return this.container.querySelector(".typed")?.value || ""; }

  render() {
    const typedMode = currentEngine() === "typed";
    this.stop(true);
    this.audio = null;
    this.container.innerHTML = typedMode
      ? `<label class="field" style="margin:0"><span class="field-label">What the child said</span>
           <textarea class="typed deva" rows="3" placeholder="${esc(this.typedPlaceholder)}"></textarea></label>`
      : `<div class="input-row">
           <button class="btn btn-record" type="button"><span class="dot"></span><span class="rec-label">Record</span></button>
           <span class="timer">0:00</span>
           <span class="input-or">or</span>
           <label class="btn btn-secondary upload-label">Upload a recording
             <input type="file" accept=".wav,.mp3,.flac,.ogg,audio/*"></label>
           <span class="status file-status"></span>
         </div>
         <audio class="playback" controls hidden></audio>`;
    if (typedMode) return;
    this.container.querySelector(".btn-record").addEventListener("click", () => (this.stream ? this.stop() : this.start()));
    this.container.querySelector("input[type=file]").addEventListener("change", (e) => this.upload(e.target.files[0]));
  }

  async start() {
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      this.container.querySelector(".file-status").textContent = "Microphone not available. Upload a recording instead.";
      return;
    }
    this.ctx = new AudioContext();
    this.source = this.ctx.createMediaStreamSource(this.stream);
    this.node = this.ctx.createScriptProcessor(4096, 1, 1);
    this.chunks = [];
    this.node.onaudioprocess = (e) => this.chunks.push(new Float32Array(e.inputBuffer.getChannelData(0)));
    this.source.connect(this.node);
    this.node.connect(this.ctx.destination);
    const btn = this.container.querySelector(".btn-record");
    btn.classList.add("recording");
    btn.querySelector(".rec-label").textContent = "Stop";
    const started = Date.now();
    this.timer = setInterval(() => {
      const s = Math.floor((Date.now() - started) / 1000);
      this.container.querySelector(".timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
    }, 250);
  }

  async stop(silent = false) {
    if (!this.stream) return;
    clearInterval(this.timer);
    this.node.disconnect(); this.source.disconnect();
    this.stream.getTracks().forEach((t) => t.stop());
    const rate = this.ctx.sampleRate;
    await this.ctx.close();
    this.stream = null;
    if (silent) return;
    const total = this.chunks.reduce((n, c) => n + c.length, 0);
    const samples = new Float32Array(total);
    let offset = 0;
    for (const c of this.chunks) { samples.set(c, offset); offset += c.length; }
    const blob = encodeWav(samples, rate);
    this.audio = { b64: await blobToBase64(blob), filename: "recording.wav" };
    const btn = this.container.querySelector(".btn-record");
    btn.classList.remove("recording");
    btn.querySelector(".rec-label").textContent = "Record again";
    this.showPlayback(blob, "Recorded.");
  }

  async upload(file) {
    if (!file) return;
    this.audio = { b64: await blobToBase64(file), filename: file.name };
    this.showPlayback(file, `Using ${file.name}.`);
  }

  showPlayback(blob, message) {
    const player = this.container.querySelector(".playback");
    player.src = URL.createObjectURL(blob);
    player.hidden = false;
    this.container.querySelector(".file-status").textContent = message;
  }
}

// -- settings --------------------------------------------------------------------------

function currentEngine() { return $("engine").value; }

async function loadConfig(language) {
  state.config = await api(`/api/config?language=${language}`);
  state.language = language;
  $("language-switch").innerHTML = state.config.languages
    .map((l) => `<button type="button" data-lang="${l.code}" class="${l.code === language ? "active" : ""}" role="radio" aria-checked="${l.code === language}">${esc(l.name)}</button>`)
    .join("");
  $("language-switch").querySelectorAll("button").forEach((b) => b.addEventListener("click", () => {
    if (b.dataset.lang !== state.language) loadConfig(b.dataset.lang).then(resetSession);
  }));
  const previous = $("engine").value;
  $("engine").innerHTML = state.config.engines.map((e) => `<option value="${e.key}">${esc(e.label)}</option>`).join("");
  if (state.config.engines.some((e) => e.key === previous)) $("engine").value = previous;
  $("model-id").value = state.config.default_local_model;
  $("quick-text").value = language === "hi" ? "चाँद और गाँव" : "माझे घर";
  onEngineChange();
}

function onEngineChange() {
  const engine = state.config.engines.find((e) => e.key === currentEngine());
  $("engine-note").textContent = engine?.note || "";
  $("local-fields").hidden = currentEngine() !== "local";
  taskRecorder?.render();
  quickRecorder?.render();
  if (currentEngine() === "typed" && quickRecorder) {
    const typed = $("quick-input").querySelector(".typed");
    if (typed && state.language === "hi") typed.value = "चाद और गाव";
  }
}

// -- rendering a result -------------------------------------------------------------------

function wordChips(result) {
  return result.ops.map((op) => {
    const gop = op.gop != null ? ` · GOP ${op.gop.toFixed(1)}` : "";
    if (op.kind === "match") {
      if (op.rule || op.note) return `<span class="chip forgiven">${esc(op.ref)}<small>${esc(op.rule)}${esc(op.note ? " · " + op.note : "")}${gop}</small></span>`;
      return `<span class="chip ok">${esc(op.ref)}<small>read${gop}</small></span>`;
    }
    if (op.kind === "substitution") {
      const why = op.counts_as_mistake ? (op.subtypes.join("/") || op.category) : op.rule;
      return `<span class="chip ${op.counts_as_mistake ? "bad" : "forgiven"}">${esc(op.ref)}<small>heard ${esc(op.hyp)} · ${esc(why)}${gop}</small></span>`;
    }
    if (op.kind === "deletion") return `<span class="chip bad"><span class="struck">${esc(op.ref)}</span><small>not read</small></span>`;
    const label = op.rule ? `${op.category} · ${op.rule}` : op.category;
    return `<span class="chip ${op.counts_as_mistake ? "bad" : "extra"}">+${esc(op.hyp)}<small>${esc(label)}</small></span>`;
  }).join("");
}

const LEGEND = `<div class="legend">
  <span class="chip ok">read right</span><span class="chip forgiven">forgiven (rule id)</span>
  <span class="chip bad">mistake</span><span class="chip extra">extra sound, not counted</span></div>`;

function resultCard(result, engineLabel) {
  const o = result.outcome;
  const pace = o.wpm != null ? `<p class="reasons">Pace: ${Math.round(o.wpm)} words per minute${o.longest_pause != null ? `, longest pause ${o.longest_pause.toFixed(1)}s` : ""} (measured, not enforced)</p>` : "";
  const doubts = result.acoustic_doubts.length
    ? `<div class="warning">The audio does not sound like: <span class="deva">${esc(result.acoustic_doubts.join(", "))}</span>. The transcript says correct, so these are flagged for review, not counted.</div>` : "";
  const changes = result.changes.length
    ? `<div class="table-wrap"><table><tr><th>Rule</th><th>Before</th><th>After</th></tr>${result.changes.map((c) => `<tr><td>${esc(c.rule)}</td><td class="deva">${esc(c.before)}</td><td class="deva">${esc(c.after)}</td></tr>`).join("")}</table></div>`
    : "<p class='muted'>No rule changed the heard text.</p>";
  return `<div class="card">
    <div class="stats">
      <div class="stat"><div class="stat-label">Verdict</div><div class="stat-value ${o.passed ? "pass" : "fail"}">${o.passed ? "Pass" : "Not yet"}</div></div>
      <div class="stat"><div class="stat-label">Mistakes</div><div class="stat-value">${o.mistakes}</div></div>
      <div class="stat"><div class="stat-label">Read correctly</div><div class="stat-value">${o.correct} / ${o.total}</div></div>
      <div class="stat"><div class="stat-label">Engine time</div><div class="stat-value">${result.seconds.toFixed(1)}s</div></div>
    </div>
    <p class="reasons">${esc(o.reasons.join(" · "))}</p>
    <p class="heard"><b>Heard</b> (${esc(engineLabel)}): <span>${esc(result.transcript) || "nothing"}</span></p>
    ${pace}${doubts}${LEGEND}
    <div class="words">${wordChips(result)}</div>
    <details><summary>Show the working</summary><div class="inner">
      <p><b>Text after normalisation:</b> <span class="deva">${esc(result.canonical_normalized)}</span></p>
      <p><b>Heard after normalisation:</b> <span class="deva">${esc(result.transcript_normalized)}</span></p>
      ${changes}
    </div></details>
    <div class="result-extra"></div>
  </div>`;
}

function engineLabel() {
  const e = state.config.engines.find((x) => x.key === currentEngine());
  return (e?.label || "").replace(/ \(.*$/, "");
}

async function score(level, text, recorder, target) {
  const status = document.createElement("p");
  status.className = "status";
  status.textContent = currentEngine() === "typed" ? "Scoring..." : "Listening...";
  target.innerHTML = "";
  target.appendChild(status);
  try {
    const result = await api("/api/score", {
      language: state.language, level, text, engine: currentEngine(),
      typed: recorder.typed, audio_b64: recorder.audio?.b64, filename: recorder.audio?.filename,
      model_id: $("model-id").value, gop_threshold: Number($("gop").value),
    });
    target.innerHTML = resultCard(result, engineLabel());
    return result;
  } catch (err) {
    target.innerHTML = `<div class="error">${esc(err.message)}</div>`;
    return null;
  }
}

// -- the ASER session -----------------------------------------------------------------------

let taskRecorder = null;
let quickRecorder = null;

function taskText(level) {
  return state.texts[`${state.language}:${level}`] ?? state.config.content[level];
}

function pathSoFar() {
  const path = ["PARAGRAPH"];
  const o = state.outcomes;
  if (o.PARAGRAPH) path.push(o.PARAGRAPH.passed ? "STORY" : "WORD");
  if (o.WORD && !o.WORD.passed) path.push("LETTER");
  return path;
}

function renderStepper(done) {
  const items = pathSoFar().map((level) => {
    const o = state.outcomes[level];
    const cls = !done && level === state.current ? "now" : o ? (o.passed ? "pass" : "fail") : "";
    const mark = o ? (o.passed ? "passed" : "not yet") : level === state.current ? "now" : "";
    return `<li class="${cls}">${esc(state.config.tasks[level].title)}${mark ? ` · ${mark}` : ""}</li>`;
  });
  if (!done) items.push("<li>Level</li>");
  $("stepper").innerHTML = items.join("");
}

function showTask() {
  const level = state.current;
  const copy = state.config.tasks[level];
  const child = $("child").value.trim();
  $("final-view").hidden = true;
  $("task-view").hidden = false;
  renderStepper(false);
  $("task-title").textContent = child ? `${copy.title}: ${child}` : copy.title;
  $("task-instruction").textContent = copy.instruction;
  $("task-rule").textContent = copy.rule;
  $("task-text").textContent = taskText(level);
  $("task-text").hidden = false;
  $("task-text-editor").hidden = true;
  $("edit-text").textContent = "Change text";
  state.pending = null;
  $("task-result").innerHTML = "";
  $("task-input").innerHTML = `<div class="recorder"></div>
    <div class="actions"><button class="btn btn-primary" id="score-btn" type="button">Score this reading</button></div>`;
  taskRecorder = new Recorder($("task-input").querySelector(".recorder"), {
    typedPlaceholder: "Type exactly what you heard, including repeats and restarts",
  });
  $("score-btn").addEventListener("click", scoreTask);
}

async function scoreTask() {
  const level = state.current;
  $("score-btn").disabled = true;
  const result = await score(level, taskText(level), taskRecorder, $("task-result"));
  $("score-btn").disabled = false;
  if (!result) return;
  state.pending = result;
  const extra = $("task-result").querySelector(".result-extra");
  extra.innerHTML = `
    <details><summary>The assessor disagrees with this verdict</summary><div class="inner">
      <div class="row">
        <label class="field narrow"><span class="field-label">Mistakes you counted</span>
          <input type="number" min="0" id="human-mistakes" value="${result.outcome.mistakes}"></label>
        <label class="field grow"><span class="field-label">What did the child actually read?</span>
          <input type="text" id="dispute-note" placeholder="Which word did the app get wrong?"></label>
      </div>
      <div class="actions"><button class="btn btn-secondary" id="save-dispute" type="button">Save as a field case</button>
        <span class="saved" id="dispute-saved"></span></div>
    </div></details>
    <div class="actions">
      <button class="btn btn-primary" id="accept-btn" type="button">Accept and continue</button>
      <button class="btn btn-secondary" id="again-btn" type="button">Record again</button>
    </div>`;
  $("accept-btn").addEventListener("click", accept);
  $("again-btn").addEventListener("click", showTask);
  $("save-dispute").addEventListener("click", async () => {
    const saved = await api("/api/dispute", {
      language: state.language, level, text: taskText(level), transcript: result.transcript,
      system_mistakes: result.outcome.mistakes, human_mistakes: Number($("human-mistakes").value),
      note: $("dispute-note").value,
    });
    $("dispute-saved").textContent = `Saved to ${saved.saved}`;
  });
}

async function accept() {
  const level = state.current;
  state.outcomes[level] = state.pending.outcome;
  state.results[level] = state.pending;
  const next = await api("/api/next", { outcomes: state.outcomes });
  if (next.next) {
    state.current = next.next;
    showTask();
    window.scrollTo({ top: 0, behavior: "smooth" });
  } else {
    showFinal(next.placement);
  }
}

function showFinal(placement) {
  const child = $("child").value.trim();
  $("task-view").hidden = true;
  $("final-view").hidden = false;
  renderStepper(true);

  const rows = Object.entries(state.results).map(([level, r]) => `<tr>
      <td>${esc(state.config.tasks[level].title)}</td>
      <td>${r.outcome.passed ? "pass" : "not yet"}</td>
      <td>${r.outcome.mistakes}</td>
      <td>${r.outcome.correct}/${r.outcome.total}</td>
      <td class="deva">${esc(r.transcript)}</td></tr>`).join("");

  const sounds = {};
  Object.values(state.results).forEach((r) => Object.entries(r.mistake_profile).forEach(([k, v]) => {
    if (k.startsWith("sound:")) sounds[k.slice(6)] = (sounds[k.slice(6)] || 0) + v;
  }));
  const max = Math.max(1, ...Object.values(sounds));
  const bars = Object.keys(sounds).length
    ? `<h3 class="section-title">Mistake profile: the kinds of sounds this child gets wrong</h3>
       <div class="bars">${Object.entries(sounds).sort((a, b) => b[1] - a[1]).map(([k, v]) =>
         `<span>${esc(k.replace(/_/g, " "))}</span><div class="bar" style="width:${(v / max) * 100}%"></div><span>${v}</span>`).join("")}</div>`
    : "";

  const detail = Object.entries(state.results).map(([level, r]) =>
    `<details><summary>${esc(state.config.tasks[level].title)}: word by word</summary><div class="inner"><div class="words">${wordChips(r)}</div></div></details>`).join("");

  $("final-view").innerHTML = `<div class="card">
      <p class="level-eyebrow">${child ? `${esc(child)} reads at` : "Reading level"}</p>
      <p class="level-name">${esc(placement.label)}</p>
      <div class="callout">${esc(placement.next_step)}</div>
      <h3 class="section-title">How the app got there</h3>
      <ul class="path">${placement.path.map((p) => `<li>${esc(p)}</li>`).join("")}</ul>
      <h3 class="section-title">Tasks</h3>
      <div class="table-wrap"><table><tr><th>Task</th><th>Verdict</th><th>Mistakes</th><th>Read correctly</th><th>Heard</th></tr>${rows}</table></div>
      ${bars}
      ${detail}
      <div class="actions">
        <button class="btn btn-secondary" id="download-btn" type="button">Download result</button>
        <button class="btn btn-primary" id="another-btn" type="button">Test another child</button>
      </div>
    </div>`;
  $("download-btn").addEventListener("click", () => {
    const report = { language: state.language, child: child || null, level: placement.label, path: placement.path,
      mistake_profile: sounds, tasks: state.results };
    const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: "application/json" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: "readnet_result.json" });
    a.click();
    URL.revokeObjectURL(url);
  });
  $("another-btn").addEventListener("click", () => { $("child").value = ""; resetSession(); });
}

function resetSession() {
  state.outcomes = {};
  state.results = {};
  state.current = "PARAGRAPH";
  showTask();
  quickRecorder.render();
  $("quick-result").innerHTML = "";
}

// -- quick check ---------------------------------------------------------------------------

function setupQuick() {
  $("quick-input").innerHTML = `<div class="recorder"></div>
    <div class="actions"><button class="btn btn-primary" id="quick-btn" type="button">Score</button></div>`;
  quickRecorder = new Recorder($("quick-input").querySelector(".recorder"), { typedPlaceholder: "What was said" });
  $("quick-btn").addEventListener("click", () => score($("quick-level").value, $("quick-text").value, quickRecorder, $("quick-result")));
}

// -- wiring ---------------------------------------------------------------------------------

document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
  $("tab-assessment").hidden = tab.dataset.tab !== "assessment";
  $("tab-quick").hidden = tab.dataset.tab !== "quick";
}));

$("engine").addEventListener("change", onEngineChange);
$("gop").addEventListener("input", () => { $("gop-value").textContent = Number($("gop").value).toFixed(1); });
$("new-child").addEventListener("click", () => { $("child").value = ""; resetSession(); });
$("edit-text").addEventListener("click", () => {
  const editing = $("task-text-editor").hidden;
  if (editing) {
    $("task-text-editor").value = taskText(state.current);
    $("edit-text").textContent = "Done";
  } else {
    state.texts[`${state.language}:${state.current}`] = $("task-text-editor").value.trim() || state.config.content[state.current];
    $("task-text").textContent = taskText(state.current);
    $("edit-text").textContent = "Change text";
  }
  $("task-text-editor").hidden = !editing;
  $("task-text").hidden = editing;
});

(async function init() {
  await loadConfig("hi");
  setupQuick();
  showTask();
  onEngineChange();
})();
