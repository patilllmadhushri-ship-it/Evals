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

  micProblem(message) {
    let box = this.container.querySelector(".mic-error");
    if (!box) {
      box = Object.assign(document.createElement("div"), { className: "error mic-error" });
      this.container.appendChild(box);
    }
    box.innerHTML = `${esc(message)} You can still use <b>Upload a recording</b>.`;
  }

  async start() {
    this.container.querySelector(".mic-error")?.remove();
    if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
      this.micProblem(`This browser only allows the microphone on a secure page. Open http://localhost:${location.port || 8600} in Chrome or Edge (use "localhost", not an IP address).`);
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      const reasons = {
        NotAllowedError: "Microphone permission is blocked. Click the icon at the left of the address bar, set Microphone to Allow, and try again. The preview pane inside the Claude app always blocks the microphone, so open this page in Chrome or Edge.",
        NotFoundError: "No microphone was found. Plug one in, or check Windows Settings > Privacy > Microphone.",
        NotReadableError: "The microphone is in use by another app, or Windows is blocking it (Settings > Privacy > Microphone > let desktop apps use it).",
      };
      this.micProblem(reasons[err.name] || `Could not start the microphone: ${err.message || err.name}.`);
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
  $("demo-banner").hidden = !state.config.public;
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
  $("gop-on").checked = state.config.gop_available;
  $("gop-note").textContent = state.config.gop_available ? "" :
    (state.config.engines.find((e) => e.key === "local")?.note || "");
  $("gop-fields").hidden = !$("gop-on").checked && state.config.gop_available;
  $("quick-text").value = language === "hi" ? "चाँद और गाँव" : "माझे घर";
  onEngineChange();
}

function onEngineChange() {
  const engine = state.config.engines.find((e) => e.key === currentEngine());
  $("engine-note").textContent = engine?.note || "";
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
    const low = op.gop != null && op.gop < Number($("gop").value) ? " lowgop" : "";
    if (op.kind === "match") {
      if (op.rule || op.note) return `<span class="chip forgiven${low}">${esc(op.ref)}<small>${esc(op.rule)}${esc(op.note ? " · " + op.note : "")}${gop}</small></span>`;
      return `<span class="chip ok${low}">${esc(op.ref)}<small>read${gop}</small></span>`;
    }
    if (op.kind === "substitution") {
      const why = op.counts_as_mistake ? (op.subtypes.join("/") || op.category) : op.rule;
      return `<span class="chip ${op.counts_as_mistake ? "bad" : "forgiven"}${low}">${esc(op.ref)}<small>heard ${esc(op.hyp)} · ${esc(why)}${gop}</small></span>`;
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
  return `<div class="card">
    <div class="stats">
      <div class="stat"><div class="stat-label">Verdict</div><div class="stat-value ${o.passed ? "pass" : "fail"}">${o.passed ? "Pass" : "Not yet"}</div></div>
      <div class="stat"><div class="stat-label">Mistakes</div><div class="stat-value">${o.mistakes}</div></div>
      <div class="stat"><div class="stat-label">Read correctly</div><div class="stat-value">${o.correct} / ${o.total}</div></div>
      <div class="stat"><div class="stat-label">Engine time</div><div class="stat-value">${result.seconds.toFixed(1)}s</div></div>
    </div>
    <p class="reasons">${esc(o.reasons.join(" · "))}</p>
    <p class="heard"><b>Heard</b> (${esc(engineLabel)}): <span>${esc(result.engine_transcript ?? result.transcript) || "nothing"}</span></p>
    ${letterCheck(result)}
    ${result.notes?.length ? `<div class="notes">${result.notes.map(esc).join("<br>")}</div>` : ""}
    ${pace}${doubts}${LEGEND}
    <div class="words">${wordChips(result)}</div>
    ${gopLine(result)}
    ${soundsView(result)}
    ${pipelineView(result)}
    <div class="result-extra"></div>
  </div>`;
}

function letterCheck(result) {
  if (!result.letter_check) return "";
  const rows = result.letter_check.map((c) => `<tr>
      <td class="deva">${esc(c.letter)}</td>
      <td class="deva"><b>${esc(c.heard)}</b></td>
      <td>${c.heard === c.letter ? "right" : "wrong"}</td>
      <td>${c.margin > 0 ? "+" : ""}${c.margin}</td>
      <td class="deva">${esc(c.candidates.join(" "))}</td></tr>`).join("");
  return `<h3 class="section-title">Letter check (the audio, letter by letter)</h3>
    <div class="table-wrap"><table>
      <tr><th>Shown</th><th>Sounded like</th><th>Result</th><th>Margin</th><th>Letters compared, best first</th></tr>${rows}
    </table></div>
    <p class="gop-line">Margin: how much better the shown letter fits than its closest rival. Negative means a rival fits better.</p>`;
}

function pipelineView(result) {
  if (!result.pipeline) return "";
  const steps = result.pipeline.map((step) => {
    let body = step.detail ? `<span class="deva">${esc(step.detail)}</span>` : "";
    if (step.rules) {
      body = step.rules.length
        ? step.rules.map(([rule, before, after]) => `<div class="rule-row"><span class="rule-id">${esc(rule)}</span>
            <span class="deva">${esc(before)}</span> <span class="arrow">to</span> <span class="deva">${esc(after)}</span></div>`).join("")
        : "<span class='muted-inline'>No rule changed it.</span>";
      body += `<div class="rule-result">Result: <span class="deva">${esc(step.result)}</span></div>`;
    }
    return `<li><div class="stage">${esc(step.stage)}</div><div class="stage-body">${body}</div></li>`;
  }).join("");
  return `<h3 class="section-title">How this was scored</h3><ol class="pipeline">${steps}</ol>`;
}

function soundsView(result) {
  const words = result.ops.filter((op) => op.units && op.units.length);
  if (!words.length) return "";
  const threshold = Number($("gop").value);
  const rows = words.map((op) => `<tr>
      <td class="deva">${esc(op.ref)}</td>
      <td>${op.start_s.toFixed(2)}–${op.end_s.toFixed(2)}s</td>
      <td class="deva">/${esc(op.units.map((u) => u.sounds).filter(Boolean).join(" "))}/</td>
      <td><div class="units">${op.units.filter((u) => u.sounds).map((u) => `<span class="unit${u.gop < threshold ? " low" : ""}">
          <b>/${esc(u.sounds)}/</b>${u.letter ? `<small class="deva">${esc(u.letter)}</small>` : ""}<small>${u.start_s.toFixed(2)}–${u.end_s.toFixed(2)}s</small><small>GOP ${u.gop > 0 ? "+" : ""}${u.gop}</small>${u.gop < threshold ? `<small class="deva">audio: ${u.heard ? esc(u.heard) : "nothing"}</small>` : ""}</span>`).join("")}</div></td>
    </tr>`).join("");
  return `<details class="sounds" open><summary>Sounds: forced alignment and GOP per phoneme</summary><div class="inner">
    <p class="gop-line">Each word is turned into its sounds (G2P), the recording is cut into 20 ms frames, forced alignment finds which frames carry each sound, and each sound gets a GOP: how much better the expected sound fits its frames than the strongest other sound. Red is below the threshold. With the Hindi model, an unwritten /ə/ shares its consonant's frames and score (shown together, e.g. /ɡʰ ə/).</p>
    <div class="table-wrap"><table><tr><th>Word</th><th>Said at</th><th>G2P</th><th>Each sound: time and GOP</th></tr>${rows}</table></div>
  </div></details>`;
}

function gopLine(result) {
  if (!result.gop_ran) return "";
  const scored = result.ops.filter((op) => op.gop != null);
  const low = scored.filter((op) => op.gop < Number($("gop").value));
  return `<p class="gop-line">Pronunciation (GOP) scored on ${scored.length} words. ${low.length
    ? `${low.length} below the threshold, underlined: <span class="deva">${esc(low.map((op) => op.ref).join(", "))}</span>.`
    : "None below the threshold."} GOP near 0 means the audio matches the expected sound; strongly negative means it does not.</p>`;
}

function engineLabel() {
  const e = state.config.engines.find((x) => x.key === currentEngine());
  return (e?.label || "").replace(/ \(.*$/, "");
}

async function score(level, text, recorder, target) {
  const status = document.createElement("p");
  status.className = "status";
  status.textContent = currentEngine() === "typed" ? "Scoring..." :
    $("gop-on").checked ? "Listening and checking pronunciation... (the first time loads the model, which can take a few minutes)" : "Listening...";
  target.innerHTML = "";
  target.appendChild(status);
  try {
    const result = await api("/api/score", {
      language: state.language, level, text, engine: currentEngine(),
      typed: recorder.typed, audio_b64: recorder.audio?.b64, filename: recorder.audio?.filename,
      model_id: $("model-id").value, gop_threshold: Number($("gop").value), gop: $("gop-on").checked, audio_decides: $("audio-decides").checked,
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
  const child = studentName();
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
  $("task-result").innerHTML = state.lastSaved ? `<p class="saved">${esc(state.lastSaved)}</p>` : "";
  state.lastSaved = "";
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

async function saveReading(level, text, result) {
  const sid = studentId();
  if (!sid) return "";
  const saved = await api("/api/attempts", {
    student_id: sid, language: state.language, task: level, text, result,
    engine: currentEngine(), threshold: Number($("gop").value),
  });
  return `Saved to ${studentName()}'s record: ${saved.sounds_saved} sounds, ${saved.sounds_wrong} to practise.`;
}

async function accept() {
  const level = state.current;
  state.outcomes[level] = state.pending.outcome;
  state.results[level] = state.pending;
  try {
    state.lastSaved = await saveReading(level, taskText(level), state.pending);
  } catch (err) {
    state.lastSaved = `Not saved: ${err.message}`;
  }
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
  const child = studentName();
  if (studentId()) {
    api("/api/sessions", { student_id: studentId(), language: state.language, placement }).catch(() => {});
  }
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
        ${studentId() ? `<button class="btn btn-secondary" id="open-record" type="button">Open ${esc(child)}'s record</button>` : ""}
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
  $("another-btn").addEventListener("click", () => { $("student").value = ""; resetSession(); });
  $("open-record")?.addEventListener("click", () => openStudent(studentId()));
}

function resetSession() {
  state.outcomes = {};
  state.results = {};
  state.current = "PARAGRAPH";
  showTask();
  quickRecorder.render();
  $("quick-result").innerHTML = "";
}

// -- students ------------------------------------------------------------------------------

let students = [];

function studentId() { return Number($("student").value) || null; }
function studentName() { return students.find((s) => s.id === studentId())?.name || ""; }

async function loadStudents(select) {
  students = (await api("/api/students")).students;
  const current = select ?? studentId();
  $("student").innerHTML = `<option value="">No student (not saved)</option>` +
    students.map((s) => `<option value="${s.id}">${esc(s.name)}${s.grade ? ` · class ${esc(s.grade)}` : ""}</option>`).join("");
  if (current && students.some((s) => s.id === current)) $("student").value = String(current);
}

async function addStudent() {
  try {
    const s = await api("/api/students", { name: $("new-name").value, grade: $("new-grade").value, language: state.language });
    $("new-name").value = ""; $("new-grade").value = "";
    await loadStudents(s.id);
    $("add-student-note").textContent = `Added ${s.name}.`;
    resetSession();
  } catch (err) {
    $("add-student-note").textContent = err.message;
  }
}

const LEVELS = ["Beginner", "Letter", "Word", "Paragraph", "Story"];
const day = (iso) => (iso || "").slice(0, 10);

async function renderRoster() {
  await loadStudents();
  $("roster").innerHTML = students.length ? `<h2 class="card-title">Students</h2>
    <div class="table-wrap"><table>
      <tr><th>Name</th><th>Class</th><th>Latest level</th><th>Readings</th><th>Last seen</th><th>Letters to practise</th></tr>
      ${students.map((s) => `<tr class="clickable" data-id="${s.id}">
        <td><b>${esc(s.name)}</b></td><td>${esc(s.grade || "")}</td><td>${esc(s.latest_level || "not placed yet")}</td>
        <td>${s.readings}</td><td>${day(s.last_seen)}</td>
        <td class="deva">${s.practise.map((l) => `<span class="pill">${esc(l)}</span>`).join(" ") || "none yet"}</td></tr>`).join("")}
    </table></div>`
    : `<p class="muted">No students yet. Add one under Setup, choose them, and every accepted reading is saved to their record.</p>`;
  $("roster").querySelectorAll("tr.clickable").forEach((tr) => tr.addEventListener("click", () => openStudent(Number(tr.dataset.id))));
}

function levelChart(sessions) {
  if (!sessions.length) return `<p class="muted">No full ASER test yet.</p>`;
  const w = 560, h = 170, left = 78, right = 16, top = 12, bottom = 28;
  const x = (i) => left + (sessions.length === 1 ? (w - left - right) / 2 : (i * (w - left - right)) / (sessions.length - 1));
  const y = (lv) => top + ((4 - lv) * (h - top - bottom)) / 4;
  const pts = sessions.map((s, i) => `${x(i)},${y(s.level_index)}`).join(" ");
  return `<svg class="level-chart" viewBox="0 0 ${w} ${h}" role="img" aria-label="ASER level over time">
    ${LEVELS.map((name, lv) => `<line x1="${left}" x2="${w - right}" y1="${y(lv)}" y2="${y(lv)}" class="grid"/>
      <text x="${left - 8}" y="${y(lv) + 4}" text-anchor="end" class="axis">${name}</text>`).join("")}
    <polyline points="${pts}" class="line"/>
    ${sessions.map((s, i) => `<circle cx="${x(i)}" cy="${y(s.level_index)}" r="4.5" class="dot"><title>${day(s.created_at)}: ${esc(s.level)}</title></circle>
      <text x="${x(i)}" y="${h - 8}" text-anchor="middle" class="axis">${day(s.created_at).slice(5)}</text>`).join("")}
  </svg>`;
}

async function openStudent(sid) {
  showTab("students");
  const d = await api(`/api/students/${sid}`);
  const recentDots = (r) => [...r].map((c) => `<span class="dot-${c === "1" ? "ok" : "bad"}" title="${c === "1" ? "said right" : "said wrong"}"></span>`).join("");
  const sounds = d.sounds.length ? `<div class="table-wrap"><table>
      <tr><th>Letter</th><th>Sound</th><th>Tried</th><th>Wrong</th><th></th><th>Recent (oldest to newest)</th><th>In words</th></tr>
      ${d.sounds.map((r) => `<tr>
        <td class="deva big">${esc(r.letter)}</td><td>${(r.sound || "").split(",").map((x) => `/${esc(x)}/`).join(" ")}</td><td>${r.tries}</td><td>${r.wrong}</td>
        <td><div class="rate"><div style="width:${Math.round(r.rate * 100)}%"></div></div>${Math.round(r.rate * 100)}%</td>
        <td>${recentDots(r.recent)}${r.last_ok ? ` <span class="improving">last time right</span>` : ""}</td>
        <td class="deva">${esc(r.words.join(", "))}</td></tr>`).join("")}
    </table></div>`
    : `<p class="muted">Nothing to practise yet: every letter tracked so far was said right.</p>`;
  const words = d.words.length ? `<div class="words">${d.words.map((w) => `<span class="chip bad">${esc(w.word)}<small>${w.sounds_wrong} wrong · ${esc(w.letters || "")}</small></span>`).join("")}</div>`
    : `<p class="muted">None yet.</p>`;
  const readings = d.attempts.map((a) => `<tr><td>${day(a.created_at)}</td><td>${esc((state.config.tasks[a.task] || {}).title || a.task)}</td>
      <td>${a.passed ? "pass" : "not yet"}</td><td>${a.mistakes}</td><td>${a.correct}/${a.total}</td><td class="deva">${esc(a.transcript)}</td></tr>`).join("");
  const tracked = d.sounds_tracked;
  $("student-detail").innerHTML = `<div class="card">
      <div class="task-head"><div>
        <h1 class="task-title">${esc(d.student.name)}</h1>
        <p class="task-instruction">${d.student.grade ? `Class ${esc(d.student.grade)} · ` : ""}${d.attempts.length} readings · ${tracked.n || 0} sounds tracked, ${tracked.wrong || 0} said wrong</p>
      </div>
      <a class="btn btn-secondary" href="/api/students/${sid}/export.csv">Download CSV</a></div>
      <h3 class="section-title">Letters and sounds to practise</h3>
      <p class="gop-line">Every accepted reading adds one entry per letter the child tried: right or wrong, from the audio check when it ran, otherwise from the transcript. Most often wrong first.</p>
      ${sounds}
      <h3 class="section-title">Words to practise</h3>${words}
      <h3 class="section-title">ASER level over time</h3>${levelChart(d.sessions)}
      <h3 class="section-title">Readings</h3>
      <div class="table-wrap"><table><tr><th>Date</th><th>Task</th><th>Verdict</th><th>Mistakes</th><th>Read correctly</th><th>Heard</th></tr>${readings}</table></div>
      <div class="actions"><button class="btn btn-primary" id="assess-student" type="button">Assess ${esc(d.student.name)} now</button></div>
    </div>`;
  $("assess-student").addEventListener("click", () => { $("student").value = String(sid); resetSession(); showTab("assessment"); });
  $("student-detail").scrollIntoView({ behavior: "smooth" });
}

// -- quick check ---------------------------------------------------------------------------

function setupQuick() {
  $("quick-input").innerHTML = `<div class="recorder"></div>
    <div class="actions"><button class="btn btn-primary" id="quick-btn" type="button">Score</button></div>`;
  quickRecorder = new Recorder($("quick-input").querySelector(".recorder"), { typedPlaceholder: "What was said" });
  $("quick-btn").addEventListener("click", async () => {
    $("quick-save").innerHTML = "";
    const level = $("quick-level").value, text = $("quick-text").value;
    const result = await score(level, text, quickRecorder, $("quick-result"));
    if (!result || !studentId()) return;
    $("quick-save").innerHTML = `<div class="actions"><button class="btn btn-secondary" id="quick-save-btn" type="button">Save to ${esc(studentName())}'s record</button><span class="saved" id="quick-saved"></span></div>`;
    $("quick-save-btn").addEventListener("click", async () => {
      $("quick-save-btn").disabled = true;
      try { $("quick-saved").textContent = await saveReading(level, text, result); }
      catch (err) { $("quick-saved").textContent = `Not saved: ${err.message}`; }
    });
  });
}

// -- wiring ---------------------------------------------------------------------------------

function showTab(name) {
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.tab === name));
  $("tab-assessment").hidden = name !== "assessment";
  $("tab-quick").hidden = name !== "quick";
  $("tab-students").hidden = name !== "students";
  if (name === "students") renderRoster();
}
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showTab(tab.dataset.tab)));

$("engine").addEventListener("change", onEngineChange);
$("gop-on").addEventListener("change", () => { $("gop-fields").hidden = !$("gop-on").checked; });
$("gop").addEventListener("input", () => { $("gop-value").textContent = Number($("gop").value).toFixed(1); });
$("new-child").addEventListener("click", resetSession);
$("student").addEventListener("change", resetSession);
$("add-student").addEventListener("click", addStudent);
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

function startupProblem(message) {
  const box = document.createElement("div");
  box.className = "error startup-error";
  box.innerHTML = message;
  document.querySelector(".main").prepend(box);
}

const RESTART = "Restart the server: in its PowerShell window press Ctrl+C, then run <b>py -m readnet.web</b> from the stt folder, and reload this page.";

(async function init() {
  // The server reads these page files fresh on every request but loads its own
  // code only at start, so a server started before an update serves the new page
  // with an old API. Never let that leave a blank page.
  try {
    await loadConfig("hi");
  } catch (err) {
    startupProblem(`The page could not load its settings (${esc(err.message)}). ${RESTART}`);
    return;
  }
  try {
    await loadStudents();
  } catch (err) {
    $("student").innerHTML = `<option value="">Student records unavailable</option>`;
    startupProblem(`Student records are unavailable (${esc(err.message)}): the server is probably older than this page. ${RESTART}`);
  }
  setupQuick();
  showTask();
  onEngineChange();
})();
