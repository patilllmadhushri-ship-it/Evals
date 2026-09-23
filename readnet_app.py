"""ReadNet test bench — run a child through the ASER reading test.

    py -m streamlit run readnet_app.py

The adult picks a language and a speech engine, the child reads each task
aloud, and the app transcribes, marks every word, and walks the ASER order
(paragraph → story, or words → letters) to a reading level. Every verdict can
be disputed; disputes are saved as candidate field cases for the rulebook.
"""

from __future__ import annotations

import csv
import html
import importlib.util
import json
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from readnet import languages
from readnet.aser import Level, Rules, next_task, place
from readnet.pipeline import assess_task
from stt_eval import audio as stt_audio
from stt_eval import env, providers
from stt_eval.providers import ProviderError

st.set_page_config(page_title="ReadNet · Reading Level", page_icon="📖", layout="wide")
env.load()

DISPUTE_LOG = Path(".stt_eval_runs") / "readnet_disputes.csv"  # git-ignored: holds children's readings
UPLOAD_TYPES = ["wav", "mp3", "m4a", "ogg", "webm", "flac"]
LOCALES = {"hi": "hi-IN", "mr": "mr-IN"}
DEFAULT_LOCAL_MODEL = {"hi": "ai4bharat/indicwav2vec-hindi", "mr": ""}

# Simple texts written for this test bench, graded like the ASER tool:
# paragraph ≈ Grade 1, story ≈ Grade 2, five common words, five letters.
CONTENT = {
    "hi": {
        Level.PARAGRAPH: "मेरा नाम राजू है। मेरे घर में एक गाय है। गाय का रंग सफ़ेद है। वह हरी घास खाती है।",
        Level.STORY: (
            "रीना के पास एक छोटा कुत्ता था। उसका नाम मोती था। एक दिन मोती घर से बाहर चला गया। "
            "रीना बहुत डर गई। उसने पूरे गाँव में मोती को ढूँढा। शाम को मोती पेड़ के नीचे सोता मिला। "
            "रीना ने उसे गले से लगा लिया। अब वह मोती का पूरा ध्यान रखती है।"
        ),
        Level.WORD: "घर पानी कमल नाक बस",
        Level.LETTER: "क म स ल र",
    },
    "mr": {
        Level.PARAGRAPH: "माझे नाव सीमा आहे. माझ्या घरी एक मांजर आहे. मांजर पांढरी आहे. ती दूध पिते.",
        Level.STORY: (
            "राजूकडे एक लाल सायकल होती. तो रोज सायकलने शाळेत जायचा. एक दिवस सायकलचे चाक पंक्चर झाले. "
            "राजू खूप नाराज झाला. त्याच्या बाबांनी चाक दुरुस्त केले. राजूने बाबांना धन्यवाद दिले. "
            "आता तो सायकल नीट चालवतो."
        ),
        Level.WORD: "घर पाणी कमळ नाक बस",
        Level.LETTER: "क म स ल र",
    },
}

TASK_COPY = {
    Level.PARAGRAPH: ("Paragraph", "Ask the child to read the whole paragraph aloud.",
                      "Pass with 3 mistakes or fewer → Story next. Otherwise → Words."),
    Level.STORY: ("Story", "Ask the child to read the story aloud.",
                  "Pass with 3 mistakes or fewer → Story level. Otherwise → Paragraph level."),
    Level.WORD: ("Words", "Ask the child to read the five words, one after another, in one recording.",
                 "Pass with 4 of 5 correct → Word level. Otherwise → Letters."),
    Level.LETTER: ("Letters", "Ask the child to say the five letters, one after another, in one recording.",
                   "Pass with 4 of 5 correct → Letter level. Otherwise → Beginner."),
}

NEXT_STEPS = {
    Level.STORY: "Reads a Grade 2 story. Move on to reading for meaning and longer texts.",
    Level.PARAGRAPH: "Reads Grade 1 text but not yet a Grade 2 story. Practise longer texts and fluency.",
    Level.WORD: "Reads words. Practise joining words into sentences.",
    Level.LETTER: "Knows letters. Practise building words — letters with matras, then short words.",
    Level.BEGINNER: "Start with letters and their sounds.",
}

TYPED, LOCAL = "__typed__", "__local__"

st.markdown(
    """
<link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari:wght@400;600&display=swap" rel="stylesheet">
<style>
.rn-text { font-family: 'Noto Sans Devanagari', sans-serif; font-size: 1.9rem; line-height: 2.6rem;
           padding: 1.1rem 1.4rem; border-radius: 12px; border: 1px solid rgba(127,127,127,.35);
           background: rgba(127,127,127,.08); margin: .4rem 0 1rem 0; }
.rn-marked { font-family: 'Noto Sans Devanagari', sans-serif; line-height: 3.6rem; font-size: 1.35rem; }
.rn-w { display: inline-block; padding: .05rem .5rem; margin: .15rem .15rem; border-radius: 8px;
        border: 1px solid transparent; line-height: 1.9rem; vertical-align: top; text-align: center; }
.rn-w small { display: block; font-size: .68rem; line-height: .95rem; opacity: .9; font-family: sans-serif; }
.rn-ok { background: rgba(46,160,67,.18); border-color: rgba(46,160,67,.45); }
.rn-forgiven { background: rgba(210,153,34,.18); border-color: rgba(210,153,34,.55); }
.rn-bad { background: rgba(218,54,51,.20); border-color: rgba(218,54,51,.6); }
.rn-skip { text-decoration: line-through; }
.rn-extra { background: rgba(127,127,127,.12); border: 1px dashed rgba(127,127,127,.5); opacity: .85; }
.rn-steps { display: flex; flex-wrap: wrap; gap: .5rem; margin: .2rem 0 1rem 0; }
.rn-step { padding: .3rem .8rem; border-radius: 999px; border: 1px solid rgba(127,127,127,.4); font-size: .9rem; }
.rn-step.pass { background: rgba(46,160,67,.2); border-color: rgba(46,160,67,.6); }
.rn-step.fail { background: rgba(218,54,51,.18); border-color: rgba(218,54,51,.55); }
.rn-step.now { border: 2px solid rgba(56,139,253,.9); font-weight: 600; }
.rn-level { font-size: 2.6rem; font-weight: 700; margin: 0; }
.rn-legend span { margin-right: .4rem; }
</style>
""",
    unsafe_allow_html=True,
)


# -- engines -------------------------------------------------------------------------


def local_model_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "transformers"))


def engine_options(language: str) -> dict[str, str]:
    locale = LOCALES[language]
    options: dict[str, str] = {}
    # Indic-first engines first: Sarvam is built for Hindi and Marathi.
    preferred = ["sarvam", "google", "deepgram", "elevenlabs", "openai", "mock"]
    available = providers.providers_for_language(locale)
    for key in sorted(available, key=lambda k: preferred.index(k) if k in preferred else len(preferred)):
        if key == "mock":
            options[key] = "Mock — simulates a reading from the screen text (no audio needed)"
        elif env.provider_key(key):
            options[key] = f"{providers.provider_label(key)} (cloud)"
    options[LOCAL] = "Local model — wav2vec2 on this machine (+ word timings & GOP)"
    options[TYPED] = "No ASR — type what the child said"
    return options


@st.cache_resource(show_spinner="Loading the local model (first time downloads it)…")
def load_local_model(model_id: str):
    from readnet.acoustic import Wav2Vec2Emissions

    return Wav2Vec2Emissions(model_id)


def transcribe(engine: str, recording, canonical: str, language: str, model_id: str):
    """Returns (transcript, seconds, acoustic evidence or None)."""
    started = time.perf_counter()
    if engine == "mock":
        mock = providers.build("mock", "")
        mock._ground_truth_hint = canonical
        raw = recording.getvalue() if recording is not None else canonical.encode("utf-8")
        return mock.transcribe(raw, LOCALES[language]).text, time.perf_counter() - started, None

    clip = stt_audio.normalize(recording.getvalue(), getattr(recording, "name", "recording.wav"), sample_rate=16000)
    if engine == LOCAL:
        from readnet.acoustic import evidence_for_words, greedy_decode

        model = load_local_model(model_id)
        em = model.emissions(clip.wav_bytes)
        words = languages.get(language).normalize(canonical)[0].split()
        evidence = evidence_for_words(
            em.log_probs, words, em.vocab, frame_seconds=em.frame_seconds, blank=em.blank,
            word_delimiter=em.word_delimiter,
        )
        return greedy_decode(em), time.perf_counter() - started, evidence

    provider = providers.build(engine, env.provider_key(engine))
    return provider.transcribe(clip.wav_bytes, LOCALES[language]).text, time.perf_counter() - started, None


# -- rendering -----------------------------------------------------------------------


def marked_words_html(item) -> str:
    """Every word of the text, coloured by verdict, with what was heard underneath."""
    e = html.escape
    timings = item.acoustic.words if item.acoustic else None
    chips: list[str] = []
    for op in item.result.ops:
        extra = ""
        if timings and op.ref_index is not None and op.ref_index < len(timings):
            w = timings[op.ref_index]
            if w.llr is not None:
                extra = f" · GOP {w.llr:.1f}"
        if op.kind == "match":
            if op.rule or op.note:
                chips.append(f'<span class="rn-w rn-forgiven">{e(op.ref)}<small>{e(op.rule or "")}'
                             f'{e(" · " + op.note if op.note else "")}{e(extra)}</small></span>')
            else:
                chips.append(f'<span class="rn-w rn-ok">{e(op.ref)}<small>✓{e(extra)}</small></span>')
        elif op.kind == "substitution":
            why = op.rule if not op.counts_as_mistake else "/".join(op.subtypes) or op.category or ""
            css = "rn-bad" if op.counts_as_mistake else "rn-forgiven"
            chips.append(f'<span class="rn-w {css}">{e(op.ref)}<small>heard {e(op.hyp)} · {e(why)}{e(extra)}</small></span>')
        elif op.kind == "deletion":
            chips.append(f'<span class="rn-w rn-bad"><span class="rn-skip">{e(op.ref)}</span><small>not read</small></span>')
        else:
            label = f"{op.category} · {op.rule}" if op.rule else (op.category or "")
            css = "rn-bad" if op.counts_as_mistake else "rn-extra"
            chips.append(f'<span class="rn-w {css}">+{e(op.hyp)}<small>{e(label)}</small></span>')
    return '<div class="rn-marked">' + "".join(chips) + "</div>"


LEGEND = (
    '<div class="rn-legend"><span class="rn-w rn-ok">read right</span>'
    '<span class="rn-w rn-forgiven">difference forgiven (rule id)</span>'
    '<span class="rn-w rn-bad">mistake</span>'
    '<span class="rn-w rn-extra">+ extra sound, not counted</span></div>'
)


def show_result(outcome, item, transcript: str, seconds: float, engine_label: str) -> None:
    verdict = "✅ Pass" if outcome.passed else "❌ Not yet"
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Verdict", verdict)
    c2.metric("Mistakes", outcome.mistakes)
    c3.metric("Read correctly", f"{outcome.correct} / {outcome.total}")
    c4.metric("Engine time", f"{seconds:.1f}s")
    st.caption(" · ".join(outcome.reasons))
    st.markdown(f"**Heard** ({engine_label}): {html.escape(transcript) or '—'}")
    st.markdown(LEGEND, unsafe_allow_html=True)
    st.markdown(marked_words_html(item), unsafe_allow_html=True)
    if outcome.wpm is not None:
        pause = f", longest pause {outcome.longest_pause:.1f}s" if outcome.longest_pause is not None else ""
        st.caption(f"Pace: {outcome.wpm:.0f} words/min{pause} (measured, not enforced)")
    if item.acoustic_doubts:
        st.warning("The audio does not sound like: " + ", ".join(item.acoustic_doubts)
                   + " — the transcript says correct, so these are flagged for review, not counted.")
    with st.expander("Show the working"):
        st.write("**Text after normalisation:**", item.canonical_normalized)
        st.write("**Heard after normalisation:**", item.transcript_normalized)
        if item.transcript_trace.steps:
            st.write("**Rules that changed the heard text:**")
            st.table([{"rule": r, "before": b, "after": a} for r, b, a in item.transcript_trace.steps])
        st.json(item.result.as_dict(), expanded=False)


def dispute_box(level: Level, canonical: str, transcript: str, outcome, key: str) -> None:
    """Assessor disagreements are the raw material for new rules and field cases."""
    with st.expander("The assessor disagrees with this verdict"):
        c1, c2 = st.columns([1, 3])
        human = c1.number_input("Mistakes you counted", min_value=0, max_value=200, value=outcome.mistakes, key=f"h_{key}")
        note = c2.text_input("What did the child actually read? Which word did the app get wrong?", key=f"n_{key}")
        if st.button("Save as a field case", key=f"s_{key}"):
            DISPUTE_LOG.parent.mkdir(exist_ok=True)
            new = not DISPUTE_LOG.exists()
            with DISPUTE_LOG.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                if new:
                    writer.writerow(["saved_at", "language", "level", "canonical", "transcript",
                                     "system_mistakes", "human_mistakes", "note"])
                writer.writerow([datetime.now().isoformat(timespec="seconds"), st.session_state.language,
                                 level.name.lower(), canonical, transcript, outcome.mistakes, human, note])
            st.success(f"Saved to {DISPUTE_LOG}. Review these with a language expert, then add them to "
                       "readnet/tests/field_cases.csv with the rule each one tests.")


def audio_input(key: str, engine: str):
    if engine == TYPED:
        return None
    rec_tab, up_tab = st.tabs(["🎙 Record", "📁 Upload"])
    with rec_tab:
        recording = st.audio_input("Record the child reading", key=f"mic_{key}")
    with up_tab:
        uploaded = st.file_uploader("…or upload a recording", type=UPLOAD_TYPES, key=f"up_{key}")
    return uploaded or recording


def run_scoring(level: Level, canonical: str, engine: str, recording, typed: str, language: str,
                model_id: str, gop_threshold: float | None):
    if engine == TYPED:
        transcript, seconds, evidence = typed, 0.0, None
    else:
        if recording is None and engine != "mock":
            st.warning("Record or upload the child's reading first.")
            return None
        with st.spinner("Listening…"):
            transcript, seconds, evidence = transcribe(engine, recording, canonical, language, model_id)
    outcome, items = assess_task(
        level, [(canonical, transcript)], language=language, acoustic=[evidence] if evidence else None,
        gop_threshold=gop_threshold if evidence else None,
    )
    return {"outcome": outcome, "item": items[0], "transcript": transcript, "seconds": seconds,
            "canonical": canonical}


# -- state -------------------------------------------------------------------------------


def accept(level: Level) -> None:
    pending = st.session_state.pending
    st.session_state.outcomes[level] = pending["outcome"]
    st.session_state.results[level] = pending
    st.session_state.pending = None


def record_again() -> None:
    st.session_state.pending = None
    st.session_state.round += 1


def reset_session() -> None:
    st.session_state.outcomes = {}
    st.session_state.results = {}
    st.session_state.pending = None
    st.session_state.round = st.session_state.get("round", 0) + 1


for name, default in (("outcomes", {}), ("results", {}), ("pending", None), ("round", 0), ("quick", None)):
    st.session_state.setdefault(name, default)


# -- sidebar -----------------------------------------------------------------------------

with st.sidebar:
    st.header("📖 ReadNet")
    language = st.radio("Language", ["hi", "mr"], format_func=lambda c: languages.get(c).name, horizontal=True,
                        key="language", on_change=reset_session)
    options = engine_options(language)
    engine = st.selectbox("Speech engine (ASR)", list(options), format_func=options.get, key="engine")
    model_id, gop_threshold = "", None
    if engine == LOCAL:
        model_id = st.text_input("Hugging Face CTC model", value=DEFAULT_LOCAL_MODEL[language],
                                 help="Any wav2vec2-style CTC checkpoint for this language.")
        if not local_model_available():
            st.error("Needs `torch` and `transformers`:\n\n`py -m pip install torch transformers`\n\n"
                     "Then restart this app.")
        gop_threshold = st.slider("Flag words with GOP below", -10.0, 2.0, -3.0, 0.5,
                                  help="Uncalibrated: flagged words are shown for review, never counted.")
    elif engine == "mock":
        st.caption("Mock drops words at random from the screen text — for trying the flow without a microphone.")
    if not any(k for k in options if k not in (LOCAL, TYPED, "mock")):
        st.warning("No cloud ASR key found in .env (SARVAM_API_KEY, GOOGLE_API_KEY, DEEPGRAM_API_KEY…).")
    st.divider()
    child = st.text_input("Child (optional, not saved)", key=f"child_{st.session_state.round}")
    st.button("🔄 New child", width="stretch", on_click=reset_session)
    st.caption("Rulebooks: readnet/languages/hi/RULES.md, mr/RULES.md, readnet/RULES_SHARED.md. "
               "Audio is never stored.")

engine_label = options[engine].split(" (")[0].split(" —")[0]

session_tab, quick_tab = st.tabs(["🧒 ASER test", "🔍 Quick check"])

# -- the ASER session ---------------------------------------------------------------------

with session_tab:
    outcomes = st.session_state.outcomes
    current = next_task(outcomes)

    # Where the child is on the path.
    path = [Level.PARAGRAPH]
    if Level.PARAGRAPH in outcomes:
        path += [Level.STORY] if outcomes[Level.PARAGRAPH].passed else [Level.WORD]
    if Level.WORD in outcomes and not outcomes[Level.WORD].passed:
        path += [Level.LETTER]
    steps = []
    for lvl in path:
        o = outcomes.get(lvl)
        css = "now" if lvl == current else ("pass" if o and o.passed else "fail" if o else "")
        mark = "●" if lvl == current else ("✓" if o and o.passed else "✗" if o else "○")
        steps.append(f'<span class="rn-step {css}">{mark} {TASK_COPY[lvl][0]}</span>')
    if current is not None:
        steps.append('<span class="rn-step">… → level</span>')
    st.markdown('<div class="rn-steps">' + "".join(steps) + "</div>", unsafe_allow_html=True)

    if current is not None:
        title, instruction, rule = TASK_COPY[current]
        st.subheader(f"{title}{' — ' + child if child else ''}")
        st.write(instruction)
        text_key = f"text_{language}_{current.name}"
        st.session_state.setdefault(text_key, CONTENT[language][current])
        canonical = st.session_state[text_key]
        st.markdown(f'<div class="rn-text">{html.escape(canonical)}</div>', unsafe_allow_html=True)
        st.caption(rule)
        with st.expander("Use a different text"):
            st.text_area("Text shown to the child", key=text_key)

        key = f"{st.session_state.round}_{current.name}"
        recording = audio_input(key, engine)
        typed = ""
        if engine == TYPED:
            typed = st.text_area("What the child said", key=f"typed_{key}",
                                 placeholder="Type exactly what you heard, including repeats and restarts")

        if st.button("Score this reading", type="primary", key=f"go_{key}"):
            try:
                st.session_state.pending = run_scoring(current, canonical, engine, recording, typed, language,
                                                       model_id, gop_threshold)
            except (ProviderError, stt_audio.AudioError, ImportError, OSError, ValueError) as error:
                st.error(f"Could not transcribe: {error}")

        pending = st.session_state.pending
        if pending and pending["canonical"] == canonical and pending["outcome"].level == current:
            st.divider()
            show_result(pending["outcome"], pending["item"], pending["transcript"], pending["seconds"], engine_label)
            dispute_box(current, canonical, pending["transcript"], pending["outcome"], key)
            a, b, _ = st.columns([1.3, 1, 3])
            a.button("Accept and continue →", type="primary", key=f"ok_{key}", on_click=accept, args=(current,))
            b.button("Record again", key=f"again_{key}", on_click=record_again)
    else:
        placement = place(outcomes)
        st.markdown(f"<p>{html.escape(child) + ' reads at' if child else 'Reading level'}</p>"
                    f'<p class="rn-level">{placement.level.label}</p>', unsafe_allow_html=True)
        st.info(NEXT_STEPS[placement.level])
        st.markdown("**How the app got there**")
        for line in placement.path:
            st.markdown(f"- {line}")

        profile: Counter = Counter()
        rows = []
        for lvl, res in st.session_state.results.items():
            profile.update(res["item"].result.mistake_profile)
            rows.append({"Task": TASK_COPY[lvl][0], "Verdict": "pass" if res["outcome"].passed else "not yet",
                         "Mistakes": res["outcome"].mistakes,
                         "Read correctly": f"{res['outcome'].correct}/{res['outcome'].total}",
                         "Heard": res["transcript"]})
        st.markdown("**Tasks**")
        st.dataframe(rows, hide_index=True, width="stretch")
        sounds = {k.split(":", 1)[1]: v for k, v in profile.items() if k.startswith("sound:")}
        if sounds:
            st.markdown("**Mistake profile** — the kinds of sounds this child gets wrong")
            st.bar_chart(pd.DataFrame({"mistakes": sounds}), horizontal=True)
        for lvl, res in st.session_state.results.items():
            with st.expander(f"{TASK_COPY[lvl][0]} — word by word"):
                st.markdown(marked_words_html(res["item"]), unsafe_allow_html=True)

        report = {
            "language": language, "level": placement.level.label, "path": placement.path,
            "mistake_profile": dict(profile),
            "tasks": {TASK_COPY[l][0]: {"transcript": r["transcript"], **r["item"].result.as_dict()}
                      for l, r in st.session_state.results.items()},
        }
        c1, c2 = st.columns([1, 4])
        c1.download_button("Download result", json.dumps(report, ensure_ascii=False, indent=2, default=str),
                           file_name="readnet_result.json", mime="application/json")
        c2.button("Test another child", on_click=reset_session)

# -- quick check -------------------------------------------------------------------------

with quick_tab:
    st.write("Score one reading against any text — for trying a rule or a tricky word.")
    c1, c2 = st.columns([1, 3])
    q_level = c1.selectbox("Task type", [Level.LETTER, Level.WORD, Level.PARAGRAPH, Level.STORY],
                           index=2, format_func=lambda l: TASK_COPY[l][0])
    q_text = c2.text_input("Text on the screen", value="चाँद और गाँव" if language == "hi" else "माझे घर",
                           key=f"q_text_{language}")
    q_key = f"q_{st.session_state.round}"
    q_recording = audio_input(q_key, engine)
    q_typed = ""
    if engine == TYPED:
        q_typed = st.text_input("What was said", value="चाद और गाव" if language == "hi" else "",
                                key=f"q_typed_{language}")
    if st.button("Score", type="primary", key="q_go"):
        try:
            st.session_state.quick = run_scoring(q_level, q_text, engine, q_recording, q_typed, language,
                                                 model_id, gop_threshold)
        except (ProviderError, stt_audio.AudioError, ImportError, OSError, ValueError) as error:
            st.error(f"Could not transcribe: {error}")
    quick = st.session_state.quick
    if quick:
        show_result(quick["outcome"], quick["item"], quick["transcript"], quick["seconds"], engine_label)
