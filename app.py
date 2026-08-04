"""
DeskLens Bench — prompt and model test bench.
Audio -> Sarvam Saaras v3 -> extraction models -> side-by-side comparison.
"""

import json
import os
import re
import tempfile
import time

import pandas as pd
import requests
import streamlit as st

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

SARVAM_BASE = "https://api.sarvam.ai"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"

MODELS = {
    "Gemini 2.5 Flash-Lite": {"vendor": "google", "id": "gemini-2.5-flash-lite"},
    "Gemini 2.5 Flash": {"vendor": "google", "id": "gemini-2.5-flash"},
    "Claude Haiku 4.5": {"vendor": "anthropic", "id": "claude-haiku-4-5-20251001"},
    "Claude Sonnet 5": {"vendor": "anthropic", "id": "claude-sonnet-5"},
}

# Editable in the sidebar. Rupees per 1,000,000 tokens.
DEFAULT_PRICING = {
    "gemini-2.5-flash-lite": (9.0, 36.0),
    "gemini-2.5-flash": (27.0, 216.0),
    "claude-haiku-4-5-20251001": (90.0, 450.0),
    "claude-sonnet-5": (270.0, 1350.0),
}

RUNS_FILE = "runs.jsonl"

st.set_page_config(page_title="DeskLens Bench", layout="wide")

# ----------------------------------------------------------------------------
# State
# ----------------------------------------------------------------------------

defaults = {
    "transcript": "",
    "stt_language": "",
    "stt_raw": None,
    "diarized": None,
    "results": {},
    "audio_seconds": 0.0,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def parse_json_loose(text):
    """Models sometimes wrap JSON in fences or prose. Pull out the object."""
    if not text:
        return None, "Model returned nothing."
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1]), None
        except json.JSONDecodeError as e:
            return None, f"Found braces but could not parse: {e}"
    return None, "No JSON object in the response."


def flatten(obj, prefix=""):
    """Turn nested JSON into flat field paths so two outputs can be compared."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(flatten(v, f"{prefix}.{k}" if prefix else k))
    elif isinstance(obj, list):
        if all(not isinstance(i, (dict, list)) for i in obj):
            out[prefix] = json.dumps(obj, ensure_ascii=False)
        else:
            for i, v in enumerate(obj):
                out.update(flatten(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out


def apply_pipeline_rules(data, stt_language):
    """The post-processing your production code does. Kept separate from the model."""
    if not isinstance(data, dict):
        return data, []
    d = json.loads(json.dumps(data))
    changes = []

    if "language" in d and stt_language:
        if d["language"] != stt_language:
            changes.append(f"language: `{d['language']}` -> `{stt_language}` (from STT metadata)")
        d["language"] = stt_language

    kp = d.get("key_phrases")
    if isinstance(kp, list):
        seen, deduped = set(), []
        for p in kp:
            key = p.strip().lower() if isinstance(p, str) else json.dumps(p, sort_keys=True)
            if key not in seen:
                seen.add(key)
                deduped.append(p)
        if len(deduped) != len(kp):
            changes.append(f"key_phrases: removed {len(kp) - len(deduped)} duplicate(s)")
        d["key_phrases"] = deduped

    return d, changes


def rupees(model_id, tokens_in, tokens_out, pricing):
    p_in, p_out = pricing.get(model_id, (0.0, 0.0))
    return (tokens_in / 1_000_000) * p_in + (tokens_out / 1_000_000) * p_out


# ----------------------------------------------------------------------------
# API calls
# ----------------------------------------------------------------------------

def sarvam_rest(audio_bytes, filename, key, mode, language_code):
    """Synchronous. Files under 30 seconds only."""
    r = requests.post(
        f"{SARVAM_BASE}/speech-to-text",
        headers={"api-subscription-key": key},
        files={"file": (filename, audio_bytes)},
        data={"model": "saaras:v3", "mode": mode, "language_code": language_code},
        timeout=180,
    )
    r.raise_for_status()
    return r.json()


def sarvam_batch(files, key, mode, language_code, diarize, num_speakers, progress):
    """Asynchronous job. Files up to 2 hours, and the only path that returns diarization."""
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=key)
    kwargs = {"model": "saaras:v3", "mode": mode, "language_code": language_code}
    if diarize:
        kwargs["with_diarization"] = True
        kwargs["num_speakers"] = num_speakers

    progress("Creating job")
    job = client.speech_to_text_job.create_job(**kwargs)

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for f in files:
            p = os.path.join(tmp, f.name)
            with open(p, "wb") as fh:
                fh.write(f.getvalue())
            paths.append(p)

        progress(f"Uploading {len(paths)} file(s)")
        job.upload_files(file_paths=paths)
        job.start()

        progress("Sarvam is transcribing. This takes a few minutes for a long call.")
        job.wait_until_complete()

        results = job.get_file_results()
        out_dir = os.path.join(tmp, "out")
        os.makedirs(out_dir, exist_ok=True)
        if results.get("successful"):
            job.download_outputs(output_dir=out_dir)

        payloads = []
        for name in sorted(os.listdir(out_dir)):
            if name.endswith(".json"):
                with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
                    payloads.append(json.load(fh))

        failures = [f"{f['file_name']}: {f.get('error_message')}" for f in results.get("failed", [])]
        return payloads, failures


def call_gemini(model_id, key, prompt, transcript, temperature, force_json):
    cfg = {"temperature": temperature, "thinkingConfig": {"thinkingBudget": 0}}
    if force_json:
        cfg["responseMimeType"] = "application/json"

    body = {
        "contents": [{"role": "user", "parts": [{"text": f"{prompt}\n\nTRANSCRIPT:\n{transcript}"}]}],
        "generationConfig": cfg,
    }
    t0 = time.time()
    r = requests.post(
        f"{GEMINI_BASE}/{model_id}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json=body,
        timeout=300,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    usage = data.get("usageMetadata", {})
    return text, elapsed, usage.get("promptTokenCount", 0), usage.get("candidatesTokenCount", 0)


def call_anthropic(model_id, key, prompt, transcript, temperature):
    t0 = time.time()
    r = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        json={
            "model": model_id,
            "max_tokens": 4000,
            "temperature": min(temperature, 1.0),
            "system": prompt,
            "messages": [{"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}],
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    usage = data.get("usage", {})
    return text, elapsed, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------

with st.sidebar:
    st.header("Keys")
    st.caption("Stored only for this browser session. Nothing is written to disk.")
    sarvam_key = st.text_input("Sarvam", type="password", value=os.getenv("SARVAM_API_KEY", ""))
    gemini_key = st.text_input("Google AI Studio", type="password", value=os.getenv("GEMINI_API_KEY", ""))
    anthropic_key = st.text_input("Anthropic", type="password", value=os.getenv("ANTHROPIC_API_KEY", ""))

    st.divider()
    st.header("Transcription")
    stt_mode = st.selectbox(
        "Mode", ["translate", "transcribe", "codemix", "translit", "verbatim"],
        help="translate gives English. codemix keeps Hinglish as spoken.",
    )
    language_code = st.text_input("Language code", value="unknown",
                                  help="'unknown' lets Sarvam detect it.")
    use_batch = st.toggle("Batch API", value=True,
                          help="On: files up to 2 hours, diarization available. Off: fast, but under 30 seconds only.")
    diarize = st.toggle("Speaker diarization", value=True, disabled=not use_batch)
    num_speakers = st.number_input("Speakers", 2, 20, 2, disabled=not (use_batch and diarize))

    st.divider()
    st.header("Extraction")
    temperature = st.slider("Temperature", 0.0, 2.0, 0.0, 0.1,
                            help="Keep at 0 for strict extraction. Anthropic caps at 1.")
    force_json = st.toggle("Force JSON output (Gemini)", value=True)
    clean_output = st.toggle("Show pipeline-cleaned output", value=True,
                             help="Applies your post-processing rules on top of the raw model output.")

    with st.expander("Pricing (₹ per 1M tokens)"):
        st.caption("Estimates. Edit to match your current rates.")
        pricing = {}
        for label, m in MODELS.items():
            c1, c2 = st.columns(2)
            d_in, d_out = DEFAULT_PRICING[m["id"]]
            pricing[m["id"]] = (
                c1.number_input(f"{label} in", value=d_in, key=f"pi_{m['id']}"),
                c2.number_input(f"{label} out", value=d_out, key=f"po_{m['id']}"),
            )

# ----------------------------------------------------------------------------
# Step 1 — audio
# ----------------------------------------------------------------------------

st.title("DeskLens Bench")
st.caption("Audio → Sarvam Saaras v3 → extraction → compare. Every stage shows its raw output.")

st.subheader("1 · Audio")
tab_audio, tab_paste = st.tabs(["Upload recordings", "Paste a transcript"])

with tab_audio:
    uploads = st.file_uploader(
        "Drop call recordings here",
        type=["mp3", "wav", "m4a", "aac", "ogg", "opus", "flac", "amr", "webm"],
        accept_multiple_files=True,
        help="Up to 20 files per batch job. Download them from Drive first.",
    )
    if uploads:
        st.write(f"{len(uploads)} file(s) ready.")
        if st.button("Transcribe with Sarvam", type="primary"):
            if not sarvam_key:
                st.error("Add your Sarvam key in the sidebar first.")
            else:
                status = st.status("Starting", expanded=True)
                try:
                    if use_batch:
                        payloads, failures = sarvam_batch(
                            uploads, sarvam_key, stt_mode, language_code,
                            diarize, num_speakers, lambda m: status.write(m),
                        )
                        for f in failures:
                            st.warning(f"Sarvam could not process {f}")
                        if not payloads:
                            status.update(label="No transcripts came back", state="error")
                        else:
                            first = payloads[0]
                            st.session_state.stt_raw = payloads
                            st.session_state.transcript = first.get("transcript", "")
                            st.session_state.stt_language = first.get("language_code", "")
                            st.session_state.diarized = first.get("diarized_transcript")
                            status.update(label=f"{len(payloads)} transcript(s) ready", state="complete")
                    else:
                        f = uploads[0]
                        status.write("Sending to the REST endpoint")
                        payload = sarvam_rest(f.getvalue(), f.name, sarvam_key, stt_mode, language_code)
                        st.session_state.stt_raw = [payload]
                        st.session_state.transcript = payload.get("transcript", "")
                        st.session_state.stt_language = payload.get("language_code", "")
                        st.session_state.diarized = payload.get("diarized_transcript")
                        status.update(label="Transcript ready", state="complete")
                except requests.HTTPError as e:
                    status.update(label="Sarvam rejected the request", state="error")
                    st.error(f"{e.response.status_code}: {e.response.text[:600]}")
                except Exception as e:
                    status.update(label="Transcription failed", state="error")
                    st.error(str(e))

with tab_paste:
    pasted = st.text_area("Transcript", height=180, placeholder="Paste a transcript to skip Sarvam.")
    lang_manual = st.text_input("STT language code", value="", placeholder="en-IN")
    if st.button("Use this transcript"):
        st.session_state.transcript = pasted
        st.session_state.stt_language = lang_manual
        st.session_state.stt_raw = None
        st.session_state.diarized = None
        st.success("Loaded.")

# ----------------------------------------------------------------------------
# Step 2 — transcript
# ----------------------------------------------------------------------------

if st.session_state.transcript:
    st.subheader("2 · Transcript")

    c1, c2, c3 = st.columns(3)
    c1.metric("Detected language", st.session_state.stt_language or "—")
    c2.metric("Characters", len(st.session_state.transcript))
    c3.metric("Speakers found", len({e.get("speaker_id") for e in
                                     (st.session_state.diarized or {}).get("entries", [])}) or "—")

    if st.session_state.stt_language:
        st.info(f"`language` should be **{st.session_state.stt_language}**, taken from this metadata — "
                "never inferred from the transcript text.")

    edited = st.text_area("Transcript (editable)", st.session_state.transcript, height=220)
    if edited != st.session_state.transcript:
        st.session_state.transcript = edited

    if st.session_state.diarized:
        with st.expander("Diarized turns"):
            rows = [
                {
                    "speaker": e.get("speaker_id"),
                    "start": round(e.get("start_time_seconds", 0), 2),
                    "end": round(e.get("end_time_seconds", 0), 2),
                    "text": e.get("transcript", ""),
                }
                for e in st.session_state.diarized.get("entries", [])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    if st.session_state.stt_raw:
        with st.expander("Raw Sarvam response"):
            st.json(st.session_state.stt_raw)

# ----------------------------------------------------------------------------
# Step 3 — prompt
# ----------------------------------------------------------------------------

st.subheader("3 · Prompt")

prompt_file = st.file_uploader("Load a saved prompt", type=["txt", "md"], key="prompt_upload")
if prompt_file:
    st.session_state.prompt_text = prompt_file.getvalue().decode("utf-8")

pc1, pc2 = st.columns([3, 1])
prompt_text = pc1.text_area(
    "Extraction prompt",
    value=st.session_state.get("prompt_text", ""),
    height=280,
    placeholder="Paste your derma_v6 prompt here.",
)
prompt_version = pc2.text_input("Version label", value="derma_v6")
pc2.download_button("Save prompt", prompt_text or " ", file_name=f"{prompt_version}.txt")

# ----------------------------------------------------------------------------
# Step 4 — run
# ----------------------------------------------------------------------------

st.subheader("4 · Run models")

chosen = st.multiselect("Models", list(MODELS.keys()),
                        default=["Gemini 2.5 Flash-Lite", "Claude Haiku 4.5"])

run_col, clear_col = st.columns([1, 4])
run_now = run_col.button("Run selected", type="primary",
                         disabled=not (st.session_state.transcript and prompt_text and chosen))
if clear_col.button("Clear results"):
    st.session_state.results = {}

if run_now:
    for label in chosen:
        m = MODELS[label]
        key = gemini_key if m["vendor"] == "google" else anthropic_key
        if not key:
            st.error(f"{label} needs its API key in the sidebar.")
            continue

        with st.spinner(f"{label} running"):
            try:
                if m["vendor"] == "google":
                    text, elapsed, t_in, t_out = call_gemini(
                        m["id"], key, prompt_text, st.session_state.transcript, temperature, force_json)
                else:
                    text, elapsed, t_in, t_out = call_anthropic(
                        m["id"], key, prompt_text, st.session_state.transcript, temperature)

                parsed, err = parse_json_loose(text)
                cleaned, changes = apply_pipeline_rules(parsed, st.session_state.stt_language) if parsed else (None, [])

                st.session_state.results[label] = {
                    "model_id": m["id"], "raw_text": text, "parsed": parsed, "cleaned": cleaned,
                    "changes": changes, "error": err, "seconds": elapsed,
                    "tokens_in": t_in, "tokens_out": t_out,
                    "cost": rupees(m["id"], t_in, t_out, pricing),
                }

                with open(RUNS_FILE, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "prompt_version": prompt_version, "model": label,
                        "temperature": temperature, "stt_language": st.session_state.stt_language,
                        "seconds": round(elapsed, 2), "tokens_in": t_in, "tokens_out": t_out,
                        "output": parsed,
                    }, ensure_ascii=False) + "\n")

            except requests.HTTPError as e:
                st.error(f"{label} — {e.response.status_code}: {e.response.text[:400]}")
            except Exception as e:
                st.error(f"{label} — {e}")

# ----------------------------------------------------------------------------
# Step 5 — compare
# ----------------------------------------------------------------------------

if st.session_state.results:
    st.subheader("5 · Compare")

    summary = pd.DataFrame([
        {
            "Model": k, "JSON parsed": "yes" if v["parsed"] else "no",
            "Seconds": round(v["seconds"], 2), "Tokens in": v["tokens_in"],
            "Tokens out": v["tokens_out"], "₹ per call": round(v["cost"], 4),
            "₹ per 1000 calls": round(v["cost"] * 1000, 2),
        }
        for k, v in st.session_state.results.items()
    ])
    st.dataframe(summary, use_container_width=True, hide_index=True)

    labels = list(st.session_state.results.keys())
    cols = st.columns(len(labels))
    for col, label in zip(cols, labels):
        r = st.session_state.results[label]
        with col:
            st.markdown(f"**{label}**")
            if r["error"]:
                st.error(r["error"])
                st.code(r["raw_text"][:1500])
            else:
                show = r["cleaned"] if (clean_output and r["cleaned"]) else r["parsed"]
                st.json(show, expanded=True)
                if clean_output and r["changes"]:
                    st.caption("Pipeline changed: " + "; ".join(r["changes"]))

    parsed_ok = {k: v for k, v in st.session_state.results.items() if v["parsed"]}
    if len(parsed_ok) >= 2:
        st.markdown("**Field-by-field**")
        flats = {k: flatten(v["cleaned"] if (clean_output and v["cleaned"]) else v["parsed"])
                 for k, v in parsed_ok.items()}
        fields = sorted({f for d in flats.values() for f in d})

        rows = []
        for f in fields:
            vals = {k: d.get(f, "—") for k, d in flats.items()}
            agree = len({json.dumps(v, ensure_ascii=False, sort_keys=True) for v in vals.values()}) == 1
            rows.append({"Field": f, "Agree": "" if agree else "differs", **vals})

        diff_only = st.checkbox("Show differences only", value=True)
        table = pd.DataFrame(rows)
        if diff_only:
            table = table[table["Agree"] == "differs"]
        if table.empty:
            st.success("Every field matches across models.")
        else:
            st.dataframe(table, use_container_width=True, hide_index=True)

    st.download_button(
        "Download this comparison",
        json.dumps({
            "prompt_version": prompt_version, "temperature": temperature,
            "stt_language": st.session_state.stt_language,
            "transcript": st.session_state.transcript,
            "results": {k: {kk: vv for kk, vv in v.items() if kk != "raw_text"}
                        for k, v in st.session_state.results.items()},
        }, ensure_ascii=False, indent=2),
        file_name=f"bench_{prompt_version}_{time.strftime('%Y%m%d_%H%M')}.json",
    )

# ----------------------------------------------------------------------------
# History
# ----------------------------------------------------------------------------

if os.path.exists(RUNS_FILE):
    with st.expander("Run history"):
        with open(RUNS_FILE, encoding="utf-8") as fh:
            history = [json.loads(line) for line in fh if line.strip()]
        st.dataframe(
            pd.DataFrame([{k: v for k, v in h.items() if k != "output"} for h in history]),
            use_container_width=True, hide_index=True,
        )
        st.download_button("Download history", open(RUNS_FILE, encoding="utf-8").read(),
                           file_name="desklens_runs.jsonl")
