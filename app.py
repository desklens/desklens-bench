"""
DeskLens Bench (Gradio) — prompt and model test bench.
Audio -> Sarvam Saaras v3 -> extraction models -> side-by-side comparison.
"""

import json
import os
import re
import time

import gradio as gr
import pandas as pd
import requests

SARVAM_BASE = "https://api.sarvam.ai"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
RUNS_FILE = "runs.jsonl"

MODELS = {
    "Gemini 2.5 Flash-Lite": ("google", "gemini-2.5-flash-lite", 9.0, 36.0),
    "Gemini 2.5 Flash": ("google", "gemini-2.5-flash", 27.0, 216.0),
    "Claude Haiku 4.5": ("anthropic", "claude-haiku-4-5-20251001", 90.0, 450.0),
    "Claude Sonnet 5": ("anthropic", "claude-sonnet-5", 270.0, 1350.0),
}
MODEL_NAMES = list(MODELS.keys())
MAX_PANELS = 4


# ---------------------------------------------------------------- helpers ---

def parse_json_loose(text):
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
    if not isinstance(data, dict):
        return data, []
    d = json.loads(json.dumps(data))
    changes = []

    if "language" in d and stt_language:
        if d["language"] != stt_language:
            changes.append(f"language: {d['language']} -> {stt_language} (from STT metadata)")
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


def rupees(model_name, t_in, t_out, price_overrides):
    _, mid, d_in, d_out = MODELS[model_name]
    p_in, p_out = price_overrides.get(mid, (d_in, d_out))
    return (t_in / 1_000_000) * p_in + (t_out / 1_000_000) * p_out


# ------------------------------------------------------------- API calls ---

def sarvam_rest(path, key, mode, language_code):
    with open(path, "rb") as fh:
        r = requests.post(
            f"{SARVAM_BASE}/speech-to-text",
            headers={"api-subscription-key": key},
            files={"file": (os.path.basename(path), fh)},
            data={"model": "saaras:v3", "mode": mode, "language_code": language_code},
            timeout=180,
        )
    r.raise_for_status()
    return r.json()


def sarvam_batch(paths, key, mode, language_code, diarize, num_speakers):
    from sarvamai import SarvamAI

    client = SarvamAI(api_subscription_key=key)
    kwargs = {"model": "saaras:v3", "mode": mode, "language_code": language_code}
    if diarize:
        kwargs["with_diarization"] = True
        kwargs["num_speakers"] = int(num_speakers)

    job = client.speech_to_text_job.create_job(**kwargs)
    job.upload_files(file_paths=paths)
    job.start()
    job.wait_until_complete()

    results = job.get_file_results()
    out_dir = os.path.join(os.getcwd(), f"stt_out_{int(time.time())}")
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
    cfg = {"temperature": float(temperature), "thinkingConfig": {"thinkingBudget": 0}}
    if force_json:
        cfg["responseMimeType"] = "application/json"

    t0 = time.time()
    r = requests.post(
        f"{GEMINI_BASE}/{model_id}:generateContent",
        headers={"x-goog-api-key": key, "Content-Type": "application/json"},
        json={
            "contents": [{"role": "user",
                          "parts": [{"text": f"{prompt}\n\nTRANSCRIPT:\n{transcript}"}]}],
            "generationConfig": cfg,
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    parts = data.get("candidates", [{}])[0].get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    u = data.get("usageMetadata", {})
    return text, elapsed, u.get("promptTokenCount", 0), u.get("candidatesTokenCount", 0)


def call_anthropic(model_id, key, prompt, transcript, temperature):
    t0 = time.time()
    r = requests.post(
        ANTHROPIC_URL,
        headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json={
            "model": model_id, "max_tokens": 4000,
            "temperature": min(float(temperature), 1.0),
            "system": prompt,
            "messages": [{"role": "user", "content": f"TRANSCRIPT:\n{transcript}"}],
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    r.raise_for_status()
    data = r.json()

    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    u = data.get("usage", {})
    return text, elapsed, u.get("input_tokens", 0), u.get("output_tokens", 0)


# ------------------------------------------------------------- callbacks ---

def do_transcribe(files, key, mode, language_code, use_batch, diarize, num_speakers,
                  progress=gr.Progress()):
    empty = ("", "", None, None)
    if not files:
        return "Add at least one recording first.", *empty
    if not key:
        return "Add your Sarvam key in **Keys and settings** at the top.", *empty

    paths = [f if isinstance(f, str) else f.name for f in files]

    try:
        if use_batch:
            progress(0.2, desc="Sarvam is transcribing — a few minutes for a long call")
            payloads, failures = sarvam_batch(paths, key, mode, language_code, diarize, num_speakers)
            note = ""
            if failures:
                note = "\n\nCould not process: " + "; ".join(failures)
            if not payloads:
                return "No transcripts came back." + note, *empty
        else:
            progress(0.5, desc="Sending to the REST endpoint")
            payloads = [sarvam_rest(paths[0], key, mode, language_code)]
            note = "" if len(paths) == 1 else "\n\nREST mode used only the first file."

        first = payloads[0]
        transcript = first.get("transcript", "")
        lang = first.get("language_code", "")
        diar = first.get("diarized_transcript")

        rows = []
        if diar:
            for e in diar.get("entries", []):
                rows.append([e.get("speaker_id"),
                             round(e.get("start_time_seconds", 0), 2),
                             round(e.get("end_time_seconds", 0), 2),
                             e.get("transcript", "")])
        df = pd.DataFrame(rows, columns=["speaker", "start", "end", "text"]) if rows else None

        speakers = len({r[0] for r in rows}) if rows else 0
        msg = (f"**{len(payloads)} transcript(s) ready.** Language `{lang or 'not reported'}`, "
               f"{len(transcript)} characters, {speakers or '—'} speaker(s).\n\n"
               f"Your `language` field must be `{lang}` — copied from this metadata, "
               f"never guessed from the text." + note)

        return msg, transcript, lang, df, payloads

    except requests.HTTPError as e:
        return f"Sarvam refused: {e.response.status_code} — {e.response.text[:500]}", *empty
    except Exception as e:
        return f"Transcription failed: {e}", *empty


def load_pasted(text, lang):
    if not text.strip():
        return "Nothing to load.", "", ""
    return f"Loaded {len(text)} characters.", text, lang


def do_run(transcript, stt_lang, prompt, version, chosen, temperature, force_json, clean_output,
           k_gemini, k_anthropic, p1i, p1o, p2i, p2o, p3i, p3o, p4i, p4o,
           progress=gr.Progress()):

    blanks = [gr.update(visible=False), gr.update(value=None)] * MAX_PANELS
    nothing = [None, *blanks, None, None]

    if not transcript.strip():
        return "Transcribe something first, or paste a transcript.", *nothing
    if not prompt.strip():
        return "Paste your extraction prompt in step 2.", *nothing
    if not chosen:
        return "Pick at least one model.", *nothing

    overrides = {
        MODELS[MODEL_NAMES[0]][1]: (p1i, p1o),
        MODELS[MODEL_NAMES[1]][1]: (p2i, p2o),
        MODELS[MODEL_NAMES[2]][1]: (p3i, p3o),
        MODELS[MODEL_NAMES[3]][1]: (p4i, p4o),
    }

    results, notes = {}, []
    for i, name in enumerate(chosen):
        vendor, mid, _, _ = MODELS[name]
        key = k_gemini if vendor == "google" else k_anthropic
        if not key:
            notes.append(f"{name} skipped — no API key.")
            continue

        progress((i + 1) / (len(chosen) + 1), desc=f"{name} running")
        try:
            if vendor == "google":
                text, secs, t_in, t_out = call_gemini(mid, key, prompt, transcript,
                                                      temperature, force_json)
            else:
                text, secs, t_in, t_out = call_anthropic(mid, key, prompt, transcript, temperature)

            parsed, err = parse_json_loose(text)
            cleaned, changes = apply_pipeline_rules(parsed, stt_lang) if parsed else (None, [])

            results[name] = {"parsed": parsed, "cleaned": cleaned, "changes": changes,
                             "error": err, "raw_text": text, "seconds": secs,
                             "t_in": t_in, "t_out": t_out,
                             "cost": rupees(name, t_in, t_out, overrides)}

            with open(RUNS_FILE, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({
                    "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "prompt_version": version,
                    "model": name, "temperature": temperature, "stt_language": stt_lang,
                    "seconds": round(secs, 2), "tokens_in": t_in, "tokens_out": t_out,
                    "output": parsed,
                }, ensure_ascii=False) + "\n")

        except requests.HTTPError as e:
            notes.append(f"{name} — {e.response.status_code}: {e.response.text[:300]}")
        except Exception as e:
            notes.append(f"{name} — {e}")

    if not results:
        return "Nothing ran.\n\n" + "\n\n".join(notes), *nothing

    summary = pd.DataFrame([{
        "Model": k,
        "JSON parsed": "yes" if v["parsed"] else "no",
        "Seconds": round(v["seconds"], 2),
        "Tokens in": v["t_in"], "Tokens out": v["t_out"],
        "₹ per call": round(v["cost"], 4),
        "₹ per 1000 calls": round(v["cost"] * 1000, 2),
    } for k, v in results.items()])

    panels = []
    for i in range(MAX_PANELS):
        if i < len(results):
            name = list(results)[i]
            v = results[name]
            label = f"### {name}"
            if v["error"]:
                label += f"\n\n**{v['error']}**\n\n```\n{v['raw_text'][:800]}\n```"
                payload = None
            else:
                payload = v["cleaned"] if (clean_output and v["cleaned"]) else v["parsed"]
                if clean_output and v["changes"]:
                    label += "\n\nPipeline changed: " + "; ".join(v["changes"])
            panels += [gr.update(value=label, visible=True), gr.update(value=payload)]
        else:
            panels += [gr.update(visible=False), gr.update(value=None)]

    ok = {k: v for k, v in results.items() if v["parsed"]}
    diff_df = None
    if len(ok) >= 2:
        flats = {k: flatten(v["cleaned"] if (clean_output and v["cleaned"]) else v["parsed"])
                 for k, v in ok.items()}
        fields = sorted({f for d in flats.values() for f in d})
        rows = []
        for f in fields:
            vals = {k: d.get(f, "—") for k, d in flats.items()}
            agree = len({json.dumps(v, ensure_ascii=False, sort_keys=True)
                         for v in vals.values()}) == 1
            if not agree:
                rows.append({"Field": f, **{k: str(v) for k, v in vals.items()}})
        diff_df = pd.DataFrame(rows) if rows else pd.DataFrame(
            [{"Field": "Every field matches across models."}])

    out_path = f"bench_{version or 'run'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"prompt_version": version, "temperature": temperature,
                   "stt_language": stt_lang, "transcript": transcript,
                   "results": {k: {kk: vv for kk, vv in v.items() if kk != "raw_text"}
                               for k, v in results.items()}},
                  fh, ensure_ascii=False, indent=2)

    msg = f"Ran {len(results)} model(s)."
    if notes:
        msg += "\n\n" + "\n\n".join(notes)

    return msg, *panels, summary, diff_df, out_path


def load_history():
    if not os.path.exists(RUNS_FILE):
        return pd.DataFrame([{"info": "No runs yet."}]), None
    rows = []
    with open(RUNS_FILE, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                h = json.loads(line)
                rows.append({k: v for k, v in h.items() if k != "output"})
    return pd.DataFrame(rows), RUNS_FILE


# ------------------------------------------------------------------- app ---

with gr.Blocks(title="DeskLens Bench") as demo:
    gr.Markdown("# DeskLens Bench\nAudio → Sarvam Saaras v3 → extraction → compare. "
                "Every stage shows its raw output.")

    stt_payloads = gr.State(None)

    with gr.Accordion("Keys and settings", open=True):
        with gr.Row():
            k_sarvam = gr.Textbox(label="Sarvam key", type="password",
                                  value=os.getenv("SARVAM_API_KEY", ""))
            k_gemini = gr.Textbox(label="Google AI Studio key", type="password",
                                  value=os.getenv("GEMINI_API_KEY", ""))
            k_anthropic = gr.Textbox(label="Anthropic key", type="password",
                                     value=os.getenv("ANTHROPIC_API_KEY", ""))
        with gr.Row():
            stt_mode = gr.Dropdown(["translate", "transcribe", "codemix", "translit", "verbatim"],
                                   value="translate", label="Sarvam mode",
                                   info="translate gives English. codemix keeps Hinglish as spoken.")
            language_code = gr.Textbox(value="unknown", label="Language code",
                                       info="'unknown' lets Sarvam detect it")
            temperature = gr.Slider(0, 2, value=0, step=0.1, label="Temperature",
                                    info="0 for strict extraction. Anthropic caps at 1.")
        with gr.Row():
            use_batch = gr.Checkbox(True, label="Batch API",
                                    info="On: up to 2 hours + diarization. Off: under 30 seconds only.")
            diarize = gr.Checkbox(True, label="Speaker diarization")
            num_speakers = gr.Number(2, label="Speakers", precision=0)
            force_json = gr.Checkbox(True, label="Force JSON (Gemini)")
            clean_output = gr.Checkbox(True, label="Show pipeline-cleaned output")

        with gr.Accordion("Pricing (₹ per 1M tokens) — estimates, edit to match your rates", open=False):
            price_boxes = []
            for name in MODEL_NAMES:
                _, _, d_in, d_out = MODELS[name]
                with gr.Row():
                    price_boxes.append(gr.Number(d_in, label=f"{name} — in"))
                    price_boxes.append(gr.Number(d_out, label=f"{name} — out"))

    with gr.Tab("1 · Transcribe"):
        with gr.Row():
            with gr.Column():
                audio_files = gr.File(label="Call recordings", file_count="multiple",
                                      file_types=[".mp3", ".wav", ".m4a", ".aac", ".ogg",
                                                  ".opus", ".flac", ".amr", ".webm"])
                btn_stt = gr.Button("Transcribe with Sarvam", variant="primary")
                with gr.Accordion("Or paste a transcript instead", open=False):
                    pasted = gr.Textbox(label="Transcript", lines=6)
                    pasted_lang = gr.Textbox(label="STT language code", placeholder="en-IN")
                    btn_paste = gr.Button("Use this transcript")
            with gr.Column():
                stt_status = gr.Markdown("Upload recordings, then press Transcribe.")
                stt_lang_out = gr.Textbox(label="Detected language code", interactive=False)
        transcript_box = gr.Textbox(label="Transcript (editable — edit to retest without re-transcribing)",
                                    lines=12)
        with gr.Accordion("Diarized turns", open=False):
            diar_table = gr.Dataframe(label="Who said what", wrap=True)
        with gr.Accordion("Raw Sarvam response", open=False):
            raw_json = gr.JSON(label="Everything Sarvam returned")

    with gr.Tab("2 · Prompt and run"):
        with gr.Row():
            prompt_box = gr.Textbox(label="Extraction prompt", lines=16,
                                    placeholder="Paste your derma_v6 prompt here.")
            with gr.Column(scale=0):
                version_box = gr.Textbox(value="derma_v6", label="Version label")
                prompt_file = gr.File(label="Load a saved prompt", file_types=[".txt", ".md"])
        model_picker = gr.CheckboxGroup(MODEL_NAMES, label="Models",
                                        value=["Gemini 2.5 Flash-Lite", "Claude Haiku 4.5"])
        btn_run = gr.Button("Run selected models", variant="primary")
        run_status = gr.Markdown()

    with gr.Tab("3 · Compare"):
        summary_table = gr.Dataframe(label="Summary", wrap=True)
        panel_md, panel_json = [], []
        with gr.Row():
            for i in range(MAX_PANELS):
                with gr.Column():
                    panel_md.append(gr.Markdown(visible=False))
                    panel_json.append(gr.JSON())
        gr.Markdown("### Fields where the models disagree")
        diff_table = gr.Dataframe(wrap=True)
        download_out = gr.File(label="Download this comparison")

    with gr.Tab("Run history"):
        btn_hist = gr.Button("Refresh")
        hist_table = gr.Dataframe(wrap=True)
        hist_file = gr.File(label="Download history")
        gr.Markdown("Hugging Face wipes this file when the Space sleeps. "
                    "Download it at the end of a testing session.")

    # wiring
    btn_stt.click(
        do_transcribe,
        [audio_files, k_sarvam, stt_mode, language_code, use_batch, diarize, num_speakers],
        [stt_status, transcript_box, stt_lang_out, diar_table, stt_payloads],
    ).then(lambda p: p, stt_payloads, raw_json)

    btn_paste.click(load_pasted, [pasted, pasted_lang],
                    [stt_status, transcript_box, stt_lang_out])

    prompt_file.upload(
        lambda f: open(f.name if not isinstance(f, str) else f, encoding="utf-8").read(),
        prompt_file, prompt_box)

    panel_outputs = []
    for i in range(MAX_PANELS):
        panel_outputs += [panel_md[i], panel_json[i]]

    btn_run.click(
        do_run,
        [transcript_box, stt_lang_out, prompt_box, version_box, model_picker, temperature,
         force_json, clean_output, k_gemini, k_anthropic] + price_boxes,
        [run_status] + panel_outputs + [summary_table, diff_table, download_out],
    )

    btn_hist.click(load_history, None, [hist_table, hist_file])

demo.launch()
