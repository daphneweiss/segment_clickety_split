#!/usr/bin/env python3
"""
TotalRecal — Recording Segmentation Applet
===========================================
Flask-based web app that wraps the segmentation pipeline into a
four-step interactive workflow:

  Step 1: Setup experiment (speaker, conditions, parameters)
  Step 2: Review & adjust segments per condition
  Step 3: Pick tokens to export (multiple per word allowed)
  Step 4: Bulk export

Run:
  cd stim_pipeline/
  python app.py            # starts on http://localhost:5000
  python app.py --port 8080
"""

import json
import pickle
import time
import argparse
from pathlib import Path
from collections import defaultdict

from flask import Flask, request, jsonify, send_file, send_from_directory, abort
import numpy as np
import soundfile as sf

# Import the processing engine
import segment_recording as seg_engine

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__, static_folder=None)

# These are set at startup from --project_dir
PROJECT_ROOT = None
RECORDINGS_DIR = None
EXPERIMENT_DIR = None
STIMLISTS_DIR = None
SESSIONS_DIR = None

# In-memory session state
session_state = {
    "speaker_id": "",
    "conditions": {},      # cond_name → { audio_files, stimlist, segments, ... }
    "parameters": {
        "min_word_duration_ms": 500,
        "min_silence_ms": 150,
        "min_segment_ms": 150,
        "word_dur_max_ms": 1400,
        "denoise": True,
    },
    "step": 1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _rec_dir(speaker: str, condition: str) -> Path:
    """Return the recordings directory for a speaker/condition pair.

    If the condition subfolder doesn't exist, fall back to the speaker dir
    itself (flat layout: audio files live directly inside the speaker folder).
    """
    d = RECORDINGS_DIR / speaker / condition
    if d.is_dir():
        return d
    return RECORDINGS_DIR / speaker


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    html_path = Path(__file__).parent / "index.html"
    resp = send_file(str(html_path))
    resp.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    return resp


# ---------------------------------------------------------------------------
# API: Discovery
# ---------------------------------------------------------------------------
@app.route("/api/speakers")
def api_speakers():
    """List speakers from recordings/."""
    if not RECORDINGS_DIR.exists():
        return jsonify([])
    speakers = sorted([
        d.name for d in RECORDINGS_DIR.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])
    return jsonify(speakers)


@app.route("/api/conditions/<speaker>")
def api_conditions(speaker):
    """List recording condition folders for a speaker."""
    spk_dir = RECORDINGS_DIR / speaker
    if not spk_dir.is_dir():
        return jsonify([])
    conditions = []
    for d in sorted(spk_dir.iterdir()):
        if not d.is_dir() or d.name.startswith(".") or d.name == "raw":
            continue
        wav_files = sorted([
            f.name for f in d.iterdir()
            if f.suffix.lower() in (".wav", ".flac", ".mp3")
        ])
        conditions.append({
            "name": d.name,
            "audio_files": wav_files,
            "n_files": len(wav_files),
        })
    # Flat layout: audio files live directly in the speaker folder with no
    # condition subfolders.  Return a single editable synthetic condition.
    if not conditions:
        wav_files = sorted([
            f.name for f in spk_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".wav", ".flac", ".mp3")
        ])
        if wav_files:
            conditions.append({
                "name": "session",
                "audio_files": wav_files,
                "n_files": len(wav_files),
                "flat": True,
            })
    return jsonify(conditions)


@app.route("/api/stimlists")
def api_stimlists():
    """List available stimulus list files."""
    if not STIMLISTS_DIR.exists():
        return jsonify([])
    files = sorted([
        f.name for f in STIMLISTS_DIR.iterdir()
        if f.suffix in (".txt", ".csv") and not f.name.startswith(".")
    ])
    return jsonify(files)


@app.route("/api/upload_stimlist", methods=["POST"])
def api_upload_stimlist():
    """Upload a stimulus list .txt file into STIMLISTS_DIR."""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400
    safe_name = Path(f.filename).name
    if not safe_name.endswith((".txt", ".csv")):
        safe_name += ".txt"
    STIMLISTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = STIMLISTS_DIR / safe_name
    f.save(str(dest))
    # Return updated list
    files = sorted([
        p.name for p in STIMLISTS_DIR.iterdir()
        if p.suffix in (".txt", ".csv") and not p.name.startswith(".")
    ])
    return jsonify({"status": "ok", "name": safe_name, "stimlists": files})


@app.route("/api/stimlist_content/<filename>")
def api_stimlist_content(filename):
    """Return the words in a stimulus list."""
    safe_name = Path(filename).name  # prevent path traversal
    fpath = STIMLISTS_DIR / safe_name
    if not fpath.is_file():
        abort(404)
    words = seg_engine.load_stimulus_list(str(fpath))
    return jsonify(words)


# ---------------------------------------------------------------------------
# API: Audio streaming
# ---------------------------------------------------------------------------
@app.route("/api/audio/<speaker>/<condition>/<filename>")
def api_audio_file(speaker, condition, filename):
    """Serve an individual audio file from recordings."""
    safe_path = _rec_dir(speaker, condition) / Path(filename).name
    if not safe_path.is_file():
        abort(404)
    return send_file(str(safe_path), mimetype="audio/wav")


@app.route("/api/audio_combined/<speaker>/<condition>")
def api_audio_combined(speaker, condition):
    """Serve the combined/denoised audio for a condition (from experiment dir)."""
    cond_dir = EXPERIMENT_DIR / speaker / condition
    # Prefer denoised, then combined, then first recording
    for candidate in ["denoised.wav", "_combined.wav"]:
        p = cond_dir / candidate
        if p.is_file():
            return send_file(str(p), mimetype="audio/wav")
    # Fallback to first recording
    rec_dir = _rec_dir(speaker, condition)
    for f in sorted(rec_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in (".wav", ".flac"):
            return send_file(str(f), mimetype="audio/wav")
    abort(404)


# ---------------------------------------------------------------------------
# API: Detection
# ---------------------------------------------------------------------------
@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Run detection on a single condition."""
    data = request.json
    speaker = data["speaker"]
    condition = data["condition"]
    stimlist_file = data.get("stimlist")
    params = data.get("parameters", {})

    # Apply parameters
    if "min_word_duration_ms" in params:
        seg_engine.WORD_DUR_MIN_MS = params["min_word_duration_ms"]
    if "min_silence_ms" in params:
        seg_engine.MIN_SILENCE_MS = params["min_silence_ms"]
    if "word_dur_max_ms" in params:
        seg_engine.WORD_DUR_MAX_MS = params["word_dur_max_ms"]
    if "min_segment_ms" in params:
        seg_engine.MIN_SEGMENT_MS = params["min_segment_ms"]
    if "silence_margin_ms" in params:
        seg_engine.SILENCE_MARGIN_MS = params["silence_margin_ms"]

    # Find audio files
    rec_dir = _rec_dir(speaker, condition)
    if not rec_dir.is_dir():
        return jsonify({"error": f"No recording directory: {rec_dir}"}), 400
    audio_files = seg_engine.find_all_audio_files(rec_dir)
    if not audio_files:
        return jsonify({"error": f"No audio files in {rec_dir}"}), 400

    # Stimulus list
    stimlist_path = None
    if stimlist_file:
        stimlist_path = str(STIMLISTS_DIR / Path(stimlist_file).name)

    # Output directory
    out_dir = EXPERIMENT_DIR / speaker / condition

    denoise = params.get("denoise", True)

    # Run detection
    if len(audio_files) > 1:
        proposal = seg_engine.process_multiple(
            audio_files, stimlist_path, out_dir,
            condition=condition, speaker_id=speaker, denoise=denoise,
        )
    else:
        proposal = seg_engine.process_single(
            str(audio_files[0]), stimlist_path, out_dir,
            condition=condition, speaker_id=speaker, denoise=denoise,
        )

    # Store in session
    session_state["speaker_id"] = speaker
    cond_state = session_state["conditions"].setdefault(condition, {})
    cond_state["segments"] = proposal["segments"]
    cond_state["stimlist"] = stimlist_file
    cond_state["audio_files"] = [str(f) for f in audio_files]
    cond_state["detected"] = True
    cond_state["reviewed"] = False

    # Return the proposal (segments JSON is already saved to disk by process_*)
    return jsonify({
        "status": "ok",
        "condition": condition,
        "n_segments": len(proposal["segments"]),
        "n_words": sum(1 for s in proposal["segments"] if s.get("segment_type") == "word"),
        "stimulus_list": proposal.get("stimulus_list", []),
        "audio_duration": proposal["audio_duration"],
    })


@app.route("/api/detect_all", methods=["POST"])
def api_detect_all():
    """Run detection on all configured conditions."""
    data = request.json
    speaker = data["speaker"]
    conditions = data["conditions"]  # list of {condition, stimlist}
    params = data.get("parameters", {})

    results = []
    for cond_cfg in conditions:
        cond_data = {
            "speaker": speaker,
            "condition": cond_cfg["condition"],
            "stimlist": cond_cfg.get("stimlist"),
            "parameters": params,
        }
        # Reuse the single-condition detect
        with app.test_request_context(json=cond_data):
            resp = api_detect()
            if isinstance(resp, tuple):
                results.append({"condition": cond_cfg["condition"], "error": resp[0].json["error"]})
            else:
                results.append(resp.json)

    return jsonify(results)


# ---------------------------------------------------------------------------
# API: Segments (load/save)
# ---------------------------------------------------------------------------
@app.route("/api/segments/<speaker>/<condition>")
def api_segments(speaker, condition):
    """Load segments JSON for a condition."""
    cond_dir = EXPERIMENT_DIR / speaker / condition
    # Prefer reviewed, then proposed
    for candidate in ["reviewed_segments.json", "proposed_segments.json"]:
        p = cond_dir / candidate
        if p.is_file():
            with open(p) as f:
                data = json.load(f)
            data["_source"] = candidate
            return jsonify(data)
    return jsonify({"error": "No segments found"}), 404


@app.route("/api/save_segments", methods=["POST"])
def api_save_segments():
    """Save reviewed segments for a condition."""
    data = request.json
    speaker = data["speaker_id"]
    condition = data["condition"]

    cond_dir = EXPERIMENT_DIR / speaker / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    out_path = cond_dir / "reviewed_segments.json"
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)

    # Update session
    cond_state = session_state["conditions"].setdefault(condition, {})
    cond_state["segments"] = data.get("segments", [])
    cond_state["reviewed"] = True

    return jsonify({"status": "ok", "path": str(out_path)})


# ---------------------------------------------------------------------------
# API: Export tokens
# ---------------------------------------------------------------------------
@app.route("/api/export", methods=["POST"])
def api_export():
    """Export tokens for a condition."""
    data = request.json
    speaker = data["speaker"]
    condition = data["condition"]
    selected_tokens = data.get("selected_tokens", {})  # word → [token_indices]

    cond_dir = EXPERIMENT_DIR / speaker / condition

    # Load segments
    seg_path = cond_dir / "reviewed_segments.json"
    if not seg_path.is_file():
        seg_path = cond_dir / "proposed_segments.json"
    if not seg_path.is_file():
        return jsonify({"error": "No segments file found"}), 400

    with open(seg_path) as f:
        seg_data = json.load(f)

    # Load audio
    audio_path = None
    for candidate in ["denoised.wav", "_combined.wav"]:
        p = cond_dir / candidate
        if p.is_file():
            audio_path = str(p)
            break
    if not audio_path:
        # Fallback: original audio
        audio_path = seg_data.get("audio_file") or seg_data.get("denoised_audio_file")
    if not audio_path or not Path(audio_path).is_file():
        rec_dir = RECORDINGS_DIR / speaker / condition
        candidates = seg_engine.find_all_audio_files(rec_dir)
        if candidates:
            audio_path = str(candidates[0])
    if not audio_path:
        return jsonify({"error": "Audio file not found"}), 400

    audio, sr = sf.read(audio_path, dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    segments = seg_data["segments"]
    out_dir = cond_dir / "tokens"

    # Assign positional token_index per word among accepted segments only.
    # This matches the frontend's Step 3 display (ti+1 within accepted group),
    # and also matches what export_tokens() produces when it re-sequences from 1.
    word_counter_renum = defaultdict(int)
    for seg in segments:
        if seg.get("segment_type") == "word" and seg.get("accepted", True):
            name = seg.get("assigned_name", "")
            word_counter_renum[name] += 1
            seg["token_index"] = word_counter_renum[name]
    for seg in segments:
        if seg.get("segment_type") == "word" and seg.get("accepted", True):
            seg["cluster_size"] = word_counter_renum.get(seg.get("assigned_name", ""), 1)

    # If selected_tokens is provided, filter to only the selected positional indices
    if selected_tokens:
        filtered = []
        for seg in segments:
            if seg.get("segment_type") != "word":
                continue
            if not seg.get("accepted", True):
                continue
            name = seg.get("assigned_name", "")
            ti = seg.get("token_index", 1)
            if name in selected_tokens:
                if ti in selected_tokens[name]:
                    filtered.append(seg)
        exported, manifest = seg_engine.export_tokens(audio, sr, filtered, str(out_dir), speaker)
    else:
        exported, manifest = seg_engine.export_tokens(audio, sr, segments, str(out_dir), speaker)

    return jsonify({
        "status": "ok",
        "n_exported": len(exported),
        "tokens_per_word": manifest["tokens_per_word"],
        "output_dir": str(out_dir),
    })


@app.route("/api/export_all", methods=["POST"])
def api_export_all():
    """Export tokens for all conditions."""
    data = request.json
    speaker = data["speaker"]
    conditions = data["conditions"]  # list of condition names
    all_selections = data.get("selections", {})  # cond → { word → [indices] }

    results = []
    for cond in conditions:
        sel = all_selections.get(cond, {})
        cond_data = {
            "speaker": speaker,
            "condition": cond,
            "selected_tokens": sel,
        }
        with app.test_request_context(json=cond_data):
            resp = api_export()
            if isinstance(resp, tuple):
                results.append({"condition": cond, "error": resp[0].json["error"]})
            else:
                results.append({"condition": cond, **resp.json})

    return jsonify(results)


# ---------------------------------------------------------------------------
# API: Session save/load (pickle)
# ---------------------------------------------------------------------------
@app.route("/api/save_session", methods=["POST"])
def api_save_session():
    """Save current session state to a pickle file."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    data = request.json or {}
    name = data.get("name", f"session_{session_state['speaker_id']}_{int(time.time())}")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = SESSIONS_DIR / f"{safe_name}.pkl"

    # Include review progress from frontend
    if "reviewProgress" in data:
        session_state["reviewProgress"] = data["reviewProgress"]
    if "activeCond" in data:
        session_state["activeCond"] = data["activeCond"]
    if "currentStep" in data:
        session_state["step"] = data["currentStep"]

    with open(path, "wb") as f:
        pickle.dump(session_state, f)

    return jsonify({"status": "ok", "path": str(path), "name": safe_name})


@app.route("/api/load_session", methods=["POST"])
def api_load_session():
    """Load session state from a pickle file."""
    global session_state
    data = request.json
    name = data.get("name", "")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    path = SESSIONS_DIR / f"{safe_name}.pkl"

    if not path.is_file():
        return jsonify({"error": f"Session file not found: {path}"}), 404

    with open(path, "rb") as f:
        session_state = pickle.load(f)

    return jsonify({
        "status": "ok",
        "speaker_id": session_state.get("speaker_id", ""),
        "conditions": list(session_state.get("conditions", {}).keys()),
        "step": session_state.get("step", 1),
        "currentStep": session_state.get("step", 1),
        "activeCond": session_state.get("activeCond", ""),
        "reviewProgress": session_state.get("reviewProgress", {}),
    })


@app.route("/api/sessions")
def api_sessions():
    """List saved sessions."""
    if not SESSIONS_DIR.exists():
        return jsonify([])
    files = sorted([
        {"name": f.stem, "modified": f.stat().st_mtime, "size": f.stat().st_size}
        for f in SESSIONS_DIR.iterdir()
        if f.suffix == ".pkl"
    ], key=lambda x: x["modified"], reverse=True)
    return jsonify(files)


@app.route("/api/session")
def api_session():
    """Return current session state (without heavy data)."""
    return jsonify({
        "speaker_id": session_state["speaker_id"],
        "conditions": {
            k: {
                "detected": v.get("detected", False),
                "reviewed": v.get("reviewed", False),
                "stimlist": v.get("stimlist"),
                "n_segments": len(v.get("segments", [])),
            }
            for k, v in session_state["conditions"].items()
        },
        "parameters": session_state["parameters"],
        "step": session_state["step"],
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment Clickety Split")
    parser.add_argument("project_dir",
                        help="Project directory containing recordings/ and experiment/ folders")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    PROJECT_ROOT = Path(args.project_dir).resolve()
    RECORDINGS_DIR = PROJECT_ROOT / "recordings"
    EXPERIMENT_DIR = PROJECT_ROOT / "experiment"
    STIMLISTS_DIR = EXPERIMENT_DIR / "stimulus_lists"
    SESSIONS_DIR = PROJECT_ROOT / "sessions"
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n  Segment Clickety Split")
    print(f"  http://{args.host}:{args.port}")
    print(f"  Project:    {PROJECT_ROOT}")
    print(f"  Recordings: {RECORDINGS_DIR}")
    print(f"  Experiment: {EXPERIMENT_DIR}\n")

    app.run(host=args.host, port=args.port, debug=args.debug)
