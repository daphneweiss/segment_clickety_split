#!/usr/bin/env python3
"""
Recording Session Segmenter (v3.1)
====================================
Segments recording sessions where speakers produce each word MULTIPLE times.

Changes from v3 (og_dl):
  - Handles multiple .wav files per condition directory (e.g. filler_pseudo/,
    filler_word/) by concatenating them before segmentation.

Workflow:
  1. DETECT:    Silence-based segmentation → filter intro/crosstalk → gap-based
                word labeling (cluster repetitions by inter-word gap size)
  2. REVIEW:    Step through tokens one at a time in the browser tool:
                - Accept/adjust/reject boundaries
                - Accept or change the word label (autocomplete from stimulus list)
  3. EXPORT:    Export all accepted tokens with -1, -2, -3 suffixes per word
  4. FINALIZE:  Select best token per word → best goes to final/ (no suffix),
                rest go to alternates/ (with suffix)

Usage:
  python segment_recording.py detect \
      --audio session.wav --stimlist stimuli.txt \
      --speaker_id m1 --condition critical_s_normal \
      --output_dir output/

  python segment_recording.py batch \
      --speaker_dir speaker_01/ --output_dir speaker_01_seg/ --speaker_id m1

  # → review in browser (review_tool.html)

  python segment_recording.py export \
      --audio session.wav --segments reviewed_segments.json \
      --output_dir output/tokens/ --speaker_id m1

  # → select best tokens in browser (review_tool.html, Select Best mode)

  python segment_recording.py finalize \
      --tokens_dir output/tokens/ --selections best_selections.json \
      --output_dir output/final/ --speaker_id m1
"""

import numpy as np
import parselmouth
from parselmouth.praat import call
import soundfile as sf
from scipy.signal import medfilt
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend — no X display required
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mpatches = None
import json
import csv as _csv
import argparse
from pathlib import Path
import warnings
import re
import shutil
from collections import defaultdict

try:
    import noisereduce as nr
    HAS_NOISEREDUCE = True
except ImportError:
    HAS_NOISEREDUCE = False

try:
    import webrtcvad
    HAS_WEBRTCVAD = True
except ImportError:
    HAS_WEBRTCVAD = False

try:
    # Prefer the full torch-based silero_vad (best accuracy).
    from silero_vad import load_silero_vad, get_speech_timestamps
    HAS_SILERO = True
    _silero_model = None  # lazy-loaded singleton
except ImportError:
    try:
        # Fallback: torch-free ONNX wrapper for environments without PyTorch.
        from silero_onnx import load_silero_vad, get_speech_timestamps
        HAS_SILERO = True
        _silero_model = None
    except ImportError:
        HAS_SILERO = False
        _silero_model = None

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================================
# CONFIGURATION
# ============================================================================

MIN_SILENCE_MS = 150
MIN_SEGMENT_MS = 150
SILENCE_MARGIN_MS = 25
ENERGY_SMOOTHING_MS = 15
WORD_DUR_MAX_MS = 1400
WORD_DUR_MIN_MS = 500
LONG_SEGMENT_FACTOR = 2.2
EXPORT_PAD_MS = 20
FADE_MS = 3


# ============================================================================
# 1a. NOISE REDUCTION
# ============================================================================

def reduce_background_noise(audio, sr):
    """
    Apply spectral-gating noise reduction to a full recording.

    Estimates the noise profile from the quietest 20% of the recording
    (the silence between words), then applies spectral subtraction to
    remove stationary background noise (room tone, hum, hiss).

    Returns the denoised audio and a dict of stats.
    """
    if not HAS_NOISEREDUCE:
        print("  ⚠ noisereduce not installed — skipping noise reduction")
        print("    pip install noisereduce")
        return audio, {"applied": False, "reason": "noisereduce not installed"}

    rms_before = np.sqrt(np.mean(audio ** 2))

    # Estimate noise profile from silent portions.
    # Split audio into short frames, find the quietest 20%, concatenate as noise sample.
    frame_len = int(0.05 * sr)  # 50ms frames
    n_frames = len(audio) // frame_len
    if n_frames < 10:
        return audio, {"applied": False, "reason": "audio too short"}

    frame_rms = np.array([
        np.sqrt(np.mean(audio[i * frame_len:(i + 1) * frame_len] ** 2))
        for i in range(n_frames)
    ])

    # Bottom 20% by energy = silence / room tone
    threshold = np.percentile(frame_rms, 20)
    noise_frames = [audio[i * frame_len:(i + 1) * frame_len]
                    for i in range(n_frames) if frame_rms[i] <= threshold]

    if len(noise_frames) < 3:
        return audio, {"applied": False, "reason": "could not isolate noise profile"}

    noise_clip = np.concatenate(noise_frames)

    # Apply spectral gating
    reduced = nr.reduce_noise(
        y=audio,
        sr=sr,
        y_noise=noise_clip,
        stationary=True,
        prop_decrease=0.85,     # how aggressively to remove noise (0-1)
        n_fft=2048,
        freq_mask_smooth_hz=200,
    )

    rms_after = np.sqrt(np.mean(reduced ** 2))
    noise_rms = np.sqrt(np.mean(noise_clip ** 2))

    stats = {
        "applied": True,
        "rms_before": float(rms_before),
        "rms_after": float(rms_after),
        "noise_floor_rms": float(noise_rms),
        "noise_frames_used": len(noise_frames),
        "reduction_db": float(20 * np.log10((rms_before + 1e-12) / (rms_after + 1e-12))),
    }

    return reduced, stats


# ============================================================================
# 1. SILENCE-BASED SEGMENTATION (unchanged from v2)
# ============================================================================

def compute_energy_envelope(audio, sr, frame_ms=5):
    frame_len = int(frame_ms * sr / 1000)
    hop = frame_len // 2
    n_frames = (len(audio) - frame_len) // hop + 1
    if n_frames <= 0:
        return np.array([0]), np.array([0])
    energy = np.zeros(n_frames)
    times = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop
        end = start + frame_len
        if end > len(audio):
            break
        energy[i] = np.sqrt(np.mean(audio[start:end] ** 2))
        times[i] = (start + frame_len // 2) / sr
    kernel = max(3, int(ENERGY_SMOOTHING_MS / frame_ms)) | 1
    energy = medfilt(energy, kernel_size=kernel)
    return times, energy


def detect_segments_raw(audio, sr):
    snd = parselmouth.Sound(audio, sampling_frequency=sr)
    duration = snd.get_total_duration()
    times, energy = compute_energy_envelope(audio, sr)

    energy_nonzero = energy[energy > 0]
    if len(energy_nonzero) == 0:
        return [], {"times": times, "energy": energy,
                    "is_speech": np.zeros_like(energy, dtype=bool), "energy_threshold": 0}

    energy_db = 20 * np.log10(energy_nonzero + 1e-12)
    e_silence = np.percentile(energy_db, 20)
    e_speech = np.percentile(energy_db, 55)
    e_threshold = e_silence + 0.45 * (e_speech - e_silence)
    e_threshold_lin = 10 ** (e_threshold / 20)
    is_speech = energy > e_threshold_lin

    frame_ms = 5
    min_silence_frames = int(MIN_SILENCE_MS / frame_ms)
    min_segment_frames = int(MIN_SEGMENT_MS / frame_ms)

    cleaned = is_speech.copy()
    gap_start = None
    for i in range(len(cleaned)):
        if not cleaned[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                if i - gap_start < min_silence_frames:
                    cleaned[gap_start:i] = True
                gap_start = None

    speech_start = None
    for i in range(len(cleaned)):
        if cleaned[i]:
            if speech_start is None:
                speech_start = i
        else:
            if speech_start is not None:
                if i - speech_start < min_segment_frames:
                    cleaned[speech_start:i] = False
                speech_start = None

    segments = []
    seg_start = None
    for i in range(len(cleaned)):
        if cleaned[i] and seg_start is None:
            seg_start = i
        elif not cleaned[i] and seg_start is not None:
            start_sec = max(0, times[seg_start] - SILENCE_MARGIN_MS / 1000)
            end_sec = min(duration, times[min(i, len(times) - 1)] + SILENCE_MARGIN_MS / 1000)
            dur_ms = (end_sec - start_sec) * 1000
            if dur_ms >= MIN_SEGMENT_MS:
                segments.append({
                    "start": round(start_sec, 4),
                    "end": round(end_sec, 4),
                    "duration_ms": round(dur_ms, 1),
                })
            seg_start = None

    if seg_start is not None:
        end_sec = min(duration, times[-1] + SILENCE_MARGIN_MS / 1000)
        start_sec = max(0, times[seg_start] - SILENCE_MARGIN_MS / 1000)
        dur_ms = (end_sec - start_sec) * 1000
        if dur_ms >= MIN_SEGMENT_MS:
            segments.append({"start": round(start_sec, 4), "end": round(end_sec, 4),
                             "duration_ms": round(dur_ms, 1)})

    return segments, {"times": times, "energy": energy, "is_speech": cleaned,
                      "energy_threshold": float(e_threshold_lin)}


# ============================================================================
# 1c. WEBRTC VAD-BASED SEGMENTATION
# ============================================================================

def detect_segments_vad(audio, sr):
    """
    Use WebRTC VAD for speech activity detection, then refine boundaries
    with the energy envelope for sub-frame precision.

    WebRTC VAD is a trained model that discriminates speech from breaths,
    clicks, lip smacks, and ambient noise — much more accurate than
    pure energy thresholding for speech recordings.
    """
    if not HAS_WEBRTCVAD:
        print("  ⚠ webrtcvad not installed — falling back to energy-based detection")
        return detect_segments_raw(audio, sr)

    duration = len(audio) / sr

    # WebRTC VAD requires 8/16/32/48 kHz, 16-bit PCM.
    # Resample to 16 kHz if needed.
    target_sr = 16000
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), target_sr)
        audio_16k = resample_poly(audio, target_sr // g, int(sr) // g)
    else:
        audio_16k = audio.copy()

    # Clamp and convert to 16-bit PCM bytes
    audio_16k = np.clip(audio_16k, -1.0, 1.0)
    pcm = (audio_16k * 32767).astype(np.int16).tobytes()

    # Run VAD at aggressiveness 3 (most aggressive = rejects more non-speech)
    vad = webrtcvad.Vad(3)
    frame_ms = 30  # WebRTC supports 10, 20, or 30 ms frames
    samples_per_frame = int(target_sr * frame_ms / 1000)
    bytes_per_frame = samples_per_frame * 2  # 16-bit = 2 bytes/sample

    vad_flags = []
    for i in range(0, len(pcm) - bytes_per_frame + 1, bytes_per_frame):
        frame = pcm[i:i + bytes_per_frame]
        if len(frame) < bytes_per_frame:
            break
        vad_flags.append(vad.is_speech(frame, target_sr))

    n_vad = len(vad_flags)
    if n_vad == 0:
        return detect_segments_raw(audio, sr)

    # Convert to numpy boolean array aligned to time
    is_speech_vad = np.array(vad_flags, dtype=bool)
    vad_times = np.arange(n_vad) * (frame_ms / 1000) + (frame_ms / 2000)

    # --- Smooth: fill short silence gaps and remove short speech bursts ---
    min_silence_frames = max(1, int(MIN_SILENCE_MS / frame_ms))
    min_segment_frames = max(1, int(MIN_SEGMENT_MS / frame_ms))

    smoothed = is_speech_vad.copy()

    # Fill short gaps (silence runs shorter than MIN_SILENCE_MS)
    gap_start = None
    for i in range(len(smoothed)):
        if not smoothed[i]:
            if gap_start is None:
                gap_start = i
        else:
            if gap_start is not None:
                if i - gap_start < min_silence_frames:
                    smoothed[gap_start:i] = True
                gap_start = None

    # Remove short speech bursts (shorter than MIN_SEGMENT_MS)
    speech_start = None
    for i in range(len(smoothed)):
        if smoothed[i]:
            if speech_start is None:
                speech_start = i
        else:
            if speech_start is not None:
                if i - speech_start < min_segment_frames:
                    smoothed[speech_start:i] = False
                speech_start = None

    # --- Extract segments with energy-refined boundaries ---
    # Compute the energy envelope at the original sample rate for precise boundaries
    times_e, energy_e = compute_energy_envelope(audio, sr)

    segments = []
    seg_start = None
    for i in range(len(smoothed)):
        if smoothed[i] and seg_start is None:
            seg_start = i
        elif not smoothed[i] and seg_start is not None:
            # VAD-level boundaries (coarse, 30ms resolution)
            coarse_start = vad_times[seg_start] - frame_ms / 2000
            coarse_end = vad_times[min(i - 1, n_vad - 1)] + frame_ms / 2000

            # Refine using energy: search within ±1 frame for the energy onset/offset
            start_sec = _refine_boundary(times_e, energy_e, coarse_start, direction='start')
            end_sec = _refine_boundary(times_e, energy_e, coarse_end, direction='end')

            # Apply silence margin
            start_sec = max(0, start_sec - SILENCE_MARGIN_MS / 1000)
            end_sec = min(duration, end_sec + SILENCE_MARGIN_MS / 1000)

            dur_ms = (end_sec - start_sec) * 1000
            if dur_ms >= MIN_SEGMENT_MS:
                segments.append({
                    "start": round(start_sec, 4),
                    "end": round(end_sec, 4),
                    "duration_ms": round(dur_ms, 1),
                })
            seg_start = None

    # Handle segment that runs to end
    if seg_start is not None:
        coarse_start = vad_times[seg_start] - frame_ms / 2000
        coarse_end = vad_times[-1] + frame_ms / 2000
        start_sec = _refine_boundary(times_e, energy_e, coarse_start, direction='start')
        end_sec = _refine_boundary(times_e, energy_e, coarse_end, direction='end')
        start_sec = max(0, start_sec - SILENCE_MARGIN_MS / 1000)
        end_sec = min(duration, end_sec + SILENCE_MARGIN_MS / 1000)
        dur_ms = (end_sec - start_sec) * 1000
        if dur_ms >= MIN_SEGMENT_MS:
            segments.append({"start": round(start_sec, 4), "end": round(end_sec, 4),
                             "duration_ms": round(dur_ms, 1)})

    # Build analysis dict (energy data for display, VAD-based speech mask)
    # Interpolate VAD speech mask onto energy time grid for display
    is_speech_display = np.zeros(len(times_e), dtype=bool)
    for i, t in enumerate(times_e):
        vad_idx = int(t / (frame_ms / 1000))
        if 0 <= vad_idx < n_vad:
            is_speech_display[i] = smoothed[vad_idx]

    energy_nonzero = energy_e[energy_e > 0]
    if len(energy_nonzero) > 0:
        energy_db = 20 * np.log10(energy_nonzero + 1e-12)
        e_silence = np.percentile(energy_db, 20)
        e_speech = np.percentile(energy_db, 55)
        e_threshold = e_silence + 0.45 * (e_speech - e_silence)
        e_threshold_lin = 10 ** (e_threshold / 20)
    else:
        e_threshold_lin = 0

    return segments, {"times": times_e, "energy": energy_e,
                      "is_speech": is_speech_display,
                      "energy_threshold": float(e_threshold_lin)}


def _refine_boundary(times, energy, coarse_t, direction='start', window_ms=40):
    """
    Refine a coarse VAD boundary by finding the energy onset/offset
    within a small window around the coarse time.

    For 'start': search backward from coarse_t to find where energy
    drops below 20% of the local max (the true onset).
    For 'end': search forward from coarse_t similarly.
    """
    if len(times) == 0:
        return coarse_t

    dt = times[1] - times[0] if len(times) > 1 else 0.005
    window_frames = max(1, int(window_ms / 1000 / dt))

    # Find the frame closest to coarse_t
    idx = np.searchsorted(times, coarse_t)
    idx = min(max(idx, 0), len(times) - 1)

    if direction == 'start':
        # Search backward: find where energy drops below threshold
        search_start = max(0, idx - window_frames)
        search_end = min(len(energy), idx + window_frames // 2)
        region = energy[search_start:search_end]
        if len(region) == 0:
            return coarse_t
        local_max = np.max(region)
        threshold = local_max * 0.15
        # Walk backward from idx to find onset
        for j in range(idx, search_start - 1, -1):
            if energy[j] < threshold:
                return float(times[min(j + 1, len(times) - 1)])
        return float(times[search_start])
    else:
        # Search forward: find where energy drops below threshold
        search_start = max(0, idx - window_frames // 2)
        search_end = min(len(energy), idx + window_frames)
        region = energy[search_start:search_end]
        if len(region) == 0:
            return coarse_t
        local_max = np.max(region)
        threshold = local_max * 0.15
        # Walk forward from idx to find offset
        for j in range(idx, search_end):
            if j < len(energy) and energy[j] < threshold:
                return float(times[max(j - 1, 0)])
        return float(times[min(search_end - 1, len(times) - 1)])


# ============================================================================
# 1d. SILERO VAD-BASED SEGMENTATION
# ============================================================================

def _get_silero_model():
    """Lazy-load the Silero VAD ONNX model (singleton)."""
    global _silero_model
    if _silero_model is None:
        _silero_model = load_silero_vad(onnx=True)
    return _silero_model


def detect_segments_silero(audio, sr):
    """
    Use Silero VAD (neural-network-based) for speech activity detection,
    with energy-envelope boundary refinement.

    Silero VAD is a trained deep-learning model that discriminates speech
    from non-speech (breaths, clicks, lip smacks, silence, noise) with
    much higher accuracy than WebRTC VAD or energy thresholding.

    It operates on the acoustic signal only — no transcription — so it
    works identically for real words and pseudowords.
    """
    if not HAS_SILERO:
        print("  \u26a0 silero-vad not installed \u2014 falling back")
        if HAS_WEBRTCVAD:
            return detect_segments_vad(audio, sr)
        return detect_segments_raw(audio, sr)

    duration = len(audio) / sr
    model = _get_silero_model()

    # Silero requires 16 kHz mono float32
    target_sr = 16000
    if sr != target_sr:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(sr), target_sr)
        audio_16k = resample_poly(audio, target_sr // g, int(sr) // g).astype(np.float32)
    else:
        audio_16k = audio.astype(np.float32)

    # Get speech timestamps from Silero (returns sample indices at 16kHz)
    speech_timestamps = get_speech_timestamps(
        audio_16k,
        model,
        sampling_rate=target_sr,
        threshold=0.35,                # speech probability threshold
        min_speech_duration_ms=MIN_SEGMENT_MS,
        min_silence_duration_ms=MIN_SILENCE_MS,
        speech_pad_ms=0,               # we handle padding ourselves
        return_seconds=False,          # get sample indices
    )

    if not speech_timestamps:
        print("  \u26a0 Silero found no speech \u2014 falling back to energy detection")
        return detect_segments_raw(audio, sr)

    # --- Energy envelope for boundary refinement and display ---
    times_e, energy_e = compute_energy_envelope(audio, sr)

    # --- Convert Silero segments to our format, refine boundaries ---
    segments = []
    for ts in speech_timestamps:
        # Convert 16kHz sample indices to seconds in the original audio
        coarse_start = ts['start'] / target_sr
        coarse_end = ts['end'] / target_sr

        # Refine using energy envelope for sub-frame precision
        start_sec = _refine_boundary(times_e, energy_e, coarse_start, direction='start')
        end_sec = _refine_boundary(times_e, energy_e, coarse_end, direction='end')

        # Apply silence margin
        start_sec = max(0, start_sec - SILENCE_MARGIN_MS / 1000)
        end_sec = min(duration, end_sec + SILENCE_MARGIN_MS / 1000)

        dur_ms = (end_sec - start_sec) * 1000
        if dur_ms >= MIN_SEGMENT_MS:
            segments.append({
                "start": round(start_sec, 4),
                "end": round(end_sec, 4),
                "duration_ms": round(dur_ms, 1),
            })

    # --- Build analysis dict for display ---
    # Create speech mask on the energy time grid from Silero timestamps
    is_speech_display = np.zeros(len(times_e), dtype=bool)
    for ts in speech_timestamps:
        seg_start_sec = ts['start'] / target_sr
        seg_end_sec = ts['end'] / target_sr
        mask = (times_e >= seg_start_sec) & (times_e <= seg_end_sec)
        is_speech_display[mask] = True

    energy_nonzero = energy_e[energy_e > 0]
    if len(energy_nonzero) > 0:
        energy_db = 20 * np.log10(energy_nonzero + 1e-12)
        e_silence = np.percentile(energy_db, 20)
        e_speech = np.percentile(energy_db, 55)
        e_threshold = e_silence + 0.45 * (e_speech - e_silence)
        e_threshold_lin = 10 ** (e_threshold / 20)
    else:
        e_threshold_lin = 0

    return segments, {"times": times_e, "energy": energy_e,
                      "is_speech": is_speech_display,
                      "energy_threshold": float(e_threshold_lin)}


# ============================================================================
# 2. CLASSIFY: filter intro/crosstalk, then label word clusters
# ============================================================================

def classify_and_label(segments, stimuli):
    """
    1. Detect intro block (everything before the first big gap).
    2. Flag long non-intro segments as crosstalk.
    3. Among word-length segments, cluster by gap size:
       - With N stimuli, speakers produce each word K times consecutively.
       - Gaps between repetitions of the SAME word are short.
       - Gaps when SWITCHING to the next word are longer.
       - Find the N-1 largest gaps among word segments → those are word boundaries.
       - Assign stimulus names to each cluster in order.
    """
    n_seg = len(segments)
    n_stim = len(stimuli) if stimuli else 0

    if n_seg == 0:
        return segments

    # ---- Detect INTRO block ----
    gaps = []
    for i in range(1, n_seg):
        gaps.append(segments[i]["start"] - segments[i - 1]["end"])

    intro_end_idx = 0
    if len(gaps) >= 2:
        scan_range = max(3, min(len(gaps), n_seg // 3))
        early_gaps = gaps[:scan_range]
        max_early_idx = int(np.argmax(early_gaps))
        max_early_gap = early_gaps[max_early_idx]
        later_gaps = gaps[scan_range:] if len(gaps) > scan_range else gaps
        median_later = np.median(later_gaps)
        if max_early_gap > median_later * 1.5 and max_early_gap > 0.6:
            intro_end_idx = max_early_idx + 1

    for i in range(intro_end_idx):
        segments[i]["segment_type"] = "intro"
        segments[i]["assigned_name"] = "intro"
        segments[i]["status"] = "intro"

    # ---- Estimate word duration, flag crosstalk ----
    non_intro = [(i, s) for i, s in enumerate(segments) if s.get("segment_type") != "intro"]
    if non_intro:
        durs = np.array([s["duration_ms"] for _, s in non_intro])
        p25, p75 = np.percentile(durs, 25), np.percentile(durs, 75)
        word_like = durs[(durs >= max(p25, WORD_DUR_MIN_MS)) & (durs <= p75)]
        median_dur = np.median(word_like) if len(word_like) > 0 else np.median(durs)
    else:
        median_dur = 500
    long_thresh = max(WORD_DUR_MAX_MS, median_dur * LONG_SEGMENT_FACTOR)

    for i, seg in non_intro:
        if seg["duration_ms"] > long_thresh:
            seg["segment_type"] = "crosstalk"
            seg["assigned_name"] = "crosstalk"
            seg["status"] = "crosstalk"

    # ---- Filter sub-word-length segments as noise ----
    for i, seg in non_intro:
        if seg.get("segment_type") not in ("intro", "crosstalk"):
            if seg["duration_ms"] < WORD_DUR_MIN_MS:
                seg["segment_type"] = "noise"
                seg["assigned_name"] = "noise"
                seg["status"] = "noise"

    # ---- Collect word candidates ----
    word_cands = [(i, s) for i, s in enumerate(segments)
                  if s.get("segment_type") not in ("intro", "crosstalk", "noise")]

    if not stimuli or not word_cands:
        for rank, (i, seg) in enumerate(word_cands):
            seg["segment_type"] = "word"
            seg["assigned_name"] = f"segment_{rank + 1:03d}"
            seg["status"] = "auto"
        _finalize_fields(segments)
        return segments

    # ---- Gap-based word clustering ----
    # Compute gaps between consecutive word candidates (skipping intro/crosstalk)
    wc_gaps = []
    for k in range(1, len(word_cands)):
        prev_i = word_cands[k - 1][0]
        curr_i = word_cands[k][0]
        gap = segments[curr_i]["start"] - segments[prev_i]["end"]
        wc_gaps.append((gap, k))

    # We need to split word_cands into n_stim clusters.
    # The (n_stim - 1) largest gaps are the word-switch boundaries.
    n_splits = n_stim - 1

    if n_splits > 0 and len(wc_gaps) >= n_splits:
        # Sort by gap size descending, take top n_splits
        sorted_gaps = sorted(wc_gaps, key=lambda g: g[0], reverse=True)
        split_indices = sorted([g[1] for g in sorted_gaps[:n_splits]])

        # Build clusters
        clusters = []
        prev = 0
        for si in split_indices:
            clusters.append(word_cands[prev:si])
            prev = si
        clusters.append(word_cands[prev:])
    elif n_splits == 0:
        clusters = [word_cands]
    else:
        # Fewer gaps than needed — just assign in order
        clusters = [[wc] for wc in word_cands]

    # Assign stimulus names to clusters
    for c_idx, cluster in enumerate(clusters):
        name = stimuli[c_idx] if c_idx < n_stim else f"extra_{c_idx + 1:03d}"
        for token_idx, (seg_i, seg) in enumerate(cluster):
            seg["segment_type"] = "word"
            seg["assigned_name"] = name
            seg["token_index"] = token_idx + 1
            seg["cluster_size"] = len(cluster)
            seg["status"] = "auto"

    # If there are more clusters than stimuli, extras are unlabeled
    if len(clusters) > n_stim:
        for c_idx in range(n_stim, len(clusters)):
            for _, seg in clusters[c_idx]:
                seg["assigned_name"] = f"unlabeled_{c_idx - n_stim + 1:03d}"
                seg["status"] = "auto_extra"

    _finalize_fields(segments)
    return segments


def _finalize_fields(segments):
    """Ensure all segments have required fields."""
    for seg in segments:
        seg.setdefault("segment_type", "word")
        seg.setdefault("assigned_name", "unknown")
        seg.setdefault("status", "unknown")
        seg.setdefault("token_index", 1)
        seg.setdefault("cluster_size", 1)
        seg.setdefault("accepted", True)


# ============================================================================
# 3. I/O HELPERS
# ============================================================================

def load_stimulus_list(path, condition=None):
    """Load a stimulus list from a .txt or .csv file.

    For .csv files the first column is treated as the word and the second
    column (if present) as the condition label.  When *condition* is given,
    only rows whose condition column matches (case-insensitive) are returned.
    If no rows match the condition, all words are returned as a fallback so
    the caller always gets a usable list.
    """
    path = Path(path)
    stimuli = []

    if path.suffix.lower() == '.csv':
        with open(path, newline='', encoding='utf-8') as f:
            reader = _csv.reader(f)
            rows = [r for r in reader if r]  # skip blank rows
        if not rows:
            return stimuli
        # Skip header row if first cell looks like a column name
        _HEADER_NAMES = {'word', 'words', 'stimulus', 'stimuli', 'stim', 'item', 'items'}
        data_rows = rows[1:] if rows[0][0].strip().lower() in _HEADER_NAMES else rows
        has_cond_col = any(len(r) >= 2 for r in data_rows)
        filtered = []
        for row in data_rows:
            word = row[0].strip()
            if not word or word.startswith('#'):
                continue
            if condition and has_cond_col and len(row) >= 2:
                if row[1].strip().lower() != condition.lower():
                    continue
            filtered.append(word)
        # Fall back to all words if condition filter matched nothing
        if condition and has_cond_col and not filtered:
            filtered = [r[0].strip() for r in data_rows if r and r[0].strip() and not r[0].startswith('#')]
        stimuli = filtered
    else:
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    stimuli.append(line)
    return stimuli


def find_stimulus_list(directory):
    directory = Path(directory)
    candidates = (list(directory.glob("stimuli.txt")) +
                  list(directory.glob("stim*.txt")) +
                  list(directory.glob("*.txt")))
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[0] if unique else None


def find_audio_file(directory):
    directory = Path(directory)
    for ext in [".wav", ".WAV", ".flac", ".mp3"]:
        files = list(directory.glob(f"*{ext}"))
        if files:
            return files[0]
    return None


def find_all_audio_files(directory):
    """Return all audio files in a directory, sorted by name."""
    directory = Path(directory)
    files = []
    for ext in [".wav", ".WAV", ".flac", ".mp3"]:
        files.extend(directory.glob(f"*{ext}"))
    # Deduplicate (e.g. .wav and .WAV on case-insensitive FS)
    seen = set()
    unique = []
    for f in sorted(files, key=lambda p: p.name.lower()):
        if f.resolve() not in seen:
            seen.add(f.resolve())
            unique.append(f)
    return unique


def concatenate_audio_files(audio_paths, output_path, silence_sec=0.5):
    """
    Concatenate multiple audio files into one WAV, inserting a brief
    silence between each recording so the segmenter can detect a gap.

    All files are resampled to match the sample rate of the first file.
    Returns the path to the concatenated file.
    """
    if not audio_paths:
        return None
    if len(audio_paths) == 1:
        return audio_paths[0]

    # Read the first file to establish sample rate
    first_audio, sr = sf.read(str(audio_paths[0]), dtype="float64")
    if first_audio.ndim > 1:
        first_audio = first_audio.mean(axis=1)

    silence = np.zeros(int(silence_sec * sr))
    parts = [first_audio]

    for path in audio_paths[1:]:
        audio, file_sr = sf.read(str(path), dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if file_sr != sr:
            # Resample to match the first file's rate
            import soxr
            audio = soxr.resample(audio, file_sr, sr)
        parts.append(silence)
        parts.append(audio)

    combined = np.concatenate(parts)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), combined.astype(np.float32), sr, subtype="FLOAT")
    print(f"  Concatenated {len(audio_paths)} recordings → {output_path.name} "
          f"({len(combined) / sr:.1f}s)")
    return output_path


# ============================================================================
# 4. TEXTGRID EXPORT
# ============================================================================

def export_textgrid(segments, duration, output_path):
    lines = [
        'File type = "ooTextFile"', 'Object class = "TextGrid"', '',
        f'xmin = 0', f'xmax = {duration}', 'tiers? <exists>', 'size = 1', 'item []:',
    ]
    intervals = []
    prev_end = 0.0
    for seg in segments:
        if seg["start"] > prev_end + 0.001:
            intervals.append((prev_end, seg["start"], ""))
        stype = seg.get("segment_type", "word")
        name = seg.get("assigned_name", "?")
        if stype in ("intro", "crosstalk"):
            label = f'[{stype}]'
        elif stype == "word":
            ti = seg.get("token_index", 1)
            label = f'{name}-{ti}'
        else:
            label = name
        intervals.append((seg["start"], seg["end"], label))
        prev_end = seg["end"]
    if prev_end < duration:
        intervals.append((prev_end, duration, ""))

    lines += [
        '    item [1]:', '        class = "IntervalTier"', '        name = "words"',
        f'        xmin = 0', f'        xmax = {duration}',
        f'        intervals: size = {len(intervals)}',
    ]
    for idx, (xmin, xmax, text) in enumerate(intervals, 1):
        lines += [f'        intervals [{idx}]:', f'            xmin = {xmin}',
                  f'            xmax = {xmax}', f'            text = "{text}"']

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ============================================================================
# 5. OVERVIEW PLOT
# ============================================================================

def plot_overview(audio, sr, segments, analysis, output_path):
    if not HAS_MATPLOTLIB:
        print("  (skipping plot — matplotlib not available)")
        return
    duration = len(audio) / sr
    fig, axes = plt.subplots(2, 1, figsize=(max(16, duration * 1.2), 7), sharex=True)
    t = np.arange(len(audio)) / sr
    type_colors = {"word": "#69b7a5", "intro": "#888888", "crosstalk": "#c0a050"}

    ax = axes[0]
    ax.plot(t, audio, color="0.4", linewidth=0.3)

    # Color clusters distinctly
    cluster_names = []
    for seg in segments:
        if seg["segment_type"] == "word" and seg["assigned_name"] not in cluster_names:
            cluster_names.append(seg["assigned_name"])

    cluster_palette = plt.cm.Set3(np.linspace(0, 1, max(len(cluster_names), 1)))
    cluster_color_map = {name: cluster_palette[i % len(cluster_palette)]
                         for i, name in enumerate(cluster_names)}

    for seg in segments:
        stype = seg.get("segment_type", "word")
        if stype == "word":
            color = cluster_color_map.get(seg["assigned_name"], "#69b7a5")
            alpha = 0.35
        else:
            color = type_colors.get(stype, "#888888")
            alpha = 0.2
        ax.axvspan(seg["start"], seg["end"], alpha=alpha, color=color)

        name = seg.get("assigned_name", "?")
        if stype == "word":
            name = f'{name}-{seg.get("token_index", "?")}'
        elif stype in ("intro", "crosstalk"):
            name = f'[{stype}]'
        mid = (seg["start"] + seg["end"]) / 2
        ylim = ax.get_ylim()
        ax.text(mid, ylim[1] * 0.88, name, ha="center", fontsize=5, rotation=45, va="bottom")

    word_count = sum(1 for s in segments if s["segment_type"] == "word")
    n_clusters = len(cluster_names)
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{len(segments)} segments: {word_count} tokens across {n_clusters} words, "
                 f"{sum(1 for s in segments if s['segment_type'] == 'intro')} intro, "
                 f"{sum(1 for s in segments if s['segment_type'] == 'crosstalk')} crosstalk")

    ax2 = axes[1]
    ax2.plot(analysis["times"], analysis["energy"], color="navy", linewidth=0.5)
    ax2.axhline(analysis["energy_threshold"], color="red", linestyle=":", alpha=0.5, label="Threshold")
    for seg in segments:
        stype = seg.get("segment_type", "word")
        color = type_colors.get(stype, "#69b7a5")
        if stype == "word":
            color = cluster_color_map.get(seg["assigned_name"], "#69b7a5")
        ax2.axvspan(seg["start"], seg["end"], alpha=0.15, color=color)
    ax2.set_ylabel("RMS Energy")
    ax2.set_xlabel("Time (s)")
    ax2.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# 6. EXPORT (with -1, -2, -3 suffixes)
# ============================================================================

def export_tokens(audio, sr, segments, output_dir, speaker_id=""):
    """Export all accepted word tokens with -N suffixes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pad = int(EXPORT_PAD_MS * sr / 1000)
    fade = int(FADE_MS * sr / 1000)
    prefix = re.sub(r'[^\w\-]', '_', speaker_id).strip('_') + '_' if speaker_id else ""

    # Count tokens per word to assign -1, -2, -3
    word_counter = defaultdict(int)
    exported = []

    for seg in segments:
        if seg.get("segment_type") not in ("word",):
            continue
        if not seg.get("accepted", True):
            continue

        name = seg.get("assigned_name", "unknown")
        safe_name = re.sub(r'[^\w\-.]', '_', name)
        word_counter[name] += 1
        idx = word_counter[name]

        start_sample = max(0, int(seg["start"] * sr) - pad)
        end_sample = min(len(audio), int(seg["end"] * sr) + pad)
        chunk = audio[start_sample:end_sample].copy()

        if fade > 0 and len(chunk) > 2 * fade:
            chunk[:fade] *= np.linspace(0, 1, fade)
            chunk[-fade:] *= np.linspace(1, 0, fade)

        fname = f"{prefix}{safe_name}-{idx}.wav"
        out_path = output_dir / fname
        sf.write(str(out_path), chunk.astype(np.float32), sr, subtype="FLOAT")

        exported.append({
            "word": name,
            "token_index": idx,
            "filename": fname,
            "start": seg["start"],
            "end": seg["end"],
            "duration_ms": round(len(chunk) / sr * 1000, 1),
        })

    # Save manifest
    manifest = {
        "speaker_id": speaker_id,
        "n_words": len(set(e["word"] for e in exported)),
        "n_tokens": len(exported),
        "tokens_per_word": {word: count for word, count in word_counter.items()},
        "tokens": exported,
    }
    with open(output_dir / "token_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return exported, manifest


# ============================================================================
# 7. FINALIZE (best token selection)
# ============================================================================

def finalize_tokens(tokens_dir, selections, output_dir, speaker_id=""):
    """
    Move best tokens to final/, rest to alternates/.

    selections: dict mapping word_name → filename of best token
    """
    tokens_dir = Path(tokens_dir)
    output_dir = Path(output_dir)
    final_dir = output_dir
    alt_dir = output_dir / "alternates"
    final_dir.mkdir(parents=True, exist_ok=True)
    alt_dir.mkdir(parents=True, exist_ok=True)

    prefix = re.sub(r'[^\w\-]', '_', speaker_id).strip('_') + '_' if speaker_id else ""

    # Load token manifest
    manifest_path = tokens_dir / "token_manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Build lookup: word → list of token filenames
    word_tokens = defaultdict(list)
    for tok in manifest["tokens"]:
        word_tokens[tok["word"]].append(tok["filename"])

    finalized = []
    for word, tokens in word_tokens.items():
        best_file = selections.get(word)
        safe_word = re.sub(r'[^\w\-.]', '_', word)

        for token_file in tokens:
            src = tokens_dir / token_file
            if not src.exists():
                continue

            if token_file == best_file:
                # Best → final directory, no suffix
                dest = final_dir / f"{prefix}{safe_word}.wav"
                shutil.copy2(src, dest)
                finalized.append({
                    "word": word, "source": token_file,
                    "destination": str(dest.relative_to(output_dir)),
                    "is_best": True,
                })
            else:
                # Alternate → alternates directory, keep suffix
                dest = alt_dir / token_file
                shutil.copy2(src, dest)
                finalized.append({
                    "word": word, "source": token_file,
                    "destination": str(dest.relative_to(output_dir)),
                    "is_best": False,
                })

    # Save finalization manifest
    with open(output_dir / "finalize_manifest.json", "w") as f:
        json.dump({"speaker_id": speaker_id, "selections": selections,
                    "files": finalized}, f, indent=2)

    return finalized


# ============================================================================
# 8. COMMANDS
# ============================================================================

def process_single(audio_path, stimlist_path, output_dir, audio_label=None,
                   condition=None, speaker_id=None, denoise=True, vad_method='silero'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = audio_label or Path(audio_path).stem

    print(f"\n  --- Processing: {label} ---")
    audio, sr = sf.read(str(audio_path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr
    print(f"  Audio: {duration:.1f}s at {sr} Hz")
    if speaker_id:
        print(f"  Speaker: {speaker_id}")
    if condition:
        print(f"  Condition: {condition}")

    # --- Noise reduction ---
    denoise_stats = {"applied": False}
    if denoise:
        print(f"  Applying noise reduction...")
        audio, denoise_stats = reduce_background_noise(audio, sr)
        if denoise_stats["applied"]:
            print(f"    Noise floor: {denoise_stats['noise_floor_rms']:.6f} RMS "
                  f"({denoise_stats['noise_frames_used']} silent frames)")
            print(f"    Overall change: {denoise_stats['reduction_db']:+.1f} dB")
            # Save denoised audio for the export step to use
            denoised_path = output_dir / "denoised.wav"
            sf.write(str(denoised_path), audio.astype(np.float32), sr, subtype="FLOAT")
            print(f"    Saved: {denoised_path.name}")

    stimuli = load_stimulus_list(stimlist_path, condition=condition) if stimlist_path else None
    if stimuli:
        print(f"  Stimulus list: {len(stimuli)} words")

    detect_fn = _pick_detect_fn(vad_method)
    detect_name = {detect_segments_silero: 'Silero VAD', detect_segments_vad: 'WebRTC VAD',
                   detect_segments_raw: 'energy threshold'}.get(detect_fn, 'unknown')
    print(f"  Detection method: {detect_name}")
    segments, analysis = detect_fn(audio, sr)
    print(f"  Raw segments: {len(segments)}")

    segments = classify_and_label(segments, stimuli)

    # Summary
    words = [s for s in segments if s["segment_type"] == "word"]
    intros = [s for s in segments if s["segment_type"] == "intro"]
    xtalks = [s for s in segments if s["segment_type"] == "crosstalk"]
    noise = [s for s in segments if s["segment_type"] == "noise"]
    unique_words = sorted(set(s["assigned_name"] for s in words))

    print(f"  → {len(words)} word tokens across {len(unique_words)} word types")
    if intros:
        print(f"  → {len(intros)} intro segments filtered")
    if xtalks:
        print(f"  → {len(xtalks)} crosstalk segments filtered")
    if noise:
        print(f"  → {len(noise)} noise/click segments filtered (<{WORD_DUR_MIN_MS}ms)")

    # Token counts per word
    for wname in unique_words:
        count = sum(1 for s in words if s["assigned_name"] == wname)
        print(f"    {wname}: {count} tokens")

    # Print segment table
    for i, seg in enumerate(segments):
        stype = seg["segment_type"]
        name = seg["assigned_name"]
        if stype == "intro":
            marker = "🎤"
        elif stype == "crosstalk":
            marker = "💬"
        elif stype == "noise":
            marker = "🔇"
        else:
            marker = f"  "
        ti = f"-{seg.get('token_index', '?')}" if stype == "word" else ""
        tag = f" [{stype}]" if stype != "word" else ""
        print(f"  {marker} {i + 1:3d}. {name}{ti}{tag}  "
              f"[{seg['start']:.2f}–{seg['end']:.2f}s] ({seg['duration_ms']:.0f}ms)")

    # Save
    denoised_path_str = str((output_dir / "denoised.wav").resolve()) if denoise_stats.get("applied") else None
    proposal = {
        "audio_file": str(Path(audio_path).resolve()),
        "denoised_audio_file": denoised_path_str,
        "audio_duration": duration,
        "sample_rate": sr,
        "speaker_id": speaker_id or "",
        "condition": condition or label,
        "stimulus_list": stimuli,
        "denoise": denoise_stats,
        "segments": segments,
    }

    json_path = output_dir / "proposed_segments.json"
    with open(json_path, "w") as f:
        json.dump(proposal, f, indent=2)
    export_textgrid(segments, duration, output_dir / "proposed_segments.TextGrid")
    print(f"  Saved: proposed_segments.json, TextGrid")
    return proposal


def _pick_detect_fn(vad_method):
    """Select the best available detection function based on preference."""
    if vad_method == 'silero' and HAS_SILERO:
        return detect_segments_silero
    if vad_method in ('silero', 'webrtc') and HAS_WEBRTCVAD:
        return detect_segments_vad
    return detect_segments_raw


def process_multiple(audio_paths, stimlist_path, output_dir, audio_label=None,
                     condition=None, speaker_id=None, denoise=True, silence_sec=0.5,
                     vad_method='silero'):
    """
    Segment each audio file independently, then merge the results into a
    single timeline for the review tool.

    Each file gets its own noise profile and gap-based word clustering;
    timestamps are offset so segments from file 2 follow file 1, etc.
    Token indices are renumbered globally per word across all files.
    A merged _combined.wav is saved as the audio reference for export.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    label = audio_label or Path(audio_paths[0]).stem

    stimuli = load_stimulus_list(stimlist_path, condition=condition) if stimlist_path else None
    if stimuli:
        print(f"  Stimulus list: {len(stimuli)} words")

    audio_parts = []
    all_segments = []
    all_analyses = []
    sr_ref = None
    time_offset = 0.0

    for file_idx, audio_path in enumerate(audio_paths):
        print(f"\n  --- File {file_idx + 1}/{len(audio_paths)}: {Path(audio_path).name} ---")
        audio, sr = sf.read(str(audio_path), dtype="float64")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        if sr_ref is None:
            sr_ref = sr
        elif sr != sr_ref:
            import soxr
            audio = soxr.resample(audio, sr, sr_ref)
            sr = sr_ref

        file_duration = len(audio) / sr
        print(f"  Audio: {file_duration:.1f}s at {sr} Hz")

        # Per-file noise reduction
        if denoise:
            print(f"  Applying noise reduction...")
            audio, denoise_stats = reduce_background_noise(audio, sr)
            if denoise_stats["applied"]:
                print(f"    Noise floor: {denoise_stats['noise_floor_rms']:.6f} RMS "
                      f"({denoise_stats['noise_frames_used']} silent frames)")
                print(f"    Overall change: {denoise_stats['reduction_db']:+.1f} dB")

        # Per-file segmentation and labeling
        detect_fn = _pick_detect_fn(vad_method)
        if file_idx == 0:
            detect_name = {detect_segments_silero: 'Silero VAD', detect_segments_vad: 'WebRTC VAD',
                           detect_segments_raw: 'energy threshold'}.get(detect_fn, 'unknown')
            print(f"  Detection method: {detect_name}")
        segments, analysis = detect_fn(audio, sr)
        print(f"  Raw segments: {len(segments)}")
        segments = classify_and_label(segments, stimuli)

        # Shift timestamps and tag source file
        for seg in segments:
            seg["start"] = round(seg["start"] + time_offset, 4)
            seg["end"] = round(seg["end"] + time_offset, 4)
            seg["source_file"] = Path(audio_path).name

        # Shift analysis times for merged overview plot
        analysis["times"] = analysis["times"] + time_offset

        audio_parts.append(audio)
        all_segments.extend(segments)
        all_analyses.append(analysis)

        time_offset += file_duration + silence_sec

    # Build combined audio (silence padding between files)
    silence = np.zeros(int(silence_sec * sr_ref))
    combined_parts = []
    for i, part in enumerate(audio_parts):
        combined_parts.append(part)
        if i < len(audio_parts) - 1:
            combined_parts.append(silence)
    audio_combined = np.concatenate(combined_parts)
    total_duration = len(audio_combined) / sr_ref

    # Renumber token_index globally per word across all files
    word_counter = defaultdict(int)
    for seg in all_segments:
        if seg.get("segment_type") == "word":
            word_counter[seg["assigned_name"]] += 1
            seg["token_index"] = word_counter[seg["assigned_name"]]
    for seg in all_segments:
        if seg.get("segment_type") == "word":
            seg["cluster_size"] = word_counter[seg["assigned_name"]]

    # Merge analysis arrays for the overview plot
    merged_analysis = {
        "times":            np.concatenate([a["times"]    for a in all_analyses]),
        "energy":           np.concatenate([a["energy"]   for a in all_analyses]),
        "is_speech":        np.concatenate([a["is_speech"] for a in all_analyses]),
        "energy_threshold": float(np.mean([a["energy_threshold"] for a in all_analyses])),
    }

    # Summary
    words = [s for s in all_segments if s["segment_type"] == "word"]
    unique_words = sorted(set(s["assigned_name"] for s in words))
    intros  = [s for s in all_segments if s["segment_type"] == "intro"]
    xtalks  = [s for s in all_segments if s["segment_type"] == "crosstalk"]
    noise   = [s for s in all_segments if s["segment_type"] == "noise"]
    print(f"\n  MERGED: {len(words)} word tokens across {len(unique_words)} word types")
    if intros:
        print(f"  → {len(intros)} intro segments filtered")
    if xtalks:
        print(f"  → {len(xtalks)} crosstalk segments filtered")
    if noise:
        print(f"  → {len(noise)} noise/click segments filtered (<{WORD_DUR_MIN_MS}ms)")
    for wname in unique_words:
        count = sum(1 for s in words if s["assigned_name"] == wname)
        print(f"    {wname}: {count} tokens")

    # Save combined audio as the reference for the export step
    combined_path = output_dir / "_combined.wav"
    sf.write(str(combined_path), audio_combined.astype(np.float32), sr_ref, subtype="FLOAT")
    print(f"  Saved combined audio: {combined_path.name} ({total_duration:.1f}s)")

    proposal = {
        "audio_file":         str(combined_path.resolve()),
        "denoised_audio_file": None,
        "audio_duration":     total_duration,
        "sample_rate":        sr_ref,
        "speaker_id":         speaker_id or "",
        "condition":          condition or label,
        "stimulus_list":      stimuli,
        "source_files":       [str(Path(p).resolve()) for p in audio_paths],
        "segments":           all_segments,
    }

    json_path = output_dir / "proposed_segments.json"
    with open(json_path, "w") as f:
        json.dump(proposal, f, indent=2)
    export_textgrid(all_segments, total_duration, output_dir / "proposed_segments.TextGrid")
    print(f"  Saved: proposed_segments.json, TextGrid")
    return proposal


def cmd_detect(args):
    print("=" * 60)
    print("RECORDING SESSION SEGMENTER v3.1 — Detect")
    print("=" * 60)
    process_single(args.audio, args.stimlist, args.output_dir,
                   condition=getattr(args, 'condition', None),
                   speaker_id=getattr(args, 'speaker_id', None),
                   denoise=not getattr(args, 'no_denoise', False))
    print(f"\n  NEXT: Open review_tool.html → Review mode")


def cmd_batch(args):
    speaker_dir = Path(args.speaker_dir)
    output_dir = Path(args.output_dir)
    speaker_id = getattr(args, 'speaker_id', None) or ""
    denoise = not getattr(args, 'no_denoise', False)

    print("=" * 60)
    print("RECORDING SESSION SEGMENTER v3.1 — Batch")
    print("=" * 60)
    print(f"\n  Speaker: {speaker_dir}")
    if speaker_id:
        print(f"  Speaker ID: {speaker_id}")
    if denoise:
        print(f"  Noise reduction: ON")

    subdirs = sorted([d for d in speaker_dir.iterdir() if d.is_dir()])
    results = {}
    for d in subdirs:
        audio_files = find_all_audio_files(d)
        if not audio_files:
            continue
        stim = find_stimulus_list(d)
        out = output_dir / d.name
        if len(audio_files) > 1:
            print(f"\n  Found {len(audio_files)} recordings in {d.name}/ — segmenting each then merging:")
            for af in audio_files:
                print(f"    {af.name}")
            p = process_multiple(audio_files, stim, out, audio_label=d.name,
                                 condition=d.name, speaker_id=speaker_id,
                                 denoise=denoise)
        else:
            p = process_single(audio_files[0], stim, out, audio_label=d.name,
                               condition=d.name, speaker_id=speaker_id,
                               denoise=denoise)
        words = [s for s in p["segments"] if s["segment_type"] == "word"]
        results[d.name] = len(words)

    review_src = Path(__file__).parent / "review_tool.html"
    if review_src.exists():
        shutil.copy2(review_src, output_dir / "review_tool.html")

    print(f"\n{'=' * 60}")
    print(f"BATCH SUMMARY")
    for name, count in results.items():
        print(f"  {name}: {count} tokens")
    print(f"\n  NEXT: Open review_tool.html for each condition")
    print(f"{'=' * 60}")


def cmd_export(args):
    print("=" * 60)
    print("EXPORT TOKENS")
    print("=" * 60)

    with open(args.segments) as f:
        data = json.load(f)

    # Prefer denoised audio if it exists
    audio_path = args.audio
    denoised_path = data.get("denoised_audio_file")
    if denoised_path and Path(denoised_path).exists():
        audio_path = denoised_path
        print(f"\n  Using denoised audio: {Path(denoised_path).name}")
    else:
        print(f"\n  Using original audio: {args.audio}")

    audio, sr = sf.read(str(audio_path), dtype="float64")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    with open(args.segments) as f:
        data = json.load(f)

    speaker_id = getattr(args, 'speaker_id', None) or data.get("speaker_id", "")
    segments = data["segments"]

    exported, manifest = export_tokens(audio, sr, segments, args.output_dir, speaker_id)

    print(f"\n  Exported {len(exported)} tokens:")
    for word, count in manifest["tokens_per_word"].items():
        print(f"    {word}: {count} tokens")
    print(f"\n  NEXT: Open review_tool.html → Select Best mode")


def cmd_finalize(args):
    print("=" * 60)
    print("FINALIZE — Select best tokens")
    print("=" * 60)

    with open(args.selections) as f:
        sel_data = json.load(f)

    selections = sel_data.get("selections", {})
    speaker_id = getattr(args, 'speaker_id', None) or sel_data.get("speaker_id", "")

    finalized = finalize_tokens(args.tokens_dir, selections, args.output_dir, speaker_id)

    best = [f for f in finalized if f["is_best"]]
    alt = [f for f in finalized if not f["is_best"]]
    print(f"\n  Best tokens ({len(best)}):")
    for f in best:
        print(f"    {f['destination']}")
    print(f"\n  Alternates ({len(alt)}) → alternates/")
    print(f"\n  ✓ Done! {len(best)} final stimulus files ready.")


# ============================================================================
# 9. CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Recording session segmenter v3.1 — multi-token workflow with multi-file concatenation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("detect")
    p.add_argument("--audio", required=True)
    p.add_argument("--stimlist", default=None)
    p.add_argument("--output_dir", default="segments")
    p.add_argument("--condition", default=None)
    p.add_argument("--speaker_id", default=None)
    p.add_argument("--no-denoise", action="store_true",
                   help="Skip background noise reduction")

    p = sub.add_parser("batch")
    p.add_argument("--speaker_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--speaker_id", default=None)
    p.add_argument("--no-denoise", action="store_true",
                   help="Skip background noise reduction")

    p = sub.add_parser("export")
    p.add_argument("--audio", required=True)
    p.add_argument("--segments", required=True)
    p.add_argument("--output_dir", default="segments/tokens")
    p.add_argument("--speaker_id", default=None)

    p = sub.add_parser("finalize")
    p.add_argument("--tokens_dir", required=True)
    p.add_argument("--selections", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--speaker_id", default=None)

    args = parser.parse_args()
    cmds = {"detect": cmd_detect, "batch": cmd_batch,
            "export": cmd_export, "finalize": cmd_finalize}
    if args.command in cmds:
        cmds[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
