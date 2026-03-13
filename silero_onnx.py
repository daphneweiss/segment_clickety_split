"""
Torch-free Silero VAD wrapper using onnxruntime + numpy only.

Provides the same interface used by segment_recording.py:
    load_silero_vad(onnx=True)  → model
    get_speech_timestamps(audio_np, model, sampling_rate=...)  → list of {start, end}

The ONNX model file (silero_vad.onnx) must be in the same directory as this
module, or next to the executable when running as a PyInstaller bundle.
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import numpy as np
import onnxruntime


def _model_path() -> str:
    """Locate silero_vad.onnx — handles both source tree and PyInstaller bundle."""
    # PyInstaller bundles data files next to the executable via sys._MEIPASS
    if hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / "silero_vad.onnx"
        if candidate.is_file():
            return str(candidate)
    # Source tree — model sits alongside this module
    here = Path(__file__).parent / "silero_vad.onnx"
    if here.is_file():
        return str(here)
    raise FileNotFoundError(
        "silero_vad.onnx not found. Expected alongside silero_onnx.py or in sys._MEIPASS."
    )


class SileroOnnxModel:
    """Minimal stateful wrapper around the silero VAD ONNX model."""

    CHUNK_SIZES = {8000: 256, 16000: 512}  # sr → samples per inference call

    def __init__(self, model_path: str):
        opts = onnxruntime.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.session = onnxruntime.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
            sess_options=opts,
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def reset_states(self):
        self._state = np.zeros((2, 1, 128), dtype=np.float32)

    def __call__(self, chunk: np.ndarray, sr: int) -> float:
        """
        Run one inference step.

        chunk: 1-D float32 numpy array of exactly CHUNK_SIZES[sr] samples.
        Returns speech probability in [0, 1].
        """
        x = chunk.astype(np.float32).reshape(1, -1)
        out, state_n = self.session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(sr, dtype=np.int64),
            },
        )
        self._state = state_n
        return float(out[0, 0])


def load_silero_vad(onnx: bool = True):
    """Drop-in replacement for silero_vad.load_silero_vad."""
    return SileroOnnxModel(_model_path())


def get_speech_timestamps(
    audio: np.ndarray,
    model: SileroOnnxModel,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    speech_pad_ms: int = 30,
    return_seconds: bool = False,
    neg_threshold: float | None = None,
    **_kwargs,
) -> list[dict]:
    """
    Drop-in replacement for silero_vad.get_speech_timestamps.
    Returns list of {start, end} dicts in samples (or seconds if return_seconds=True).
    """
    if neg_threshold is None:
        neg_threshold = threshold - 0.15

    # Resample from multiples of 16 kHz down to 16 kHz
    sr = sampling_rate
    if sr > 16000 and sr % 16000 == 0:
        step = sr // 16000
        audio = audio[::step]
        sr = 16000
        warnings.warn("Sampling rate is a multiple of 16000; downsampled to 16000.")

    if sr not in (8000, 16000):
        raise ValueError(f"silero_onnx supports 8000 and 16000 Hz (got {sr})")

    chunk_size = SileroOnnxModel.CHUNK_SIZES[sr]
    min_speech_samples = sr * min_speech_duration_ms // 1000
    min_silence_samples = sr * min_silence_duration_ms // 1000
    speech_pad_samples = sr * speech_pad_ms // 1000

    model.reset_states()

    # Run inference over all chunks
    audio = audio.astype(np.float32)
    n = len(audio)
    probs = []
    for i in range(0, n, chunk_size):
        chunk = audio[i : i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
        probs.append(model(chunk, sr))

    # Hysteresis detection: triggered → speech, untriggered → silence
    speeches: list[dict] = []
    triggered = False
    speech_start = 0
    silence_start = 0

    for idx, prob in enumerate(probs):
        sample_pos = idx * chunk_size

        if prob >= threshold and not triggered:
            triggered = True
            speech_start = max(0, sample_pos - speech_pad_samples)
            silence_start = 0
        elif prob < neg_threshold and triggered:
            if silence_start == 0:
                silence_start = sample_pos
            if (sample_pos - silence_start) >= min_silence_samples:
                speech_end = silence_start + speech_pad_samples
                dur = speech_end - speech_start
                if dur >= min_speech_samples:
                    speeches.append({"start": speech_start, "end": min(speech_end, n)})
                triggered = False
                silence_start = 0
        elif prob >= threshold and triggered:
            silence_start = 0  # reset silence counter on resumed speech

    # Close any open speech segment at end of audio
    if triggered:
        speech_end = min(n, len(probs) * chunk_size + speech_pad_samples)
        dur = speech_end - speech_start
        if dur >= min_speech_samples:
            speeches.append({"start": speech_start, "end": speech_end})

    if return_seconds:
        speeches = [
            {"start": s["start"] / sr, "end": s["end"] / sr} for s in speeches
        ]

    return speeches
