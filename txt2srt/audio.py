"""Audio decoding to 16 kHz mono float32.

Two paths, and which one runs is decided by what is installed. PyAV keeps the
decode in process, with ffmpeg's libav* linked into the wheel. Homebrew installs
get the ffmpeg binary instead: the dylibs PyAV bundles are ad-hoc signed, and
Homebrew rewrites Mach-O install names in everything it installs, which breaks
those signatures and makes macOS kill the process on import.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np

SAMPLE_RATE = 16000


def load(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    try:
        import av  # noqa: F401
    except ImportError:
        return _load_ffmpeg(path, sr)
    return _load_av(path, sr)


def _load_ffmpeg(path: str, sr: int) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("neither PyAV nor ffmpeg is available to decode audio")
    out = subprocess.run(
        [exe, "-nostdin", "-threads", "0", "-i", path, "-f", "f32le",
         "-ac", "1", "-ar", str(sr), "-"],
        capture_output=True, check=True).stdout
    audio = np.frombuffer(out, dtype=np.float32)
    if audio.size == 0:
        raise ValueError(f"decoded no audio from {path}")
    return audio.copy()


def _load_av(path: str, sr: int) -> np.ndarray:
    import av

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise ValueError(f"no audio stream in {path}")
        resampler = av.audio.resampler.AudioResampler(format="flt", layout="mono", rate=sr)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):        # flush
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise ValueError(f"decoded no audio from {path}")
    return np.concatenate(chunks).astype(np.float32)


def duration(path: str) -> float:
    import av

    with av.open(path) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is not None and stream.duration and stream.time_base:
            return float(stream.duration * stream.time_base)
        return float(container.duration or 0) / 1_000_000
