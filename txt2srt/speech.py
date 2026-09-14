"""Where someone is actually speaking.

Every part of this tool that reasons about silence used to ask the energy
envelope: frames louder than the tenth percentile plus a margin were "speech".
On a studio recording that is close enough. On the location recordings this
tool is for, it is not. Measured over an hour of interview, the energy gate
found 8425 speech runs with a median length of 0.16s, which is not speech, it
is an amplitude envelope flickering on syllables and room tone. Silero finds
1182 segments with a median of 1.60s, which is what an hour of conversation
looks like.

That gap was not academic. It decided where cue boundaries were allowed to
move, what `--report` graded against, and which words the aligner believed had
run into a pause.

Silero VAD, MIT, 2.3 MB of ONNX weights carried in the package so that nothing
here needs the network or a 2 GB torch install (see data/silero_vad.LICENSE).
It runs at about 11s per hour of audio on one CPU thread.

The mask is resampled onto the same 20 ms grid the rest of the tool counts in,
so callers keep their arithmetic and only their evidence improves.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .audio import SAMPLE_RATE

MODEL = Path(__file__).parent / "data" / "silero_vad.onnx"

FRAME = 512                  # samples the model scores at a time, 32 ms at 16 kHz
CONTEXT = 64                 # samples of the previous frame it also wants to see
HOP = 0.02                   # the grid everything else in the tool counts in

# Speech starts at `ON` and only stops below `OFF`. One threshold would chatter
# through every unvoiced consonant in the middle of a word.
ON = 0.5
OFF = 0.35
MIN_SPEECH_S = 0.10
MIN_SILENCE_S = 0.10


@dataclass
class Speech:
    """A speech/not-speech decision per 20 ms of the recording."""
    mask: np.ndarray                      # (N,) bool
    onsets: np.ndarray                    # (K,) seconds, where speech begins
    offsets: np.ndarray                   # (K,) seconds, where it stops
    cum: np.ndarray                       # (N+1,) cumulative seconds of speech

    def at(self, t: float) -> bool:
        i = int(min(max(t / HOP, 0), len(self.mask) - 1))
        return bool(self.mask[i]) if self.mask.size else False

    def any_between(self, t0: float, t1: float) -> bool:
        i, j = self._span(t0, t1)
        return bool(self.mask[i:j].any())

    def seconds_between(self, t0: float, t1: float) -> float:
        i, j = self._span(t0, t1)
        return float(self.cum[j] - self.cum[i])

    def fraction_between(self, t0: float, t1: float) -> float:
        i, j = self._span(t0, t1)
        return float(self.mask[i:j].mean()) if j > i else 0.0

    def last_onset_before(self, t: float) -> float | None:
        """Where the speech that is going on at `t` began, or the most recent
        speech to have begun before it."""
        k = int(np.searchsorted(self.onsets, t, side="right")) - 1
        return float(self.onsets[k]) if k >= 0 else None

    def regions(self, min_gap: float = 0.35, min_len: float = 0.15) -> list[tuple[float, float]]:
        """Speech segments, with anything closer than `min_gap` merged."""
        out: list[tuple[float, float]] = []
        for s, e in zip(self.onsets, self.offsets):
            if out and s - out[-1][1] < min_gap:
                out[-1] = (out[-1][0], float(e))
            else:
                out.append((float(s), float(e)))
        return [(s, e) for s, e in out if e - s >= min_len]

    def _span(self, t0: float, t1: float) -> tuple[int, int]:
        n = len(self.mask)
        i = int(min(max(t0 / HOP, 0), n))
        j = int(min(max(t1 / HOP, 0), n)) + 1
        return i, min(max(j, i), n)


def detect(audio: np.ndarray, *, on: float = ON, off: float = OFF,
           min_speech: float = MIN_SPEECH_S, min_silence: float = MIN_SILENCE_S) -> Speech:
    probs = _probs(audio)
    frames = _hysteresis(probs, on, off)
    step = FRAME / SAMPLE_RATE
    _close_gaps(frames, min_silence / step)
    _drop_blips(frames, min_speech / step)
    n = int(len(audio) / SAMPLE_RATE / HOP) + 1
    grid = (np.arange(n) * HOP / step).astype(np.int64)
    mask = frames[np.clip(grid, 0, max(len(frames) - 1, 0))] if frames.size else np.zeros(n, bool)
    edges = np.diff(mask.astype(np.int8))
    return Speech(mask=mask,
                  onsets=np.flatnonzero(edges == 1) * HOP + HOP,
                  offsets=np.flatnonzero(edges == -1) * HOP + HOP,
                  cum=np.concatenate([[0.0], np.cumsum(mask) * HOP]))


def _probs(audio: np.ndarray) -> np.ndarray:
    try:
        import onnxruntime as ort
    except ImportError as e:                      # pragma: no cover
        raise RuntimeError(
            "onnxruntime is needed to detect speech. Install txt2srt's dependencies, "
            "or `pip install onnxruntime`.") from e

    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    net = ort.InferenceSession(str(MODEL), opts, providers=["CPUExecutionProvider"])
    sr = np.array(SAMPLE_RATE, dtype=np.int64)
    state = np.zeros((2, 1, 128), dtype=np.float32)
    window = np.zeros((1, CONTEXT + FRAME), dtype=np.float32)
    n = len(audio) // FRAME
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        chunk = audio[i * FRAME:(i + 1) * FRAME]
        window[0, CONTEXT:] = chunk
        p, state = net.run(None, {"input": window, "state": state, "sr": sr})
        out[i] = p[0, 0]
        window[0, :CONTEXT] = chunk[-CONTEXT:]
    return out


def _hysteresis(probs: np.ndarray, on: float, off: float) -> np.ndarray:
    live = False
    out = np.zeros(len(probs), dtype=bool)
    for i, v in enumerate(probs):
        live = v >= on if not live else v >= off
        out[i] = live
    return out


def _runs(mask: np.ndarray, value: bool):
    d = np.diff(np.concatenate([[0], (mask == value).view(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _close_gaps(mask: np.ndarray, min_frames: float) -> None:
    for a, b in _runs(mask, False):
        if b - a < min_frames and a > 0 and b < len(mask):
            mask[a:b] = True


def _drop_blips(mask: np.ndarray, min_frames: float) -> None:
    for a, b in _runs(mask, True):
        if b - a < min_frames:
            mask[a:b] = False
