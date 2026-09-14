"""Energy-VAD proportional alignment (--backend vad).

No model, no downloads, no licence question: speech regions are found by energy
and the transcript is poured into them in proportion to how long each character
takes to say. Cue boundaries land on real speech onsets, but which words are in
which region is an assumption, so this is a draft to review, not a result. It
exists for the cases where the model backends cannot run: no network, an
unsupported language, or a machine without Metal.
"""
from __future__ import annotations

import numpy as np

from ..audio import SAMPLE_RATE, envelope, speech_mask
from ..transcript import Doc, alignable, is_japanese
from . import Times

HOP = 320                    # 20 ms


def _weight(ch: str) -> float:
    """Rough time cost of a character, relative to one Japanese mora."""
    if not alignable(ch):
        return 0.12
    return 1.0 if is_japanese(ch) else 0.42


def _regions(audio: np.ndarray, min_gap_s: float = 0.35, min_len_s: float = 0.15):
    db = envelope(audio, HOP)
    mask = speech_mask(db)
    if not mask.any():
        return [(0.0, len(audio) / SAMPLE_RATE)]
    idx = np.flatnonzero(mask)
    breaks = np.flatnonzero(np.diff(idx) > int(min_gap_s / 0.02))
    starts = np.concatenate([[idx[0]], idx[breaks + 1]])
    ends = np.concatenate([idx[breaks], [idx[-1]]])
    out = [(s * 0.02, (e + 1) * 0.02) for s, e in zip(starts, ends)
           if (e - s + 1) * 0.02 >= min_len_s]
    return out or [(0.0, len(audio) / SAMPLE_RATE)]


def align(audio: np.ndarray, doc: Doc, *, verbose: bool = False, **_) -> Times:
    regions = _regions(audio)
    total_speech = sum(e - s for s, e in regions)
    weights = np.array([_weight(c) for c in doc.stream], dtype=np.float64)
    total_weight = float(weights.sum()) or 1.0
    times = Times.empty(len(doc.stream), "vad", "energy")

    ri, cursor = 0, regions[0][0]
    for i, w in enumerate(weights):
        need = (w / total_weight) * total_speech
        while ri < len(regions) - 1 and cursor + need > regions[ri][1]:
            ri += 1
            cursor = regions[ri][0]
        times.set(i, i + 1, cursor, min(cursor + need, regions[ri][1]), 0.0)
        cursor += need
    if verbose:
        print(f"  {len(regions)} speech regions, {total_speech:.0f}s of speech", flush=True)
    return times
