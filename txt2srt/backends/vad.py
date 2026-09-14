"""Proportional alignment over detected speech (--backend vad).

No ASR: speech regions come from the VAD and the transcript is poured into them
in proportion to how long each character takes to say. Cue boundaries land on
real speech onsets, but which words are in which region is an assumption, so
this is a draft to review, not a result. It exists for the cases where the ASR
backends cannot run: no Whisper weights, an unsupported language, or a machine
without Metal.
"""
from __future__ import annotations

import numpy as np

from ..audio import SAMPLE_RATE
from ..transcript import Doc, alignable, is_japanese
from . import Times


def _weight(ch: str) -> float:
    """Rough time cost of a character, relative to one Japanese mora."""
    if not alignable(ch):
        return 0.12
    return 1.0 if is_japanese(ch) else 0.42


def align(audio: np.ndarray, doc: Doc, *, speech, verbose: bool = False, **_) -> Times:
    regions = speech.regions() or [(0.0, len(audio) / SAMPLE_RATE)]
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
