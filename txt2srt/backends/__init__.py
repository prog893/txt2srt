"""Alignment backends.

A backend's whole job: given audio and a Doc, return a start and end time for
as many stream positions as it can. It never returns text.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..transcript import Doc

BACKENDS = ("whisper", "vad")

# A word may keep this much silence in front of it once the VAD has said where
# the speech it belongs to began, and may be credited with no slower delivery
# than this, in weighted characters per second.
MAX_LEAD_S = 1.0
MIN_RATE = 4.0


@dataclass
class Times:
    """Per-character times over Doc.stream. NaN means "this backend had nothing to say"."""
    start: np.ndarray            # (N,) float64
    end: np.ndarray              # (N,)
    prob: np.ndarray             # (N,) float32, backend-defined confidence
    backend: str
    model: str = ""
    clamped: int = 0             # units whose start clamp_spans had to pull in

    @classmethod
    def empty(cls, n: int, backend: str, model: str = "") -> "Times":
        nan = np.full(n, np.nan)
        return cls(nan.copy(), nan.copy(), np.zeros(n, dtype=np.float32), backend, model)

    def set(self, i0: int, i1: int, start: float, end: float, prob: float = 0.0) -> None:
        self.start[i0:i1] = start
        self.end[i0:i1] = end
        self.prob[i0:i1] = prob

    @property
    def timed(self) -> int:
        return int((~np.isnan(self.start)).sum())

    def fill_gaps(self) -> None:
        """Give silent characters (punctuation, spaces, dropped glyphs) a position.

        Interpolated between the neighbours that do have times, so a cue that
        begins with an opening bracket still starts when its first spoken
        character does.
        """
        n = len(self.start)
        known = np.flatnonzero(~np.isnan(self.start))
        if known.size == 0:
            raise RuntimeError("no character received a timestamp")
        first, last = known[0], known[-1]
        self.start[:first] = self.start[first]
        self.end[:first] = self.start[first]
        self.start[last + 1:] = self.end[last]
        self.end[last + 1:] = self.end[last]
        for a, b in zip(known[:-1], known[1:]):
            if b == a + 1:
                continue
            lo, hi = self.end[a], self.start[b]
            span = max(hi - lo, 0.0)
            gap = b - a
            for k in range(1, gap):
                t = lo + span * (k - 0.5) / gap
                self.start[a + k] = t
                self.end[a + k] = t
        # Monotonicity: DTW-based backends can emit a word that starts before
        # its predecessor ended. Cues are built from min/max over a range, so a
        # single inversion would otherwise widen a cue across a neighbour.
        np.maximum.accumulate(self.start, out=self.start)
        np.maximum.accumulate(self.end, out=self.end)
        np.minimum(self.start, self.end, out=self.start)
        assert not np.isnan(self.start).any()


def clamp_spans(times: Times, doc: Doc, speech, *,
                max_lead: float = MAX_LEAD_S, min_rate: float = MIN_RATE) -> int:
    """Take back the silence a word was charged for but never occupied.

    DTW has to account for every frame of its window, so a pause between two
    words is charged to the word after it: that word ends where it was really
    said and starts wherever the silence began. In an hour of interview, 335 of
    10866 timed units come back holding a pause, the worst of them 24.5s of it,
    and each one drags a cue onto the screen that long early. Cue boundaries are
    the minimum start over a range, so the bad time always wins.

    Nothing else can undo it. --snap-window moves a boundary by fractions of a
    second and this error is measured in tens of them, and splitting a
    four-character line only produces two cues that are both still wrong.

    So a word is moved up to wherever the VAD says the speech it ends inside
    began. That bound alone is not enough: a transcript is an account of what
    was said, not of every sound in the room, and an untranscribed interjection
    or an overlapping second speaker leaves real speech that no word belongs
    to. Measured on the same hour, the pauses these units were holding are 43%
    speech by the VAD, so an onset can sit ten seconds before a word that takes
    a tenth of one. `max_lead` and `min_rate` are the backstop for that case,
    and the tighter of the two bounds wins.

    The end is never touched: the end is where the word was actually spoken.
    """
    from ..cues import text_weight

    start, end = times.start, times.end
    n, fixed, i = len(start), 0, 0
    while i < n:
        if np.isnan(start[i]):
            i += 1
            continue
        j = i + 1                       # characters a backend timed as one unit
        while j < n and start[j] == start[i] and end[j] == end[i]:
            j += 1
        allowed = max_lead + text_weight(doc.stream[i:j]) / min_rate
        if end[i] - start[i] > allowed:
            floor = end[i] - allowed
            onset = speech.last_onset_before(end[i] - 0.05)
            new = floor if onset is None else min(max(onset, floor), end[i] - 0.05)
            start[i:j] = max(new, start[i])
            fixed += 1
        i = j
    return fixed


def align(backend: str, audio: np.ndarray, doc: Doc, *, speech, **opts) -> Times:
    if backend == "whisper":
        from .whisper import align as fn
    elif backend == "vad":
        from .vad import align as fn
    else:
        raise SystemExit(f"unknown backend {backend!r}; choose from {', '.join(BACKENDS)}")
    times = fn(audio, doc, speech=speech, **opts)
    times.clamped = clamp_spans(times, doc=doc, speech=speech)
    times.fill_gaps()
    return times
