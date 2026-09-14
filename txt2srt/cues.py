"""Turning timed characters into subtitle cues.

Cue text is always a slice of the source line, so splitting can move a break
but can never rewrite a word. Where to break is scored by how long a character
takes to say rather than by counting characters, because a Japanese line and an
English line of the same length are not the same amount of speech.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transcript import Doc, Line, is_japanese
from .backends import Times

SENT_END = "。？！?!"
SOFT_BREAK = "、，,；;：:"
CLOSERS = "」』）)]】〉》\"'”’"
OPENERS = "「『（([【〈《\"'“‘"


@dataclass
class Cue:
    index: int
    start: float
    end: float
    text: str
    speaker: str | None
    line: int
    a: int                      # stream offsets, for the CSV map and for JSON
    b: int
    score: float


def weight(ch: str) -> float:
    if is_japanese(ch):
        return 1.0
    if ch.isspace():
        return 0.15
    if ch in SENT_END:
        return 0.25
    if ch in SOFT_BREAK:
        return 0.18
    if ch.isascii() and ch.isalnum():
        return 0.42
    return 0.35


def text_weight(s: str) -> float:
    return sum(weight(c) for c in s)


def _split_once(s: str, start: int, target: float, hard: float) -> int:
    """Index to break at, preferring sentence end, then soft punctuation, then a space."""
    n = len(s)
    strong: list[tuple[float, int]] = []
    soft: list[tuple[float, int]] = []
    spaces: list[tuple[float, int]] = []
    run = 0.0
    i = start
    end = start + 1
    while i < n:
        run += weight(s[i])
        end = i + 1
        while end < n and s[end] in CLOSERS:     # never orphan a closing bracket
            run += weight(s[end])
            end += 1
            i = end - 1
        if s[i] in SENT_END:
            strong.append((run, end))
        elif s[i] in SOFT_BREAK:
            soft.append((run, end))
        elif s[i].isspace():
            spaces.append((run, end))
        if run >= hard:
            break
        i += 1
    if end >= n:
        return n

    def best(cands: list[tuple[float, int]], floor_ratio: float) -> int | None:
        ok = [(w, idx) for w, idx in cands if target * floor_ratio <= w <= hard * 1.1]
        return min(ok, key=lambda x: (abs(x[0] - target), -x[0]))[1] if ok else None

    for cands, ratio in ((strong, 0.35), (soft, 0.55), (spaces, 0.55)):
        idx = best(cands, ratio)
        if idx is not None:
            return idx
    run, idx = 0.0, start + 1
    for j in range(start, n):
        run += weight(s[j])
        idx = j + 1
        if run >= target:
            break
    while idx < n and s[idx] in CLOSERS:
        idx += 1
    while start + 1 < idx < n and s[idx - 1] in OPENERS:
        idx -= 1
    return idx


def split_ranges(s: str, target: float = 24.0, hard: float = 38.0) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = 0
    while start < len(s):
        end = max(_split_once(s, start, target, hard), start + 1)
        out.append((start, end))
        start = end
    if len(out) >= 2:                      # do not leave a two-character orphan
        (a, b), (pa, _) = out[-1], out[-2]
        if text_weight(s[a:b]) < 4 and text_weight(s[pa:b]) <= hard * 1.25:
            out[-2] = (pa, b)
            out.pop()
    assert "".join(s[a:b] for a, b in out) == s
    return out


def build(doc: Doc, times: Times, *, mode: str = "split", target: float = 24.0,
          hard: float = 38.0, max_dur: float = 6.0, gap_split: float = 0.7,
          min_dur: float = 0.7, pad: float = 0.2) -> list[Cue]:
    cues: list[Cue] = []
    for line in doc.lines:
        ranges = [(0, len(line.text))] if mode == "line" else split_ranges(line.text, target, hard)
        for a, b in ranges:
            for sa, sb in _by_time(line, a, b, times, max_dur, gap_split) if mode != "line" else [(a, b)]:
                i0, i1 = line.s0 + sa, line.s0 + sb
                cues.append(Cue(index=len(cues) + 1,
                                start=float(np.nanmin(times.start[i0:i1])),
                                end=float(np.nanmax(times.end[i0:i1])),
                                text=line.text[sa:sb].strip(),
                                speaker=line.speaker, line=line.index, a=i0, b=i1,
                                score=float(np.mean(times.prob[i0:i1]))))
    cues = [c for c in cues if c.text]
    for i, c in enumerate(cues):
        c.index = i + 1
        nxt = cues[i + 1].start if i + 1 < len(cues) else float("inf")
        c.end = min(c.end + pad, max(c.end, nxt - 0.05))
        if c.end - c.start < min_dur:
            c.end = min(c.start + min_dur, max(c.start + 0.25, nxt - 0.05))
    return cues


MIN_PIECE_WEIGHT = 4.0


def _by_time(line: Line, a: int, b: int, times: Times, max_dur: float, gap_split: float):
    """Break a range further where the audio says it should: a long pause, or a cue
    that would otherwise sit on screen past max_dur.

    A pause is only a reason to break if what it separates is worth its own cue.
    Without that floor, a backend that gives a whole word one timestamp turns the
    next character's apparent gap into a one-character cue.
    """
    i0 = line.s0
    out: list[tuple[int, int]] = []
    s = a
    for k in range(a + 1, b):
        gap = times.start[i0 + k] - times.end[i0 + k - 1]
        too_long = times.end[i0 + k] - times.start[i0 + s] > max_dur
        if not (gap > gap_split or too_long):
            continue
        if text_weight(line.text[s:k]) < MIN_PIECE_WEIGHT:
            continue
        if text_weight(line.text[k:b]) < MIN_PIECE_WEIGHT:
            continue
        out.append((s, k))
        s = k
    out.append((s, b))
    return out


def snap(cues: list[Cue], audio, window: float = 0.35, margin_db: float = 12.0) -> int:
    """Pull cue boundaries onto the nearest speech onset.

    Cross-attention DTW (the whisper backend) tends to start a word slightly
    before it is spoken, consistently by a few hundred milliseconds. The audio
    knows better: if a speech onset is within `window` of where a cue starts,
    that onset is where the cue starts. Never moves a boundary past a neighbour,
    and never invents one where the energy is flat.
    """
    from .audio import envelope, speech_mask

    db = envelope(audio)
    mask = speech_mask(db, margin_db)
    if mask.size == 0:
        return 0
    onsets = np.flatnonzero(mask[1:] & ~mask[:-1]) * 0.02
    offsets = np.flatnonzero(~mask[1:] & mask[:-1]) * 0.02
    if onsets.size == 0:
        return 0
    moved = 0
    for i, c in enumerate(cues):
        prev_end = cues[i - 1].end if i else 0.0
        next_start = cues[i + 1].start if i + 1 < len(cues) else float("inf")
        j = int(np.argmin(np.abs(onsets - c.start)))
        cand = float(onsets[j])
        if abs(cand - c.start) <= window and prev_end <= cand < c.end - 0.1:
            if cand != c.start:
                moved += 1
            c.start = cand
        if offsets.size:
            k = int(np.argmin(np.abs(offsets - c.end)))
            cand_e = float(offsets[k])
            # An end may only move to where the speech stops, never past where
            # the next cue begins.
            if abs(cand_e - c.end) <= window and c.start + 0.1 < cand_e < next_start:
                c.end = cand_e
    return moved


def enforce_gaps(cues: list[Cue], gap: float = 0.04, min_dur: float = 0.3) -> int:
    """Guarantee that no cue is still on screen when the next one appears.

    Nothing upstream can promise this. A word's end time is not bounded by the
    next word's start, a trailing syllable overruns, end padding extends, and
    snapping moves boundaries, any of which can leave two cues overlapping.
    Players stack overlapping subtitles or drop one, so the file is defective
    however reasonable each step was.

    The fix is always the same: trim the earlier cue to where the next one
    starts. A cue's start is the moment someone began speaking, which is the
    part worth keeping, while its end is padding or a trailing silence far more
    often than it is speech. So an end moves and a start does not, even when
    trimming leaves the earlier cue shorter than `min_dur` (counted, and
    reported, but still preferred to shifting a real onset).

    The one exception is a later cue that starts at or before the earlier one
    does, where there is no end to trim to. Only then does a start move, because
    the alternative is a cue of negative length.
    """
    fixed = 0
    for prev, cur in zip(cues, cues[1:]):
        if cur.start >= prev.end + gap:
            continue
        fixed += 1
        trimmed = cur.start - gap
        if trimmed > prev.start + 1e-3:
            prev.end = trimmed
            continue
        prev.end = prev.start + 0.05
        cur.start = max(cur.start, prev.end + gap)
        cur.end = max(cur.end, cur.start + 0.05)
    return fixed


def wrap(text: str, max_line: int, lines: int = 2) -> str:
    """Fold a cue onto at most `lines` display lines, breaking at the nearest sane point."""
    if len(text) <= max_line or lines < 2:
        return text
    mid = len(text) // 2
    best, best_d = None, 1e9
    for i in range(1, len(text)):
        if text[i - 1] in SENT_END + SOFT_BREAK or text[i].isspace():
            d = abs(i - mid)
            if d < best_d:
                best, best_d = i, d
    if best is None:
        best = mid
    head, tail = text[:best].strip(), text[best:].strip()
    if lines > 2:
        tail = wrap(tail, max_line, lines - 1)
    return f"{head}\n{tail}"
