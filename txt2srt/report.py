"""Did the alignment actually land on the speech?

An aligner always returns something: forced alignment has no failure mode where
it declines to answer, so a wrong answer looks exactly like a right one. The
independent check is the VAD, which the aligner's own decisions never touch. If
a cue starts where nobody is speaking, the number here moves, and it moves
before anyone watches an hour of subtitles to notice.
"""
from __future__ import annotations

import numpy as np

from .cues import text_weight


def build(speech, cues, times, *, pre: float = 0.1, post: float = 0.3,
          max_cps: float = 20.0) -> dict:
    mask = speech.mask
    n = len(mask)

    def frame(t: float) -> int:
        return int(min(max(t / 0.02, 0), max(n - 1, 0)))

    starts_ok = ends_ok = 0
    misses = []
    for c in cues:
        f = frame(c.start)
        if mask[max(0, f - int(pre / 0.02)): f + int(post / 0.02)].any():
            starts_ok += 1
        else:
            misses.append(c)
        e = frame(c.end)
        if mask[max(0, e - int(post / 0.02)): e + int(pre / 0.02)].any():
            ends_ok += 1

    covered = np.zeros(n, dtype=bool)
    for c in cues:
        covered[frame(c.start): frame(c.end) + 1] = True

    overlaps = sum(1 for a, b in zip(cues, cues[1:]) if b.start < a.end - 1e-9)
    short = sum(1 for c in cues if c.end - c.start < 0.3)
    durations = np.array([c.end - c.start for c in cues])
    cps = np.array([text_weight(c.text) / max(c.end - c.start, 1e-3) for c in cues])
    scores = np.array([c.score for c in cues])
    return {
        "cues": len(cues),
        "chars_timed": int(times.timed),
        "start_on_speech": starts_ok / max(len(cues), 1),
        "end_on_speech": ends_ok / max(len(cues), 1),
        "speech_covered": float((mask & covered).sum() / max(mask.sum(), 1)),
        "duration_p50": float(np.median(durations)) if len(durations) else 0.0,
        "duration_p95": float(np.percentile(durations, 95)) if len(durations) else 0.0,
        "over_max_dur": int((durations > 6.0).sum()),
        "over_max_cps": int((cps > max_cps).sum()),
        "overlaps": overlaps,
        "short": short,
        "score_p10": float(np.percentile(scores, 10)) if len(scores) else 0.0,
        "worst": [(c.index, round(c.start, 2), round(c.score, 3), c.text[:40])
                  for c in sorted(cues, key=lambda c: c.score)[:5]],
        "misses": [(c.index, round(c.start, 2), c.text[:40]) for c in misses[:5]],
    }


def render(rep: dict) -> str:
    lines = [
        f"  cues                {rep['cues']}",
        f"  characters timed    {rep['chars_timed']}",
        f"  start on speech     {rep['start_on_speech']*100:5.1f}%",
        f"  end on speech       {rep['end_on_speech']*100:5.1f}%",
        f"  speech covered      {rep['speech_covered']*100:5.1f}%",
        f"  duration p50/p95    {rep['duration_p50']:.2f}s / {rep['duration_p95']:.2f}s",
        f"  over 6s / over cps  {rep['over_max_dur']} / {rep['over_max_cps']}",
        f"  overlapping cues    {rep['overlaps']}",
        f"  under 0.3s          {rep['short']}",
    ]
    if rep["misses"]:
        lines.append("  cue starts in silence (first few):")
        lines += [f"    #{i:<5} {t:8.2f}  {txt}" for i, t, txt in rep["misses"]]
    return "\n".join(lines)
