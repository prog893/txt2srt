"""Writers. Every one of them takes its text from the source document."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from .cues import Cue, wrap

FORMATS = ("srt", "vtt", "json", "csv", "text")


def srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")


def _label(cue: Cue, speakers: bool) -> str:
    return f"{cue.speaker}: {cue.text}" if speakers and cue.speaker else cue.text


def write_srt(cues, path, *, speakers=False, max_line=0, lines=2):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, c in enumerate(cues, 1):
            text = _label(c, speakers)
            if max_line:
                text = wrap(text, max_line, lines)
            f.write(f"{i}\n{srt_time(c.start)} --> {srt_time(c.end)}\n{text}\n\n")


def write_vtt(cues, path, *, speakers=False, max_line=0, lines=2):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("WEBVTT\n\n")
        for i, c in enumerate(cues, 1):
            text = _label(c, speakers)
            if max_line:
                text = wrap(text, max_line, lines)
            f.write(f"{i}\n{vtt_time(c.start)} --> {vtt_time(c.end)}\n{text}\n\n")


def write_csv(cues, path):
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cue", "start", "end", "speaker", "line", "score", "text"])
        for c in cues:
            w.writerow([c.index, f"{c.start:.3f}", f"{c.end:.3f}", c.speaker or "",
                        c.line, f"{c.score:.4f}", c.text])


def _words(doc, times, a: int, b: int):
    """Recover the backend's own word/unit granularity: consecutive characters
    that share a start and end were timed as one unit."""
    out = []
    i = a
    while i < b:
        j = i + 1
        while j < b and times.start[j] == times.start[i] and times.end[j] == times.end[i]:
            j += 1
        text = doc.stream[i:j]
        if text.strip():
            out.append({"start": float(times.start[i]), "end": float(times.end[j - 1]), "text": text})
        i = j
    return out


def write_json(doc, times, cues, path, *, backend: str, model: str, audio: str, duration: float):
    payload = {
        "txt2srt": "1",
        "audio": audio,
        "duration": round(duration, 3),
        "transcript": {"name": Path(doc.path).name, "sha256": doc.sha256,
                       "chars": len(doc.stream)},
        "alignment": {"backend": backend, "model": model,
                      "chars_timed": int(times.timed)},
        "speakers": doc.speakers,
        "lines": [
            {"index": ln.index, "speaker": ln.speaker, "lang": ln.lang, "text": ln.text,
             "start": float(np.nanmin(times.start[ln.s0:ln.s1])),
             "end": float(np.nanmax(times.end[ln.s0:ln.s1])),
             "words": _words(doc, times, ln.s0, ln.s1)}
            for ln in doc.lines
        ],
        "cues": [
            {"index": c.index, "start": round(c.start, 3), "end": round(c.end, 3),
             "speaker": c.speaker, "line": c.line, "score": round(c.score, 4), "text": c.text}
            for c in cues
        ],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")


def write_text(doc, times, path, *, speakers=True):
    """The transcript back out, unchanged, with a timestamp per line. Sanity check
    that nothing was rewritten: diff it against the input."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        last = None
        for ln in doc.lines:
            if speakers and ln.speaker != last:
                f.write(f"\n{ln.speaker}:\n")
                last = ln.speaker
            f.write(f"[{srt_time(float(np.nanmin(times.start[ln.s0:ln.s1])))}] {ln.text}\n")


