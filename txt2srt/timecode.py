"""Seconds to SMPTE timecode, including drop-frame.

A recording's start timecode is written the way its editor writes it, and at
29.97 that is usually drop-frame ("00:00:01;04", semicolon). Reading it back as
non-drop would put every cue 3.6 seconds out per hour, so this implements the
standard renumbering: 18 frames dropped every ten minutes, none on the tenth.
"""
from __future__ import annotations

DROP_RATES = {29.97, 59.94}


def is_drop(fps: float) -> bool:
    return round(fps, 2) in DROP_RATES


def to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def format_tc(seconds: float, fps: float, drop: bool | None = None) -> str:
    drop = is_drop(fps) if drop is None else drop
    nominal = int(round(fps))
    f = max(0, to_frames(seconds, fps))
    if drop:
        dropped = nominal // 15                      # 2 at 30, 4 at 60
        per_10min = nominal * 600 - dropped * 9
        per_min = nominal * 60 - dropped
        d, m = divmod(f, per_10min)
        f += dropped * 9 * d
        if m >= dropped:
            f += dropped * ((m - dropped) // per_min)
    h, rem = divmod(f, nominal * 3600)
    m, rem = divmod(rem, nominal * 60)
    s, fr = divmod(rem, nominal)
    return f"{h:02d}:{m:02d}:{s:02d}{';' if drop else ':'}{fr:02d}"


def parse_tc(tc: str, fps: float) -> float:
    drop = ";" in tc
    parts = tc.replace(";", ":").split(":")
    h, m, s, f = (int(p) for p in parts)
    nominal = int(round(fps))
    total = ((h * 60 + m) * 60 + s) * nominal + f
    if drop:
        dropped = nominal // 15
        total -= dropped * (9 * h * 6 + (m - m // 10))
    return total / fps


def srt_time(seconds: float) -> str:
    ms = int(round(max(0.0, seconds) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def vtt_time(seconds: float) -> str:
    return srt_time(seconds).replace(",", ".")
