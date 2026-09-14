"""The document model.

One idea holds the whole tool together: the transcript is turned into a single
character *stream*, every backend does nothing but attach times to positions in
that stream, and every output slices the ORIGINAL text by those positions. No
output ever renders text that came back from a model, so the wording, spelling,
punctuation and casing that a human fixed cannot be silently replaced by an
ASR-shaped approximation of it.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field

# "Name:" on a line of its own. Anchored and length-capped so a sentence that
# happens to end in a colon is not mistaken for a speaker label.
SPEAKER_RE = re.compile(r"^([^\s:][^:\n]{0,58}):$")
JP_RE = re.compile(r"[぀-ヿ㐀-鿿ｦ-ﾟ]")

# Lines are joined by one space so that stream slices are exactly the source
# text and the separator never lands inside a cue.
SEP = " "


@dataclass
class Line:
    index: int
    speaker: str | None
    text: str
    lang: str          # "ja" or "en", per line, so mixed transcripts work
    s0: int            # offset of this line's first char in the stream
    s1: int            # offset one past its last char


@dataclass
class Doc:
    path: str
    sha256: str
    stream: str
    lines: list[Line] = field(default_factory=list)

    @property
    def speakers(self) -> list[str]:
        seen = {}
        for ln in self.lines:
            if ln.speaker:
                seen[ln.speaker] = None
        return list(seen)

    def line_of(self, pos: int) -> Line | None:
        for ln in self.lines:            # transcripts are small; linear is fine
            if ln.s0 <= pos < ln.s1:
                return ln
        return None

    def check(self) -> None:
        """Every line must still be an exact slice of the stream."""
        for ln in self.lines:
            if self.stream[ln.s0:ln.s1] != ln.text:
                raise ValueError(f"line {ln.index} is not an exact slice of the stream")


def is_japanese(s: str) -> bool:
    return bool(JP_RE.search(s))


def parse(path: str) -> Doc:
    raw = open(path, "rb").read()
    text = raw.decode("utf-8-sig")
    lines: list[Line] = []
    speaker = None
    parts: list[str] = []
    pos = 0
    for block in text.split("\n"):
        s = block.strip()
        if not s:
            continue
        m = SPEAKER_RE.match(s)
        if m:
            speaker = m.group(1).strip()
            continue
        if parts:
            pos += len(SEP)
        lines.append(Line(index=len(lines), speaker=speaker, text=s,
                          lang="ja" if is_japanese(s) else "en",
                          s0=pos, s1=pos + len(s)))
        parts.append(s)
        pos += len(s)
    doc = Doc(path=path, sha256=hashlib.sha256(raw).hexdigest(), stream=SEP.join(parts), lines=lines)
    doc.check()
    return doc


def alignable(ch: str) -> bool:
    """Characters a model can be asked to find in audio: letters and digits.

    Punctuation and spaces are silent, so they are timed by interpolation from
    their neighbours instead. They are still carried into the output verbatim,
    they simply do not get a say in where a cue starts.
    """
    cat = unicodedata.category(ch)
    return cat.startswith("L") or cat.startswith("N")
