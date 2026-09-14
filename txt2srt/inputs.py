"""Check the two inputs and decide where the output goes."""
from __future__ import annotations

from pathlib import Path


class InputError(Exception):
    """Raised with a message meant for the user, not a stack trace."""


def looks_like_text(path: Path, probe: int = 8192) -> bool:
    try:
        head = path.open("rb").read(probe)
    except OSError as e:
        raise InputError(f"{path}: {e.strerror or e}") from e
    if not head:
        raise InputError(f"{path}: file is empty")
    if b"\x00" in head:
        return False
    try:
        text = head.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    # A truncated read can cut a multi-byte character in half, which is not a
    # reason to call a transcript binary.
    printable = sum(1 for c in text if c.isprintable() or c in "\n\r\t")
    return printable / len(text) > 0.9


def resolve(*, text: str | None, audio: str | None) -> tuple[Path, Path]:
    """Return (recording, transcript), or say which flag is missing."""
    if not text or not audio:
        missing = [flag for flag, value in (("-t", text), ("-a", audio)) if not value]
        raise InputError(
            f"missing {' and '.join(missing)}. "
            "Usage: txt2srt -t transcript.txt -a recording.mov")
    return _checked(Path(audio), "audio"), _checked(Path(text), "text")


def _checked(path: Path, kind: str) -> Path:
    """The flags say which file is which, but a transcript still has to be
    readable as text: catching that here beats a decoder failing several seconds
    later on bytes it was never meant to be given."""
    if not path.is_file():
        raise InputError(f"{path}: no such file")
    if kind == "text" and not looks_like_text(path):
        raise InputError(f"{path}: not UTF-8 text. Did you mean -a for this one?")
    return path


def output_path(out: str | None, media: Path) -> Path:
    """Where to write, given -o, which may be a file, a directory, or nothing.

    Named after the recording rather than the transcript: one recording is often
    cut from several transcripts, but the file people look for afterwards is the
    one that shares a name with the clip.
    """
    if out:
        p = Path(out)
        if p.is_dir() or out.endswith(("/", "\\")):
            return p / (media.stem + ".srt")
        if p.suffix:
            return p
        return p / (media.stem + ".srt")      # no extension, treat as a directory
    return media.with_suffix(".srt")
