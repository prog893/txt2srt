"""txt2srt: align a corrected transcript to its audio and write timed files."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import __version__
from .backends import BACKENDS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="txt2srt",
        description="Forced alignment of a known transcript to audio. "
                    "The output text is always the input text.",
    )
    p.add_argument("-t", "--text", metavar="FILE",
                   help="the transcript: UTF-8 text, 'Speaker:' lines, one paragraph per line")
    p.add_argument("-a", "--audio", metavar="FILE",
                   help="the recording: audio or video, anything ffmpeg reads")
    p.add_argument("-o", "--output",
                   help="output file, or a directory to write <recording name>.srt into "
                        "(default: alongside the recording)")
    p.add_argument("-f", "--format", default="",
                   help="extra formats, comma separated: srt,vtt,json,csv,text")
    p.add_argument("-b", "--backend", default="whisper", choices=BACKENDS,
                   help="whisper: force-decode the text through Whisper (default). "
                        "vad: energy only, no model, a draft to review")
    p.add_argument("-m", "--model", default="", help="override the backend's model")
    p.add_argument("-l", "--lang", default="ja", help="primary language of the audio (default: ja)")
    p.add_argument("--cues", default="split", choices=("split", "line"),
                   help="split: subtitle-sized cues (default). line: one cue per transcript line")
    p.add_argument("--speakers", action="store_true", help="prefix cue text with the speaker name")
    p.add_argument("--max-line", type=int, default=0, help="wrap cue text at N characters")
    p.add_argument("--lines", type=int, default=2, help="maximum display lines per cue (default: 2)")
    p.add_argument("--target", type=float, default=24.0, help="target cue size, in weighted characters")
    p.add_argument("--hard", type=float, default=38.0, help="hard cue size limit")
    p.add_argument("--max-dur", type=float, default=6.0, help="split cues longer than this (seconds)")
    p.add_argument("--min-dur", type=float, default=0.7, help="stretch cues shorter than this")
    p.add_argument("--gap-split", type=float, default=0.7, help="split a cue at a pause this long")
    p.add_argument("--pad", type=float, default=0.2, help="extend each cue end into the following silence")
    p.add_argument("--no-snap", action="store_true",
                   help="do not pull cue boundaries onto nearby speech onsets")
    p.add_argument("--snap-window", type=float, default=0.75,
                   help="how far a boundary may be pulled to reach an onset (seconds)")
    p.add_argument("--report", action="store_true",
                   help="check the cues against the recording's energy and print the numbers")
    p.add_argument("--verbose", action="store_true", help="print progress while aligning")
    p.add_argument("-v", "-V", "--version", action="version", version=f"txt2srt {__version__}")
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Called with nothing at all: show what the tool is for, the way zip and tar
    # do. An error message about a missing flag answers a question nobody has
    # asked yet.
    if not args.text and not args.audio:
        parser.print_help()
        return 2
    from . import audio as audio_mod
    from . import cues as cues_mod
    from . import inputs, outputs, report, transcript

    try:
        media, script = inputs.resolve(text=args.text, audio=args.audio)
    except inputs.InputError as e:
        # 2, the shell convention for "you called this wrong", the same code
        # argparse uses for a bad flag. 1 is reserved for an alignment that ran
        # but could not be trusted.
        print(f"txt2srt: {e}", file=sys.stderr)
        return 2

    t0 = time.time()
    doc = transcript.parse(str(script))
    if not doc.lines:
        print(f"txt2srt: {script}: no transcript lines found", file=sys.stderr)
        return 2
    wav = audio_mod.load(str(media))
    dur = len(wav) / audio_mod.SAMPLE_RATE
    say = (lambda m: print(m, flush=True)) if args.verbose else (lambda m: None)
    say(f"[{time.time()-t0:5.1f}s] {len(doc.lines)} lines, {len(doc.stream)} chars, {dur/60:.1f} min audio")

    from .backends import align
    opts = {"lang": args.lang, "verbose": args.verbose}
    if args.model:
        opts["model"] = args.model
    times = align(args.backend, wav, doc, **opts)
    say(f"[{time.time()-t0:5.1f}s] aligned with {args.backend}: {times.timed}/{len(doc.stream)} characters")

    cue_list = cues_mod.build(doc, times, mode=args.cues, target=args.target, hard=args.hard,
                              max_dur=args.max_dur, gap_split=args.gap_split,
                              min_dur=args.min_dur, pad=args.pad)
    if not cue_list:
        raise SystemExit("no cues were produced")
    if not args.no_snap:
        moved = cues_mod.snap(cue_list, wav, window=args.snap_window)
        say(f"[{time.time()-t0:5.1f}s] snapped {moved}/{len(cue_list)} cue starts to a speech onset")
    overlaps = cues_mod.enforce_gaps(cue_list)
    if overlaps:
        say(f"[{time.time()-t0:5.1f}s] separated {overlaps} overlapping cues")

    out = inputs.output_path(args.output, media)
    out.parent.mkdir(parents=True, exist_ok=True)
    wanted = [f for f in (out.suffix.lstrip(".").lower(), *args.format.split(",")) if f]
    seen, formats = set(), []
    for f in wanted:
        if f in outputs.FORMATS and f not in seen:
            seen.add(f)
            formats.append(f)
    if not formats:
        formats = ["srt"]

    written = []
    for fmt in formats:
        path = out if out.suffix.lstrip(".").lower() == fmt else out.with_suffix(f".{fmt}")
        if fmt == "srt":
            outputs.write_srt(cue_list, path, speakers=args.speakers,
                              max_line=args.max_line, lines=args.lines)
        elif fmt == "vtt":
            outputs.write_vtt(cue_list, path, speakers=args.speakers,
                              max_line=args.max_line, lines=args.lines)
        elif fmt == "csv":
            outputs.write_csv(cue_list, path)
        elif fmt == "json":
            # The recording's name, not its path: a JSON sidecar gets shared, and
            # the directory it was aligned in is nobody else's business.
            outputs.write_json(doc, times, cue_list, path, backend=times.backend,
                               model=times.model, audio=media.name, duration=dur)
        elif fmt == "text":
            outputs.write_text(doc, times, path, speakers=True)
        written.append(path)

    for path in written:
        print(path)
    if args.report:
        print(report.render(report.build(wav, cue_list, times)))
    return 0


def cli() -> None:
    sys.exit(main())
