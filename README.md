# txt2srt

Takes a recording and a transcript that has already been corrected. Returns that
same text with timings on it.

This is forced alignment, not transcription. No model is ever asked what was
said, so nothing can quietly replace the wording, punctuation, spelling or
speaker labels a human fixed. Every cue is a slice of the source file, and the
tool refuses to write output it cannot reconstruct from that file.

Built for Japanese and for mixed Japanese/English interviews, where general
subtitle tooling does worst. English-only works too.

## Install

Apple Silicon, macOS 14+.

```bash
brew install prog893/tap/txt2srt
```

<details>
<summary>From source</summary>

Python 3.12+, and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/prog893/txt2srt && cd txt2srt
uv sync --extra whisper
uv run txt2srt -t transcript.txt -a recording.mov
```
</details>

Whisper weights (about 1.6 GB for the default) download on first use into
`~/.cache/huggingface`.

## Use

```bash
txt2srt -t transcript.txt -a recording.mov
```

That writes `recording.srt` beside the recording. `-t` is the transcript, `-a`
is the recording: audio or video, anything ffmpeg reads.

`-o` says where it goes. Given a filename it is used as-is, given a directory the
file lands in there named after the recording.

```bash
txt2srt -t transcript.txt -a recording.mov -o subs/
txt2srt -t transcript.txt -a recording.mov -o subs/episode-3.srt
```

The transcript is plain UTF-8: a speaker on its own line ending in a colon, then
one paragraph per line. Blank lines are ignored.

```
話し手 A:
次の電車は7時12分に出ます

話し手 B:
間に合いますか？
```

### Options

| | |
|---|---|
| `-l ja` | language of the recording, `ja` by default |
| `-f json,csv` | write these formats alongside the subtitle |
| `--report` | print how well the cues line up with the audio |
| `--cues line` | one cue per transcript line instead of subtitle-sized cues |
| `--speakers` | put the speaker name in the cue text |
| `--max-line 42` | wrap cue text at this width, over at most `--lines` lines |
| `--start-tc 01:00:00:00` | shift every time by the recording's start timecode |
| `-b vad` | align without a model, see below |
| `--verbose` | progress while it runs |

Cue shaping has its own flags (`--target`, `--hard`, `--max-dur`, `--min-dur`,
`--gap-split`, `--pad`, `--snap-window`); `txt2srt -h` lists them with defaults.

### Formats

`-o` picks the main format by extension, `-f` adds more beside it.

| format | what it is |
|---|---|
| `srt` | subtitles, for delivery |
| `vtt` | the same, WebVTT |
| `json` | every line and word with times, scores, and the source hash |
| `csv` | one row per cue, for review in a spreadsheet |
| `text` | the transcript back out, unchanged, with a timecode per line |

### `--report`

Forced alignment always returns something, so a wrong result looks like a right
one. `--report` grades the cues against the VAD, whose answer the aligner's own
decisions never touch: what share of cues start and end where someone is
actually speaking, how much of the speech the cues cover, and whether any
overlap.

```
  cues                943
  start on speech      94.9%
  end on speech        95.1%
  speech covered       95.2%
  overlapping cues     0
```

`speech covered` is the one to watch. The others grade each cue on its own, so
an alignment that has slid off the audio entirely still passes them.

## Backends

| `-b` | model | licence | notes |
|---|---|---|---|
| `whisper` (default) | `mlx-community/whisper-large-v3-turbo` | MIT, per its model card | Force-decodes the known text through Whisper and reads the timings out of cross-attention. Japanese and English in one pass. |
| `vad` | none beyond the bundled VAD | n/a | No ASR: spreads the text across detected speech by how long each character takes to say. A draft to review, not a result. |

Both first ask a voice activity detector where anybody is speaking at all. That
answer decides where cue boundaries may move, what `--report` grades against,
and which words the aligner is told have run into a pause. It is Silero VAD,
MIT, 2.3 MB of ONNX weights carried inside the package, so there is nothing to
download and no torch install. It costs about 11 seconds per hour of audio.

## Where the transcript comes from

[mlx-asr](https://github.com/prog893/mlx-asr) for the first pass, a human for
the second, txt2srt for the timings.

## Licence

The code is MIT. Model weights are not covered by it and carry whatever licence
their publisher gives them: check the model card for whichever one you run.
