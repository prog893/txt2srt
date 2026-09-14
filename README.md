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
| `text` | the transcript back out, unchanged, with a timestamp per line |

### `--report`

Forced alignment always returns something, so a wrong result looks like a right
one. `--report` grades the cues against the recording's energy, which the aligner
never sees: what share of cues start and end where someone is actually speaking,
how much of the speech the cues cover, and whether any overlap.

```
  cues                 956
  start on speech      92.3%
  end on speech        96.4%
  overlapping cues     0
```

## Backends

| `-b` | model | licence | notes |
|---|---|---|---|
| `whisper` (default) | `mlx-community/whisper-large-v3-turbo` | MIT, per its model card | Force-decodes the known text through Whisper and reads the timings out of cross-attention. Japanese and English in one pass. |
| `vad` | none | n/a | Energy only: spreads the text across detected speech by how long each character takes to say. A draft to review, not a result. |

## Where the transcript comes from

[mlx-asr](https://github.com/prog893/mlx-asr) for the first pass, a human for
the second, txt2srt for the timings.

## Licence

The code is MIT. Model weights are not covered by it and carry whatever licence
their publisher gives them: check the model card for whichever one you run.
