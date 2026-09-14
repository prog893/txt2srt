# What the backends actually do

Measured, not estimated, on about an hour of Japanese speech with English
passages, against a human-corrected transcript of it. The recording is private,
so what transfers here is the method and the size of the effects, not the exact
values. M2 Ultra, default settings.

| backend | start on speech | end on speech | speech covered | cues | over 6s | over cps | wall |
|---|---|---|---|---|---|---|---|
| `whisper` | 91.7% | 99.0% | 99.0% | 956 | 92 | 9 | 52s |
| `vad` | 100.0%\* | 100.0%\* | 99.6% | 1114 | 58 | 29 | 2s |

\* Circular, and listed only to make the circularity visible. The `vad` backend
derives its timings from the energy envelope, and `--report` grades against the
energy envelope. It cannot fail its own exam. The column means something for
`whisper`, which never sees it.

## Snapping

`--snap-window` pulls a cue boundary onto a speech onset within that distance,
which is worth doing because cross-attention DTW opens a word slightly before it
is spoken. Measured across the same file:

| window | start on speech | end on speech |
|---|---|---|
| off | 88.4% | 93.6% |
| 0.35s | 89.9% | 96.2% |
| 0.5s | 90.9% | 97.9% |
| **0.75s (default)** | **91.7%** | **99.0%** |
| 1.0s | 91.9% | 99.3% |

Past 0.75 the gain is a rounding error and the risk is not: a wider window can
reach an onset that belongs to something else.

## What this was measured against

A second backend, `mms`, ran CTC Viterbi over the whole file using
`MahmoudAshraf/mms-300m-1130-forced-aligner`, with Japanese romanized through
cutlet. It scored better than `whisper` on this file: 99.4% of cue starts on a
speech onset against 91.7%, and it is what the numbers below compare to.

It was removed anyway. Its model card gives the weights as CC BY-NC 4.0, which
rules out commercial work, and a backend that cannot be used for the job it was
built for is a liability sitting in the repo waiting to be reached for.

Line start times, `whisper` against `mms` while both existed (at the 0.5s snap
window that was the default then):

| | |
|---|---|
| median difference | -0.68s (whisper starts earlier) |
| within 0.3s | 26.5% |
| within 1s | 56.7% |
| within 3s | 83.8% |
| worst | 34s |

For scale, the same comparison for `vad`: median +20s, 2.7% of lines within a
second. Energy alone locates speech, never which words it was.

## The drift that this measurement exists to catch

The first working version of the `whisper` backend ran out of transcript a
little past the halfway mark of the audio, and by the end it was placing text
some twenty-five minutes early. Its `--report` looked like this:

```
  start on speech      91.6%
  end on speech        94.2%
  speech covered       58.9%
```

Every character had a time, every cue was monotonic, nine cue starts in ten sat
on real speech, and the result was worthless. The only column that noticed was
`speech covered`, and only because it happens to measure the whole file rather
than each cue in isolation.

The cause: Whisper aligns whatever text it is handed across the whole 30 second
window, so surplus text does not sit idle at the end, it compresses the words
that do belong there. Feeding each window slightly less text than the estimate,
sizing that estimate by seconds of *speech* rather than seconds of wall clock,
and dropping the words nearest the window edge removed the drift
(`speech covered` 58.9% -> 99.0%). `_check_drift` now refuses to return an
alignment that ran out of transcript while audio remained.

Two things follow, and they generalise past this tool. Forced alignment has no
failure mode where it declines to answer. And a check that shares an input with
the thing it is checking is not a check.
