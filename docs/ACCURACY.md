# What the backends actually do

Measured, not estimated, on about an hour of Japanese speech with English
passages, against a human-corrected transcript of it. The recording is private,
so what transfers here is the method and the size of the effects, not the exact
values. M2 Ultra, default settings.

| backend | start on speech | end on speech | speech covered | cues | over 6s | over cps | wall |
|---|---|---|---|---|---|---|---|
| `whisper` | 94.9% | 95.1% | 95.2% | 943 | 45 | 16 | 59s |
| `vad` | 100.0%\* | 100.0%\* | 98.6% | 1042 | 53 | 31 | 15s |

\* Circular, and listed only to make the circularity visible. The `vad` backend
pours the text into the VAD's speech regions, and `--report` grades against the
VAD's speech regions. It cannot fail its own exam. The column means something
for `whisper`, which never sees it.

## What counts as speech

Every part of this tool that reasons about silence used to ask the energy
envelope: frames above the tenth percentile plus a margin were speech. On the
same hour of audio, at the 12 dB margin that was the default:

| | segments found | median segment | share of file |
|---|---|---|---|
| energy, 6 dB | 7708 | 0.12s | 75.7% |
| energy, 12 dB | 8425 | 0.16s | 56.6% |
| energy, 18 dB | 8155 | 0.14s | 44.8% |
| Silero VAD | 1182 | 1.60s | 70.5% |

Eight thousand speech runs with a median length of a sixth of a second is not
speech. It is an amplitude envelope flickering on syllables and room tone, and
no choice of margin fixes it, because the problem is that loudness is not the
question being asked. An hour of two people talking is about a thousand
segments of a second or two, which is what the VAD returns.

This was not a cosmetic difference. The energy gate decided where `--snap-window`
was allowed to move a boundary, what `--report` graded, and which words the
aligner believed had run into a pause.

## Words that swallow a pause

DTW has to account for every frame of its window, so a pause between two words
is charged to the word after it. The word ends where it was really said; it
starts wherever the silence began.

```
line 302  '手を出していただければ'
   '手'   2078.86 -> 2103.36      24.5s for one character
   'を'   2103.36 -> 2103.46
   '出'   2103.46 -> 2103.56
```

335 of 10866 timed units came back like this. Cue boundaries are the minimum
start over a range, so each one set the start of whatever cue contained it, and
nine cues ran past 20 seconds on screen holding four characters of text.
Whisper's own confidence does not find them: cues over 10s score a median 0.598
against 0.830 for the rest, but the healthy tenth percentile is 0.482, and only
7 of the 95 lowest-scoring cues are these.

`clamp_spans` moves such a word up to wherever the VAD says the speech it ends
inside began. That bound alone is not enough. A transcript is an account of what
was said, not of every sound in the room, so an untranscribed interjection or an
overlapping second speaker leaves real speech that no word belongs to: the
pauses these 335 units were holding are 43.5% speech by the VAD, 142 of them
more than half. `max_lead` (1.0s) and `min_rate` (4.0 weighted characters per
second) are the backstop for that, and the tighter of the two bounds wins.

An earlier version of the same fix used energy onsets instead of the VAD. It
scored better on the energy-graded report and was worse:

| same alignment, three cue sets | over 6s | worst cue | start on speech (VAD) | speech covered (VAD) |
|---|---|---|---|---|
| no clamp | 75 | 27.5s | 92.7% | 95.4% |
| clamp on energy onsets | 35 | 21.2s | 93.3% | **89.8%** |
| clamp on the VAD | 45 | 26.7s | **95.0%** | 95.2% |

The energy version produced the shortest cues by cutting into speech that was
really there. Graded independently, it moved cue starts by 0.6 points and gave
up 5.6 points of coverage to do it.

## Snapping

`--snap-window` pulls a cue boundary onto a speech onset within that distance,
which is worth doing because cross-attention DTW opens a word slightly before it
is spoken. Measured across the same file:

| window | start on speech | speech covered | cues under 0.3s |
|---|---|---|---|
| off | 92.7% | 94.2% | 12 |
| 0.35s | 92.9% | 95.1% | 13 |
| 0.5s | 94.2% | 95.4% | 14 |
| **0.75s (default)** | **94.9%** | **95.2%** | 18 |
| 1.0s | 95.0% | 95.0% | 19 |
| 1.5s | 96.5% | 94.3% | 22 |

Read the first column with suspicion: snapping moves cue starts onto VAD onsets
and that column asks how many cue starts are on VAD onsets, so a wider window
wins it by construction. The columns that are not circular are the other two,
and they turn over together at about 0.75. Past it, a boundary reaches an onset
belonging to something else, which shows up as overlaps to trim and therefore as
cues too short to read.

## What is still wrong

A cue spans from the earliest start to the latest end among its characters, so
scattering survives clamping:

```
line 363  'ありがとうございます'
   'あり'       2444.56 -> 2446.10     VAD: 0% speech
   'が'         2458.45 -> 2459.70     VAD: 0% speech
   'とう'       2471.26 -> 2472.76     VAD: 0% speech
   'ございます' 2472.76 -> 2473.64     VAD: 0% speech
```

Every unit is now a plausible length and every one of them is in silence. The
aligner does not know where this line was said, and says so only by spreading
it across 29 seconds. Bounding a unit cannot pull units together, and the report
counts this as one long cue rather than as the failure it is.

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
