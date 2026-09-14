"""Whisper force-decode alignment (default backend).

Whisper is never asked what was said. The known text is fed to the decoder as
teacher forcing and the timings are read out of the cross-attention with DTW,
the same machinery Whisper uses for its own word timestamps. Japanese and
English align in one pass, because the tokenizer covers both and nothing has to
be converted to a reading first.

The window loop is the whole problem. Whisper sees 30 seconds at a time, and
which piece of text belongs to which window is exactly what is being solved for,
so the text has to be guessed at before it can be aligned. DTW then aligns
whatever it is given, in full: hand a 30 second window 60 seconds of text and it
will not object, it will compress the surplus into the final frames. Accept
those and the next window starts with text that has already been spent, which
walks the alignment steadily earlier for the rest of the file.

Three things keep that from happening, and all three are load-bearing:

  * the text offered to a window is sized from a running estimate of how fast
    this speaker is actually talking, not from a fixed character count,
  * words landing in the last seconds of a window are dropped unheard, along
    with the two before them, since that is where surplus text piles up,
  * progress is checked against the audio at the end, because a drift of this
    kind produces a result that looks perfectly well-formed.
"""
from __future__ import annotations

import numpy as np

from ..transcript import Doc
from . import Times

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"

# Seconds at the end of each window whose words are never trusted.
EDGE_GUARD_S = 1.5
# Words dropped from the tail of a window's accepted run, on top of the guard.
TAIL_DROP = 1
# Initial characters per second OF SPEECH, before the audio has said otherwise.
# Per second of speech, not of wall clock: a window that is half silence needs
# half the text, and pauses are the largest source of variance in how much text
# a window can hold. Japanese is counted in characters, where one character is
# roughly one mora.
RATE0 = {"ja": 8.0, "zh": 6.5, "ko": 7.0}
RATE0_DEFAULT = 16.0
# Text offered to a window, as a fraction of what the rate estimate says fits.
# Deliberately below 1: surplus text is not merely ignored, DTW compresses the
# words that do belong in the window to make room for it, so every word in the
# window comes out early and the error compounds across the file. Underfilling
# costs an extra window here and there and nothing else.
FILL = 0.85
MAX_TOKENS = 220

# Punctuation that belongs to the word after it, and to the word before it.
# Taken from Whisper's own word-timestamp defaults (openai/whisper, MIT), which
# mlx_whisper.timing.merge_punctuations expects to be given.
PREPEND = "\"'“¿([{-"
APPEND = "\"'.。,，!！?？:：”)]}、"


def align(audio: np.ndarray, doc: Doc, *, model: str = DEFAULT_MODEL,
          lang: str = "ja", verbose: bool = False, **_) -> Times:
    import mlx.core as mx
    from mlx_whisper.audio import (HOP_LENGTH, N_FRAMES, N_SAMPLES, SAMPLE_RATE,
                                   log_mel_spectrogram)
    from mlx_whisper.load_models import load_model
    from mlx_whisper.timing import find_alignment, merge_punctuations
    from mlx_whisper.tokenizer import get_tokenizer

    net = load_model(model, dtype=mx.float16)
    tok = get_tokenizer(net.is_multilingual, num_languages=net.num_languages,
                        language=lang, task="transcribe")

    from ..audio import envelope, speech_mask

    mask = speech_mask(envelope(audio))
    speech_cum = np.concatenate([[0.0], np.cumsum(mask) * 0.02])

    def speech_between(t0: float, t1: float) -> float:
        i = min(max(int(t0 / 0.02), 0), len(speech_cum) - 1)
        j = min(max(int(t1 / 0.02), 0), len(speech_cum) - 1)
        return float(speech_cum[j] - speech_cum[i])

    stream = doc.stream
    times = Times.empty(len(stream), "whisper", model)
    window_s = N_SAMPLES / SAMPLE_RATE
    rate = RATE0.get(lang, RATE0_DEFAULT)
    seek = 0
    pos = 0
    stalls = 0

    while seek < len(audio) and pos < len(stream):
        window = audio[seek: seek + N_SAMPLES]
        n_frames = min(N_FRAMES, len(window) // HOP_LENGTH)
        mel = log_mel_spectrogram(window, net.dims.n_mels,
                                  padding=max(0, N_SAMPLES - len(window)))
        final = seek + N_SAMPLES >= len(audio)

        t_win = seek / SAMPLE_RATE
        speech_s = speech_between(t_win, t_win + window_s)
        want = int(min(max(rate * speech_s * FILL, 30), 600))
        chunk = stream[pos: pos + want]
        tokens = tok.encode(chunk)[:MAX_TOKENS]
        words = find_alignment(net, tok, tokens, mel, n_frames)
        if words:
            merge_punctuations(words, PREPEND, APPEND)
        if not words:
            seek += N_SAMPLES // 2
            stalls += 1
            continue

        if final:
            kept = words
        else:
            kept = [w for w in words if w.end <= window_s - EDGE_GUARD_S]
            if len(kept) > TAIL_DROP + 1:
                kept = kept[:-TAIL_DROP]
            elif not kept:
                kept = words[:1]
        consumed = "".join(w.word for w in kept)

        # The decoder's text must still be our text. It normally is (BPE round
        # trips), but a stray character would shift every offset after it, so
        # verify rather than assume and re-anchor by search if it drifted.
        base = pos
        if not stream.startswith(consumed, pos):
            probe = consumed.strip()[:24]
            found = stream.find(probe, pos, pos + want * 2) if probe else -1
            if found >= 0:
                base = found
        cursor = base
        for w in kept:
            s = seek / SAMPLE_RATE + float(w.start)
            e = seek / SAMPLE_RATE + float(w.end)
            times.set(cursor, cursor + len(w.word), s, e, float(w.probability))
            cursor += len(w.word)

        # Rate is measured against seconds of speech the audio cursor moved over,
        # never against DTW's own span for the words: a compressed span would
        # raise the estimate, which would feed the next window more text, which
        # would compress it further.
        advance_s = float(kept[-1].end)
        spoken = speech_between(t_win, t_win + advance_s)
        if spoken > 1.5 and len(consumed) > 8:
            observed = len(consumed) / spoken
            rate = 0.75 * rate + 0.25 * min(max(observed, 3.0), 40.0)

        advance = int(max(float(kept[-1].end), 0.1) * SAMPLE_RATE)
        if cursor <= pos or advance <= 0:
            seek += N_SAMPLES // 2
            pos = max(cursor, pos + 1)
            stalls += 1
        else:
            pos, seek = cursor, seek + advance
        if verbose:
            print(f"  {seek/SAMPLE_RATE:7.1f}s  {pos:6d}/{len(stream)} chars"
                  f"  {rate:4.1f} char/s of speech  (fed {want})", flush=True)
        if stalls > 64:
            raise RuntimeError("alignment stalled: audio and transcript may not match")

    _check_drift(times, audio, stream, seek, pos)
    return times


def _check_drift(times: Times, audio, stream: str, seek: int, pos: int) -> None:
    """Refuse to hand back an alignment that ran out of one side early.

    A drifting window loop still produces a complete, monotonic, plausible-looking
    result: every character has a time and every cue starts on some speech. What
    it does not do is reach the end of the audio at the same moment it reaches the
    end of the text.
    """
    from ..audio import SAMPLE_RATE

    total = len(audio) / SAMPLE_RATE
    reached = seek / SAMPLE_RATE
    if pos >= len(stream) and total - reached > max(30.0, 0.1 * total):
        raise RuntimeError(
            f"alignment drifted: the transcript ran out at {reached/60:.1f} min of "
            f"{total/60:.1f} min of audio. The transcript may cover only part of this "
            f"file, or the audio may not be the recording it was made from.")
