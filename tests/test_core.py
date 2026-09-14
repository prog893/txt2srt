import numpy as np
import pytest

from txt2srt import cues, transcript
from txt2srt import speech
from txt2srt.backends import Times, clamp_spans
from txt2srt.timecode import format_tc, parse_tc, srt_time


def write(tmp_path, text):
    p = tmp_path / "t.txt"
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_parse_keeps_text_exactly(tmp_path):
    doc = transcript.parse(write(tmp_path, "話し手 A:\n次の電車は7時12分です\n\n話し手 B:\n間に合いますか？\n"))
    assert [l.speaker for l in doc.lines] == ["話し手 A", "話し手 B"]
    assert [l.text for l in doc.lines] == ["次の電車は7時12分です", "間に合いますか？"]
    assert all(doc.stream[l.s0:l.s1] == l.text for l in doc.lines)


def test_colon_inside_a_sentence_is_not_a_speaker(tmp_path):
    doc = transcript.parse(write(tmp_path, "A:\nthe rule is this: never rewrite the text\n"))
    assert len(doc.lines) == 1
    assert doc.lines[0].speaker == "A"


def test_split_ranges_rejoin(tmp_path):
    line = "駅前の図書館は、平日は9時から20時まで開いていて、休館日は毎月第三月曜日と年末年始になっています"
    parts = cues.split_ranges(line)
    assert "".join(line[a:b] for a, b in parts) == line
    assert len(parts) > 1


def test_fill_gaps_is_monotonic_and_total():
    t = Times.empty(10, "test")
    t.set(2, 4, 1.0, 2.0)
    t.set(8, 10, 5.0, 6.0)
    t.fill_gaps()
    assert not np.isnan(t.start).any()
    assert (np.diff(t.start) >= -1e-9).all()
    assert (t.end >= t.start).all()


def test_cue_text_is_a_slice_of_the_source(tmp_path):
    doc = transcript.parse(write(tmp_path, "A:\nこれはテストです\n\nB:\nもう一度、テストです\n\nA:\n三回目です\n"))
    t = Times.empty(len(doc.stream), "test")
    for i in range(len(doc.stream)):
        t.set(i, i + 1, i * 0.2, i * 0.2 + 0.2, 0.9)
    built = cues.build(doc, t)
    assert built
    assert "".join(c.text for c in built).replace(" ", "") == doc.stream.replace(" ", "")
    for c in built:
        assert c.text in doc.stream


def test_timecode_round_trip():
    for fps, tc in ((24.0, "01:02:03:04"), (25.0, "00:10:00:12"), (29.97, "00:10:00;02")):
        assert format_tc(parse_tc(tc, fps), fps) == tc


def test_drop_frame_labels_track_real_time():
    # Drop-frame skips two labels at every minute except each tenth, so that
    # timecode and wall clock agree at ten-minute marks. One minute of real time
    # is 1798 frames, which is still labelled in the first minute; the labels
    # 00:01:00;00 and ;01 are the ones that never appear.
    assert format_tc(60.0, 29.97) == "00:00:59;28"
    assert format_tc(1800 / 29.97, 29.97) == "00:01:00;02"
    assert format_tc(600.0, 29.97) == "00:10:00;00"


def test_srt_time():
    assert srt_time(3661.5) == "01:01:01,500"


def test_output_path(tmp_path):
    from txt2srt.inputs import output_path

    media = tmp_path / "clip.mov"
    out = tmp_path / "out"
    out.mkdir()
    assert output_path(None, media) == tmp_path / "clip.srt"
    assert output_path(str(out), media) == out / "clip.srt"
    assert output_path(f"{out}/", media) == out / "clip.srt"
    assert output_path(str(out / "named.srt"), media) == out / "named.srt"


def test_no_cue_survives_into_the_next_one(tmp_path):
    # Word times routinely overrun the next word's start, so cues built from them
    # overlap unless something says otherwise.
    doc = transcript.parse(write(tmp_path, "A:\nこれはテストです\n\nB:\nもう一度、テストです\n\nA:\n三回目です\n"))
    t = Times.empty(len(doc.stream), "test")
    for i in range(len(doc.stream)):
        t.set(i, i + 1, i * 0.2, i * 0.2 + 1.5, 0.9)   # every word overruns by 1.3s
    t.fill_gaps()
    built = cues.build(doc, t)
    assert sum(1 for a, b in zip(built, built[1:]) if b.start < a.end), "fixture must overlap"
    cues.enforce_gaps(built)
    assert not [1 for a, b in zip(built, built[1:]) if b.start < a.end]
    assert all(c.end > c.start for c in built)


def _pair(tmp_path, media_name="clip.mov"):
    text = tmp_path / "transcript.txt"
    text.write_text("A:\nこんにちは\n", encoding="utf-8")
    media = tmp_path / media_name
    media.write_bytes(b"\x00\x00\x00\x18ftypqt  " + b"\x00" * 64)
    return text, media


def test_resolve_returns_recording_and_transcript(tmp_path):
    from txt2srt.inputs import resolve

    text, media = _pair(tmp_path)
    assert resolve(text=str(text), audio=str(media)) == (media, text)


def test_resolve_names_the_missing_flag(tmp_path):
    from txt2srt.inputs import InputError, resolve

    text, media = _pair(tmp_path)
    with pytest.raises(InputError, match="missing -a"):
        resolve(text=str(text), audio=None)
    with pytest.raises(InputError, match="missing -t"):
        resolve(text=None, audio=str(media))
    with pytest.raises(InputError, match="missing -t and -a"):
        resolve(text=None, audio=None)


def test_a_flag_pointing_at_the_wrong_kind_of_file(tmp_path):
    from txt2srt.inputs import InputError, resolve

    text, media = _pair(tmp_path)
    with pytest.raises(InputError, match="not UTF-8 text"):
        resolve(text=str(media), audio=str(media))
    with pytest.raises(InputError, match="no such file"):
        resolve(text=str(text), audio=str(tmp_path / "gone.mov"))


def test_overlap_is_fixed_by_trimming_the_earlier_cue(tmp_path):
    doc = transcript.parse(write(tmp_path, "A:\nこれはテストです\n\nB:\nもう一度、テストです\n\nA:\n三回目です\n"))
    t = Times.empty(len(doc.stream), "test")
    for i in range(len(doc.stream)):
        t.set(i, i + 1, i * 0.2, i * 0.2 + 1.5, 0.9)   # every word overruns by 1.3s
    t.fill_gaps()
    built = cues.build(doc, t)
    starts = [c.start for c in built]
    assert sum(1 for a, b in zip(built, built[1:]) if b.start < a.end), "fixture must overlap"

    cues.enforce_gaps(built)
    assert not [1 for a, b in zip(built, built[1:]) if b.start < a.end]
    assert all(c.end > c.start for c in built)
    # The onsets are the part worth keeping: ends moved, starts did not.
    assert [c.start for c in built] == starts


def test_bare_invocation_prints_usage(capsys):
    from txt2srt.cli import main

    assert main([]) == 2
    out = capsys.readouterr().out
    assert "usage: txt2srt" in out
    assert "-t" in out and "-a" in out


def tone(seconds: float, sr: int = 16000):
    """Something the VAD will call speech: a voiced buzz, not white noise."""
    t = np.arange(int(seconds * sr)) / sr
    f0 = 120.0
    wave = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)       # syllable-rate modulation
    return (0.3 * wave * env).astype(np.float32)


def quiet(seconds: float, sr: int = 16000):
    return np.random.default_rng(0).normal(0, 1e-4, int(seconds * sr)).astype(np.float32)


def test_the_vad_finds_speech_and_nothing_else():
    audio = np.concatenate([quiet(2.0), tone(2.0), quiet(2.0)])
    heard = speech.detect(audio)
    assert heard.fraction_between(0.0, 1.8) == 0.0
    assert heard.fraction_between(2.3, 3.7) > 0.9
    assert heard.fraction_between(4.5, 6.0) == 0.0
    assert 1.8 < heard.onsets[0] < 2.4
    assert 3.8 < heard.offsets[0] < 4.4


def test_a_word_does_not_keep_the_pause_in_front_of_it(tmp_path):
    """DTW charges a pause to the word after it. The word's end is right, its
    start is wherever the silence began, and the cue follows the start."""
    doc = transcript.parse(write(tmp_path, "A:\nあい\n"))
    audio = np.concatenate([quiet(10.0), tone(1.0)])
    heard = speech.detect(audio)
    t = Times.empty(len(doc.stream), "test")
    t.set(0, 1, 0.2, 10.5, 0.9)          # 'あ' handed the whole 10s pause
    t.set(1, 2, 10.5, 10.9, 0.9)
    assert clamp_spans(t, doc, heard) == 1
    assert t.end[0] == 10.5              # the end is where it was really said
    assert 9.5 < t.start[0] <= 10.45     # the start is now the speech onset
    t.fill_gaps()
    assert cues.build(doc, t)[0].start > 9.0


def test_a_word_spoken_slowly_is_left_alone(tmp_path):
    doc = transcript.parse(write(tmp_path, "A:\nあい\n"))
    heard = speech.detect(np.concatenate([quiet(0.2), tone(2.0)]))
    t = Times.empty(len(doc.stream), "test")
    t.set(0, 1, 0.3, 1.1, 0.9)
    t.set(1, 2, 1.1, 1.9, 0.9)
    assert clamp_spans(t, doc, heard) == 0
    assert t.start[0] == 0.3


def test_an_onset_is_not_trusted_past_what_a_word_could_take_to_say(tmp_path):
    """A transcript is not an account of every sound in the room. Where the
    speech a word ends inside began ten seconds earlier, because somebody
    untranscribed was talking, the rate backstop is what bounds the cue."""
    doc = transcript.parse(write(tmp_path, "A:\nあい\n"))
    heard = speech.detect(np.concatenate([quiet(0.5), tone(10.0)]))
    t = Times.empty(len(doc.stream), "test")
    t.set(0, 1, 0.6, 10.2, 0.9)
    t.set(1, 2, 10.2, 10.4, 0.9)
    assert clamp_spans(t, doc, heard) == 1
    assert t.start[0] > 8.5              # not the onset at 0.5s
