import numpy as np
import pytest

from txt2srt import cues, transcript
from txt2srt.backends import Times
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


def test_report_counts_long_cues_against_the_limit_it_was_given(tmp_path):
    """--max-dur says how long a cue may be. A report that grades against some
    other number is answering a question nobody asked."""
    from txt2srt import report

    doc = transcript.parse(write(tmp_path, "A:\nあいうえお\n"))
    t = Times.empty(len(doc.stream), "test")
    t.set(0, len(doc.stream), 0.0, 5.0, 0.9)
    t.fill_gaps()
    built = cues.build(doc, t, max_dur=99.0)
    audio = np.zeros(16000 * 10, dtype=np.float32)
    assert report.build(audio, built, t, max_dur=6.0)["over_max_dur"] == 0
    assert report.build(audio, built, t, max_dur=4.0)["over_max_dur"] == 1
