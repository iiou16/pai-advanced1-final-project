"""別々に届く同じ時刻の状態を組にする部品の検証。"""

import pytest

from cooperative_go2.use_cases.stamp_aligner import StampAligner, UnknownSourceError

SOURCES = ("a", "b")


@pytest.fixture
def aligner() -> StampAligner[str]:
    return StampAligner(SOURCES, max_pending_stamps=3)


def test_returns_none_until_all_sources_arrive(aligner: StampAligner[str]) -> None:
    aligner.put("a", 10, "a10")

    assert aligner.pop_latest_complete_or_none() is None


def test_returns_latest_complete_stamp_and_drops_older(aligner: StampAligner[str]) -> None:
    aligner.put("a", 10, "a10")
    aligner.put("b", 10, "b10")
    aligner.put("b", 20, "b20")
    aligner.put("a", 20, "a20")
    aligner.put("a", 30, "a30")

    aligned = aligner.pop_latest_complete_or_none()

    assert aligned is not None
    assert aligned.stamp_ns == 20
    assert aligned.values == {"a": "a20", "b": "b20"}
    assert aligner.pop_latest_complete_or_none() is None
    aligner.put("b", 30, "b30")
    later = aligner.pop_latest_complete_or_none()
    assert later is not None and later.stamp_ns == 30


def test_pending_stamps_beyond_limit_are_dropped_oldest_first(aligner: StampAligner[str]) -> None:
    for stamp in (10, 20, 30, 40):
        aligner.put("a", stamp, f"a{stamp}")
    aligner.put("b", 10, "b10")

    assert aligner.pop_latest_complete_or_none() is None


def test_unknown_source_is_rejected(aligner: StampAligner[str]) -> None:
    with pytest.raises(UnknownSourceError):
        aligner.put("c", 10, "c10")
