"""ROS 2 の実行系の計測値の集計と書き出しの検証。"""

import json
from pathlib import Path

import pytest

from cooperative_go2.interface_adapters.ros2_metrics import LatencyStats, RoundTripLatencyProbe, write_node_metrics


def test_empty_stats_report_zero_count_and_no_values() -> None:
    assert LatencyStats().summary() == {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}


def test_single_sample_is_its_own_percentile() -> None:
    stats = LatencyStats()
    stats.add(0.004)

    summary = stats.summary()

    assert summary["count"] == 1
    assert summary["p95_ms"] == pytest.approx(4.0)
    assert summary["max_ms"] == pytest.approx(4.0)


def test_stats_summarize_in_milliseconds() -> None:
    stats = LatencyStats()
    for milliseconds in range(1, 101):
        stats.add(milliseconds / 1000)

    summary = stats.summary()

    assert summary["mean_ms"] == pytest.approx(50.5)
    assert summary["median_ms"] == pytest.approx(50.5)
    assert 94.0 <= summary["p95_ms"] <= 96.0
    assert summary["max_ms"] == pytest.approx(100.0)


def test_round_trip_is_measured_from_publish_to_matching_target() -> None:
    now = [0.0]
    probe = RoundTripLatencyProbe(wall_clock=lambda: now[0])
    probe.on_published(100)
    now[0] = 0.012
    probe.on_target_received(100)
    probe.on_target_received(50)

    summary = probe.round_trip.summary()

    assert summary["count"] == 1
    assert summary["mean_ms"] == pytest.approx(12.0)


def test_metrics_are_written_under_existing_run_directory(tmp_path: Path) -> None:
    path = tmp_path / "nodes" / "metrics.json"

    write_node_metrics(path, {"value": 1})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_existing_metrics_are_not_overwritten(tmp_path: Path) -> None:
    path = tmp_path / "nodes" / "metrics.json"
    write_node_metrics(path, {"value": 1})

    with pytest.raises(FileExistsError):
        write_node_metrics(path, {"value": 2})

    assert json.loads(path.read_text(encoding="utf-8")) == {"value": 1}


def test_metrics_are_not_written_without_run_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        write_node_metrics(tmp_path / "missing_run" / "nodes" / "metrics.json", {"value": 1})
