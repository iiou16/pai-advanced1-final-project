import csv
import json
from pathlib import Path

import pytest

from cooperative_go2.entities.gait_evaluation import ScenarioSummary, SegmentTrackingSummary
from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.interface_adapters.gait_evaluation_store import (
    SCENARIOS_COPY_FILENAME,
    SEGMENTS_FILENAME,
    SUMMARY_FILENAME,
    EvaluationOutputExistsError,
    FileGaitEvaluationStore,
    GaitEvaluationRecord,
    evaluation_output_dir,
)


def _segment(command: VelocityCommand, fall_rate: float) -> SegmentTrackingSummary:
    return SegmentTrackingSummary(
        command=command,
        duration_s=4.0,
        num_envs=2,
        num_tracking_samples=300,
        mean_lin_vel_error_mps=0.1,
        mean_yaw_rate_error_radps=0.2,
        num_falls=1,
        fall_rate=fall_rate,
    )


def _record() -> GaitEvaluationRecord:
    stop = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=0.0)
    turn = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.0)
    scenario = ScenarioSummary(name="change", segments=(_segment(stop, 0.0), _segment(turn, 0.5)), fall_rate=0.5)
    return GaitEvaluationRecord(
        run_dir="outputs/gait/run", checkpoint_iteration=1500, seed=1001, num_envs=2, settle_time_s=1.0,
        scenarios=(scenario,),
    )  # fmt: skip


@pytest.fixture
def store(tmp_path: Path, gait_eval_config_path: Path) -> FileGaitEvaluationStore:
    return FileGaitEvaluationStore(tmp_path / "eval" / "model_1500_seed_1001", gait_eval_config_path)


def test_output_dir_is_named_by_checkpoint_and_seed(tmp_path: Path) -> None:
    assert evaluation_output_dir(tmp_path, 1500, 1001) == tmp_path / "eval" / "model_1500_seed_1001"


def test_initialize_copies_scenarios(store, tmp_path: Path, gait_eval_config_path: Path) -> None:
    store.initialize()

    copied = tmp_path / "eval" / "model_1500_seed_1001" / SCENARIOS_COPY_FILENAME
    assert copied.read_text(encoding="utf-8") == gait_eval_config_path.read_text(encoding="utf-8")


def test_existing_output_is_not_overwritten(store, tmp_path: Path) -> None:
    output_dir = tmp_path / "eval" / "model_1500_seed_1001"
    output_dir.mkdir(parents=True)
    previous = output_dir / SUMMARY_FILENAME
    previous.write_text("previous", encoding="utf-8")

    with pytest.raises(EvaluationOutputExistsError):
        store.initialize()
    assert previous.read_text(encoding="utf-8") == "previous"


def test_save_writes_summary_and_one_csv_row_per_segment(store, tmp_path: Path) -> None:
    store.initialize()

    store.save(_record())

    output_dir = tmp_path / "eval" / "model_1500_seed_1001"
    summary = json.loads((output_dir / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    assert summary["seed"] == 1001
    assert summary["scenarios"][0]["segments"][1]["command"]["yaw_rate_radps"] == pytest.approx(1.0)
    with (output_dir / SEGMENTS_FILENAME).open(encoding="utf-8", newline="") as segments_file:
        rows = list(csv.DictReader(segments_file))
    assert [(row["scenario"], row["segment_index"], row["yaw_rate_radps"]) for row in rows] == [
        ("change", "0", "0.0"),
        ("change", "1", "1.0"),
    ]
    assert rows[1]["scenario_fall_rate"] == "0.5"


def test_video_path_is_named_by_scenario(store, tmp_path: Path) -> None:
    assert store.video_path("stop") == tmp_path / "eval" / "model_1500_seed_1001" / "stop.mp4"
