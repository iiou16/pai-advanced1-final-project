"""CLI が失敗を終了コードに変換することの確認。Genesis の初期化は差し替えるか、その前に失敗する経路だけを扱う。"""

import shutil
from pathlib import Path

import pytest
import torch

from cooperative_go2.frameworks import evaluate_gait, train_gait


def test_training_with_missing_config_exits_with_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_gait.main(["--config", str(tmp_path / "absent.yaml"), "--run-name", "unused"])

    assert exc_info.value.code == train_gait.EXIT_INVALID_INPUT


def test_evaluation_with_training_seed_exits_with_invalid_input(
    tmp_path: Path, gait_config_path: Path, gait_config_data: dict, gait_eval_config_path: Path
) -> None:
    shutil.copyfile(gait_config_path, tmp_path / "gait.yaml")
    training_seed = str(gait_config_data["training"]["seed"])
    arguments = [
        "--run-dir", str(tmp_path), "--checkpoint", "100", "--scenarios", str(gait_eval_config_path),
        "--num-envs", "1", "--seed", training_seed,
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        evaluate_gait.main(arguments)

    assert exc_info.value.code == evaluate_gait.EXIT_INVALID_INPUT


def test_training_with_unsafe_run_name_exits_with_invalid_input(gait_config_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_gait.main(["--config", str(gait_config_path), "--run-name", "../escape"])

    assert exc_info.value.code == train_gait.EXIT_INVALID_INPUT


def test_training_into_existing_run_dir_exits_without_touching_it(
    tmp_path: Path, gait_config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_gait, "GAIT_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(train_gait, "init_genesis_on_cuda", lambda seed: None)
    monkeypatch.setattr(train_gait, "make_seeded_generator", lambda seed: torch.Generator().manual_seed(seed))
    previous_checkpoint = tmp_path / "existing" / "model_100.pt"
    previous_checkpoint.parent.mkdir()
    previous_checkpoint.write_bytes(b"previous")

    with pytest.raises(SystemExit) as exc_info:
        train_gait.main(["--config", str(gait_config_path), "--run-name", "existing"])

    assert exc_info.value.code == train_gait.EXIT_RUN_ALREADY_EXISTS
    assert previous_checkpoint.read_bytes() == b"previous"


def test_evaluation_with_invalid_scenarios_exits_with_invalid_input(tmp_path: Path, gait_config_path: Path) -> None:
    shutil.copyfile(gait_config_path, tmp_path / "gait.yaml")
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text("settle_time_s: .nan\nscenarios: []\n", encoding="utf-8")
    arguments = [
        "--run-dir", str(tmp_path), "--checkpoint", "100", "--scenarios", str(scenarios),
        "--num-envs", "1", "--seed", "1001",
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        evaluate_gait.main(arguments)

    assert exc_info.value.code == evaluate_gait.EXIT_INVALID_INPUT
