"""CLI が失敗を終了コードに変換することの確認。Genesis の初期化は差し替えるか、その前に失敗する経路だけを扱う。"""

import shutil
from pathlib import Path

import pytest
import torch

from conftest import REPO_ROOT

from cooperative_go2.frameworks import evaluate_gait, evaluate_team_marl, train_gait, train_team_marl

PUSH_MULTI_CONFIG = REPO_ROOT / "configs" / "push_multi.yaml"


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


def test_team_marl_training_with_missing_config_exits_with_invalid_input(tmp_path: Path) -> None:
    arguments = [
        "--config", str(tmp_path / "absent.yaml"), "--algorithm", "mappo", "--parameter-sharing", "shared",
        "--run-name", "unused",
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        train_team_marl.main(arguments)

    assert exc_info.value.code == train_team_marl.EXIT_INVALID_INPUT


def test_team_marl_training_into_existing_run_dir_exits_without_touching_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(train_team_marl, "TEAM_MARL_RUNS_ROOT", tmp_path)
    monkeypatch.setattr(train_team_marl, "init_genesis_on_cuda", lambda seed: None)
    previous = tmp_path / "existing" / "variant.json"
    previous.parent.mkdir()
    previous.write_text("previous", encoding="utf-8")
    arguments = [
        "--config", str(PUSH_MULTI_CONFIG), "--algorithm", "ippo", "--parameter-sharing", "separate",
        "--run-name", "existing",
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        train_team_marl.main(arguments)

    assert exc_info.value.code == train_team_marl.EXIT_RUN_ALREADY_EXISTS
    assert previous.read_text(encoding="utf-8") == "previous"


def test_team_marl_evaluation_with_training_seed_exits_with_invalid_input(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    shutil.copyfile(PUSH_MULTI_CONFIG, run_dir / "push_multi.yaml")
    (run_dir / "variant.json").write_text('{"algorithm": "mappo", "shared_parameters": true}', encoding="utf-8")
    arguments = [
        "--run-dir", str(run_dir), "--iteration", "50", "--num-envs", "1", "--seed", "21",
        "--drop-probability", "0.0", "--device", "cpu", "--output-dir", str(tmp_path / "eval"),
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        evaluate_team_marl.main(arguments)

    assert exc_info.value.code == evaluate_team_marl.EXIT_INVALID_INPUT
    assert not (tmp_path / "eval").exists()
