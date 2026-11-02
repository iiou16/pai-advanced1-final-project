"""CLI が入力の不正を終了コードに変換することの確認。いずれも Genesis を初期化する前に失敗する経路だけを扱う。"""

import shutil
from pathlib import Path

import pytest

from cooperative_go2.frameworks import evaluate_gait, train_gait

def test_training_with_missing_config_exits_with_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc_info:
        train_gait.main(["--config", str(tmp_path / "absent.yaml"), "--run-name", "unused"])

    assert exc_info.value.code == train_gait.EXIT_INVALID_CONFIG


def test_evaluation_with_training_seed_exits_with_invalid_input(
    tmp_path: Path, gait_config_path: Path, gait_config_data: dict
) -> None:
    shutil.copyfile(gait_config_path, tmp_path / "gait.yaml")
    training_seed = str(gait_config_data["training"]["seed"])
    arguments = [
        "--run-dir", str(tmp_path), "--checkpoint", "100", "--vx", "0.5", "--vy", "0.0", "--yaw-rate", "0.0",
        "--num-envs", "1", "--duration-s", "1.0", "--seed", training_seed,
    ]  # fmt: skip

    with pytest.raises(SystemExit) as exc_info:
        evaluate_gait.main(arguments)

    assert exc_info.value.code == evaluate_gait.EXIT_INVALID_INPUT
