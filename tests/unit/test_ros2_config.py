"""ROS 2 の実行系の設定ファイル（configs/ros2_push_single.yaml）の読み込みと検証。"""

from pathlib import Path

import pytest
import yaml

from cooperative_go2.interface_adapters.config_loader import ConfigValidationError, load_ros2_push_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "ros2_push_single.yaml"


def test_repository_ros2_config_is_valid() -> None:
    settings = load_ros2_push_settings(CONFIG_PATH)

    assert settings.runtime.torch_num_threads >= 1
    assert settings.command_expiry.joint_target_s > 0


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runtime", "torch_num_threads", 0),
        ("runtime", "max_real_time_factor", -1.0),
        ("command_expiry", "velocity_command_s", 0.0),
    ],
    ids=["zero_threads", "negative_rtf", "zero_expiry"],
)
def test_invalid_value_is_rejected(tmp_path: Path, section: str, key: str, value: object) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    data[section][key] = value
    path = tmp_path / "ros2.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_ros2_push_settings(path)


def test_missing_key_is_rejected(tmp_path: Path) -> None:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    del data["runtime"]["torch_num_threads"]
    path = tmp_path / "ros2.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")

    with pytest.raises(ConfigValidationError):
        load_ros2_push_settings(path)
