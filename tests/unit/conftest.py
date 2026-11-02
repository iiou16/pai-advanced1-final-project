from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State, initial_go2_state
from cooperative_go2.interface_adapters.config_loader import GaitSettings, parse_gait_settings

REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_CONFIG_PATH = REPO_ROOT / "configs" / "gait.yaml"
GAIT_EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "gait_eval.yaml"

Go2StateFactory = Callable[..., Go2State]


@pytest.fixture
def gait_config_path() -> Path:
    """リポジトリの configs/gait.yaml。値の検査には使わず、読み込めることの確認だけに使う。"""
    return GAIT_CONFIG_PATH


@pytest.fixture
def gait_eval_config_path() -> Path:
    """リポジトリの configs/gait_eval.yaml（評価シナリオ）。"""
    return GAIT_EVAL_CONFIG_PATH


@pytest.fixture
def gait_config_data() -> dict:
    """configs/gait.yaml を読み込んだ dict。テストごとに新しく読み込むので書き換えてよい。"""
    return yaml.safe_load(GAIT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def make_go2_state() -> Go2StateFactory:
    """すべて 0 の状態を基準に、指定したフィールドだけ差し替えた Go2State を作る関数。"""

    def factory(num_envs: int = 1, **overrides: torch.Tensor) -> Go2State:
        known_fields = {
            "base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "base_pos_m", "base_euler_deg", "base_lin_vel",
        }  # fmt: skip
        unknown = set(overrides) - known_fields
        if unknown:
            raise TypeError(f"Go2State に無いフィールド: {sorted(unknown)}")
        proprioception = Go2Proprioception(
            base_ang_vel=overrides.get("base_ang_vel", torch.zeros(num_envs, 3)),
            projected_gravity=overrides.get("projected_gravity", torch.zeros(num_envs, 3)),
            dof_pos=overrides.get("dof_pos", torch.zeros(num_envs, 12)),
            dof_vel=overrides.get("dof_vel", torch.zeros(num_envs, 12)),
        )
        return Go2State(
            proprioception=proprioception,
            base_pos_m=overrides.get("base_pos_m", torch.zeros(num_envs, 3)),
            base_euler_deg=overrides.get("base_euler_deg", torch.zeros(num_envs, 3)),
            base_lin_vel=overrides.get("base_lin_vel", torch.zeros(num_envs, 3)),
        )

    return factory


SMALL_NUM_ENVS = 2


@pytest.fixture
def small_gait_settings(gait_config_data: dict) -> GaitSettings:
    """2 環境・エピソード 5 ステップ（0.1 s）・前進 0.5 m/s 固定の指令・リセット時のノイズなしにした歩行の設定。"""
    gait_config_data["env"]["num_envs"] = SMALL_NUM_ENVS
    gait_config_data["env"]["control_dt_s"] = 0.02
    gait_config_data["env"]["episode_length_s"] = 0.1
    gait_config_data["command"]["resampling_time_s"] = 0.1
    gait_config_data["command"]["lin_vel_x_mps"] = {"low": 0.5, "high": 0.5}
    gait_config_data["command"]["lin_vel_y_mps"] = {"low": 0.0, "high": 0.0}
    gait_config_data["command"]["yaw_rate_radps"] = {"low": 0.0, "high": 0.0}
    gait_config_data["command"]["zero_lin_vel_probability"] = 0.0
    gait_config_data["command"]["zero_yaw_rate_probability"] = 0.0
    gait_config_data["env"]["reset_joint_pos_noise_rad"] = 0.0
    return parse_gait_settings(gait_config_data)


class FakeGaitSimulation:
    """GaitSimulation の偽物。read_state() は next_state、diverged_mask() は next_diverged を返す。"""

    def __init__(self, num_envs: int) -> None:
        self.default_dof_pos = torch.zeros(12)
        self.initial_state = initial_go2_state(
            base_pos_m=torch.tensor([0.0, 0.0, 0.3]),
            projected_gravity=torch.tensor([0.0, 0.0, -1.0]),
            default_dof_pos=self.default_dof_pos,
        )
        self.next_state = self.initial_state.reset_where(torch.ones(num_envs, dtype=torch.bool), self.initial_state)
        self.next_diverged = torch.zeros(num_envs, dtype=torch.bool)
        self.applied_targets: list[torch.Tensor] = []
        self.reset_masks: list[torch.Tensor] = []
        self.reset_dof_pos: list[torch.Tensor] = []
        self.num_steps = 0

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        self.applied_targets.append(joint_targets)

    def step(self) -> None:
        self.num_steps += 1

    def read_state(self) -> Go2State:
        return self.next_state

    def diverged_mask(self) -> torch.Tensor:
        return self.next_diverged

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor) -> None:
        self.reset_masks.append(mask.clone())
        self.reset_dof_pos.append(dof_pos.clone())


@pytest.fixture
def fake_simulation() -> FakeGaitSimulation:
    return FakeGaitSimulation(SMALL_NUM_ENVS)
