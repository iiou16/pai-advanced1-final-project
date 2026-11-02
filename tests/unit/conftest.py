from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State

REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_CONFIG_PATH = REPO_ROOT / "configs" / "gait.yaml"

Go2StateFactory = Callable[..., Go2State]


@pytest.fixture
def gait_config_path() -> Path:
    """リポジトリの configs/gait.yaml。値の検査には使わず、読み込めることの確認だけに使う。"""
    return GAIT_CONFIG_PATH


@pytest.fixture
def gait_config_data() -> dict:
    """configs/gait.yaml を読み込んだ dict。テストごとに新しく読み込むので書き換えてよい。"""
    return yaml.safe_load(GAIT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def make_go2_state() -> Go2StateFactory:
    """すべて 0 の状態を基準に、指定したフィールドだけ差し替えた Go2State を作る関数。"""

    def factory(num_envs: int = 1, **overrides: torch.Tensor) -> Go2State:
        proprioception_fields = {"base_ang_vel", "projected_gravity", "dof_pos", "dof_vel"}
        proprioception = Go2Proprioception(
            base_ang_vel=overrides.get("base_ang_vel", torch.zeros(num_envs, 3)),
            projected_gravity=overrides.get("projected_gravity", torch.zeros(num_envs, 3)),
            dof_pos=overrides.get("dof_pos", torch.zeros(num_envs, 12)),
            dof_vel=overrides.get("dof_vel", torch.zeros(num_envs, 12)),
        )
        unknown = set(overrides) - proprioception_fields - {"base_pos_m", "base_euler_deg", "base_lin_vel"}
        if unknown:
            raise TypeError(f"Go2State に無いフィールド: {sorted(unknown)}")
        return Go2State(
            proprioception=proprioception,
            base_pos_m=overrides.get("base_pos_m", torch.zeros(num_envs, 3)),
            base_euler_deg=overrides.get("base_euler_deg", torch.zeros(num_envs, 3)),
            base_lin_vel=overrides.get("base_lin_vel", torch.zeros(num_envs, 3)),
        )

    return factory
