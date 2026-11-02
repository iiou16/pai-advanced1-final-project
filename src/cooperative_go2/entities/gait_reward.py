"""歩行方策の報酬の計算。

速度指令への追従を正の報酬とし、上下動・胴体の高さのずれ・行動の急変・既定姿勢からの逸脱を減点する。
すべて並列環境のバッチ (num_envs, ...) のテンソルのまま計算する。
"""

from dataclasses import dataclass, fields

import torch

from cooperative_go2.entities.gait_action import GaitActionHistory
from cooperative_go2.entities.gait_config import GaitRewardConfig, GaitRewardScales
from cooperative_go2.entities.go2_state import Go2State


@dataclass(frozen=True)
class GaitRewardInputs:
    """報酬の計算に使う 1 制御周期分の材料。

    Attributes:
        state: このステップの物理計算後（リセット前）の状態。
        commands: 速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
        actions: このステップの行動と 1 つ前の行動（リセット前）。
        default_dof_pos: 既定姿勢の関節角 (12,) [rad]。
    """

    state: Go2State
    commands: torch.Tensor
    actions: GaitActionHistory
    default_dof_pos: torch.Tensor


@dataclass(frozen=True)
class GaitRewardTerms:
    """報酬の項ごとの値。各フィールドは (num_envs,) のテンソル。フィールド名は GaitRewardScales と一致する。"""

    tracking_lin_vel: torch.Tensor
    tracking_ang_vel: torch.Tensor
    lin_vel_z: torch.Tensor
    base_height: torch.Tensor
    action_rate: torch.Tensor
    similar_to_default: torch.Tensor

    def weighted(self, scales: GaitRewardScales, control_dt_s: float) -> "GaitRewardTerms":
        """各項に「係数 × 制御周期」を掛けた項を返す。"""
        return GaitRewardTerms(
            **{
                field.name: getattr(self, field.name) * getattr(scales, field.name) * control_dt_s
                for field in fields(self)
            }
        )

    def total(self) -> torch.Tensor:
        """全項の和 (num_envs,) を返す。"""
        return torch.stack([getattr(self, field.name) for field in fields(self)]).sum(dim=0)

    def named(self) -> dict[str, torch.Tensor]:
        """ログ出力用に、項の名前 → 値の対応を返す。"""
        return {field.name: getattr(self, field.name) for field in fields(self)}


GAIT_REWARD_TERM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(GaitRewardTerms))


def compute_gait_reward_terms(inputs: GaitRewardInputs, config: GaitRewardConfig) -> GaitRewardTerms:
    """係数を掛ける前の、報酬の各項の値を計算する。

    Args:
        inputs: 1 制御周期分の材料。
        config: 追従の幅と胴体の高さの目標。係数（scales）はここでは使わない。

    Returns:
        各項の値。追従の項は 0〜1（完全に追従して 1）、減点の項は 0 以上（理想で 0）。
    """
    state = inputs.state
    lin_vel_error = torch.sum(torch.square(inputs.commands[:, :2] - state.base_lin_vel[:, :2]), dim=1)
    yaw_rate_error = torch.square(inputs.commands[:, 2] - state.proprioception.base_ang_vel[:, 2])
    joint_offset = state.proprioception.dof_pos - inputs.default_dof_pos
    return GaitRewardTerms(
        tracking_lin_vel=torch.exp(-lin_vel_error / config.tracking_sigma),
        tracking_ang_vel=torch.exp(-yaw_rate_error / config.tracking_sigma),
        lin_vel_z=torch.square(state.base_lin_vel[:, 2]),
        base_height=torch.square(state.base_pos_m[:, 2] - config.base_height_target_m),
        action_rate=torch.sum(torch.square(inputs.actions.previous - inputs.actions.current), dim=1),
        similar_to_default=torch.sum(torch.abs(joint_offset), dim=1),
    )
