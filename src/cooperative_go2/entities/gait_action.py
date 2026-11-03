"""歩行方策の行動（既定姿勢からの 12 関節の残差）の履歴と、目標関節角への変換。"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GaitActionHistory:
    """直近 2 制御周期の行動 (num_envs, 12)。

    Attributes:
        current: 今回の制御周期に方策が出した行動（クリップ済み）。
        previous: 1 制御周期前の行動。
    """

    current: torch.Tensor
    previous: torch.Tensor

    @staticmethod
    def zeros(num_envs: int, num_joints: int, device: torch.device | str) -> "GaitActionHistory":
        """行動をまだ出していない状態を返す。"""
        shape = (num_envs, num_joints)
        return GaitActionHistory(current=torch.zeros(shape, device=device), previous=torch.zeros(shape, device=device))

    def advance(self, raw_actions: torch.Tensor, clip_limit: float) -> "GaitActionHistory":
        """方策の新しい行動を絶対値 clip_limit でクリップして current にし、これまでの current を previous にずらす。"""
        return GaitActionHistory(current=torch.clip(raw_actions, -clip_limit, clip_limit), previous=self.current)

    def applied(self, simulate_latency: bool) -> torch.Tensor:
        """ロボットに適用する行動を返す。遅延を模擬するなら 1 制御周期前の行動を使う。"""
        return self.previous if simulate_latency else self.current

    def reset_where(self, mask: torch.Tensor) -> "GaitActionHistory":
        """mask (num_envs,) が True の環境の行動の履歴を 0 にした新しい履歴を返す。"""
        row_mask = mask[:, None]
        return GaitActionHistory(
            current=self.current.masked_fill(row_mask, 0.0),
            previous=self.previous.masked_fill(row_mask, 0.0),
        )


def to_joint_targets(applied_actions: torch.Tensor, default_dof_pos: torch.Tensor, action_scale: float) -> torch.Tensor:
    """行動（残差）を PD 制御の目標関節角 (num_envs, 12) [rad] に変換する。

    Args:
        applied_actions: ロボットに適用する行動 (num_envs, 12)。
        default_dof_pos: 既定姿勢の関節角 (12,) [rad]。
        action_scale: 行動 1 あたりの関節角 [rad]。
    """
    return applied_actions * action_scale + default_dof_pos
