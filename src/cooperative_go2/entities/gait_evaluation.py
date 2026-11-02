"""固定した速度指令に対する歩行方策の追従誤差と転倒の集計。"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.velocity_command import VelocityCommand


@dataclass(frozen=True)
class GaitTrackingSummary:
    """1 つの速度指令で全環境を走らせた結果。

    Attributes:
        command: 与えた速度指令。
        num_envs: 並列に走らせた環境の数。
        num_steps: 制御ステップ数。
        mean_lin_vel_error_mps: 水平速度 (vx, vy) の指令との差のノルムの、全環境・全ステップでの平均 [m/s]。
        mean_yaw_rate_error_radps: ヨー角速度の指令との差の絶対値の、全環境・全ステップでの平均 [rad/s]。
        num_falls: 全環境で転倒した回数の合計。転倒した環境は初期姿勢に戻って続けるため、1 環境で複数回数えうる。
        fall_rate: 1 回以上転倒した環境の割合（0〜1）。
    """

    command: VelocityCommand
    num_envs: int
    num_steps: int
    mean_lin_vel_error_mps: float
    mean_yaw_rate_error_radps: float
    num_falls: int
    fall_rate: float


class GaitTrackingAccumulator:
    """制御ステップごとの速度と転倒を積算し、GaitTrackingSummary にまとめる。"""

    def __init__(self, command: VelocityCommand, num_envs: int, device: torch.device | str) -> None:
        self._command = command
        self._num_envs = num_envs
        self._command_xy = torch.tensor([command.vx_mps, command.vy_mps], device=device)
        self._lin_vel_error_sum = torch.zeros((), device=device)
        self._yaw_rate_error_sum = torch.zeros((), device=device)
        self._falls_per_env = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._num_steps = 0

    def add_step(self, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor, fallen: torch.Tensor) -> None:
        """1 制御ステップ分の観測を積算する。

        Args:
            base_lin_vel: 胴体の線速度 (num_envs, 3)、本体座標系 [m/s]。
            base_ang_vel: 胴体の角速度 (num_envs, 3)、本体座標系 [rad/s]。
            fallen: このステップで転倒した環境を True とするマスク (num_envs,)。
        """
        self._lin_vel_error_sum += torch.linalg.vector_norm(base_lin_vel[:, :2] - self._command_xy, dim=1).sum()
        self._yaw_rate_error_sum += torch.abs(base_ang_vel[:, 2] - self._command.yaw_rate_radps).sum()
        self._falls_per_env += fallen.to(torch.int64)
        self._num_steps += 1

    def summarize(self) -> GaitTrackingSummary:
        """積算した値を集計する。

        Raises:
            ValueError: 1 ステップも積算していない。
        """
        if self._num_steps == 0:
            raise ValueError("集計するステップが無い。add_step を 1 回以上呼ぶ必要がある")
        num_samples = self._num_steps * self._num_envs
        return GaitTrackingSummary(
            command=self._command,
            num_envs=self._num_envs,
            num_steps=self._num_steps,
            mean_lin_vel_error_mps=float(self._lin_vel_error_sum) / num_samples,
            mean_yaw_rate_error_radps=float(self._yaw_rate_error_sum) / num_samples,
            num_falls=int(self._falls_per_env.sum()),
            fall_rate=float((self._falls_per_env > 0).sum()) / self._num_envs,
        )
