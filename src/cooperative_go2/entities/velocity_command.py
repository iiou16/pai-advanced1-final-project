"""歩行方策に与える速度指令。"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.gait_config import VelocityCommandConfig

VELOCITY_COMMAND_SIZE = 3
# 指令テンソルの列のうち、線速度 (vx, vy) とヨー角速度の位置
LIN_VEL_COLUMNS = slice(0, 2)
YAW_RATE_COLUMN = 2


@dataclass(frozen=True)
class VelocityCommand:
    """本体座標系の速度指令。

    Attributes:
        vx_mps: 前後方向の速度 [m/s]。前が正。
        vy_mps: 左右方向の速度 [m/s]。左が正。
        yaw_rate_radps: ヨー角速度 [rad/s]。上から見て反時計回りが正。
    """

    vx_mps: float
    vy_mps: float
    yaw_rate_radps: float

    def as_tuple(self) -> tuple[float, float, float]:
        """(vx, vy, yaw_rate) の順のタプルを返す。指令テンソルの列の順序と同じ。"""
        return (self.vx_mps, self.vy_mps, self.yaw_rate_radps)


class VelocityCommandSampler:
    """並列環境ごとの速度指令 (num_envs, 3) を保持し、指定した環境だけ引き直す。

    既定では設定の範囲から一様に引き、設定の確率で線速度とヨー角速度をそれぞれ 0 にする。
    fix() で全環境の指令を固定値にできる（評価用）。
    """

    def __init__(self, config: VelocityCommandConfig, num_envs: int, generator: torch.Generator) -> None:
        """全環境の指令を 0 で初期化する。最初の resample() までは静止の指令になる。

        Args:
            config: 指令の範囲。
            num_envs: 並列環境の数。
            generator: 指令を引く乱数生成器。再現性のため、呼び出し側でシードを固定して渡す。テンソルはこの生成器のデバイスに置く。
        """
        ranges = (config.lin_vel_x_mps, config.lin_vel_y_mps, config.yaw_rate_radps)
        self._generator = generator
        device = generator.device
        self._low = torch.tensor([value_range.low for value_range in ranges], device=device)
        self._high = torch.tensor([value_range.high for value_range in ranges], device=device)
        self._zero_lin_vel_probability = config.zero_lin_vel_probability
        self._zero_yaw_rate_probability = config.zero_yaw_rate_probability
        self._fixed_command: torch.Tensor | None = None
        self._commands = torch.zeros((num_envs, VELOCITY_COMMAND_SIZE), device=device)

    @property
    def commands(self) -> torch.Tensor:
        """現在の指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。"""
        return self._commands

    def fix(self, command: VelocityCommand) -> None:
        """全環境の指令を直ちに command に置き換え、以後に引き直す指令も command に固定する。"""
        self._fixed_command = torch.tensor(command.as_tuple(), device=self._commands.device)
        self._commands = self._fixed_command.expand_as(self._commands).clone()

    def resample(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境だけ指令を引き直す。"""
        if self._fixed_command is not None:
            new_commands = self._fixed_command.expand_as(self._commands)
        else:
            new_commands = self._sample_uniform_with_zeros()
        self._commands = torch.where(mask[:, None], new_commands, self._commands)

    def _sample_uniform_with_zeros(self) -> torch.Tensor:
        device = self._commands.device
        num_envs = self._commands.shape[0]
        uniform = torch.rand(self._commands.shape, generator=self._generator, device=device)
        commands = self._low + (self._high - self._low) * uniform
        zero_draws = torch.rand((num_envs, 2), generator=self._generator, device=device)
        zero_lin_vel = zero_draws[:, 0] < self._zero_lin_vel_probability
        zero_yaw_rate = zero_draws[:, 1] < self._zero_yaw_rate_probability
        commands[:, LIN_VEL_COLUMNS] = commands[:, LIN_VEL_COLUMNS].masked_fill(zero_lin_vel[:, None], 0.0)
        commands[:, YAW_RATE_COLUMN] = commands[:, YAW_RATE_COLUMN].masked_fill(zero_yaw_rate, 0.0)
        return commands
