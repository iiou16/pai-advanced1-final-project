"""Go2 の 1 制御周期分の状態。テンソルは並列環境のバッチ (num_envs, ...)、関節は GO2_JOINT_NAMES の順。"""

from dataclasses import dataclass, fields, replace

import torch


def _replace_rows(current: torch.Tensor, replacement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    row_mask = mask.reshape(-1, *([1] * (current.dim() - 1)))
    return torch.where(row_mask, replacement, current)


@dataclass(frozen=True)
class Go2Proprioception:
    """実機でも IMU と関節エンコーダから得られる自己受容の状態。歩行方策の観測はこれだけから作る。

    Attributes:
        base_ang_vel: 胴体の角速度 (num_envs, 3)、本体座標系 [rad/s]。
        projected_gravity: 本体座標系で表した重力方向の単位ベクトル (num_envs, 3)。
        dof_pos: 関節角 (num_envs, 12) [rad]。
        dof_vel: 関節角速度 (num_envs, 12) [rad/s]。
    """

    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor

    def reset_where(self, mask: torch.Tensor, initial: "Go2Proprioception") -> "Go2Proprioception":
        """mask が True の環境の行だけを initial の値に置き換えた新しい状態を返す。"""
        return Go2Proprioception(
            **{
                field.name: _replace_rows(getattr(self, field.name), getattr(initial, field.name), mask)
                for field in fields(self)
            }
        )


@dataclass(frozen=True)
class Go2State:
    """シミュレーションで得られる Go2 の状態。報酬と終了判定に使う。

    Attributes:
        proprioception: 自己受容の状態。
        base_pos_m: 胴体の位置 (num_envs, 3)、ワールド座標系 [m]。
        base_euler_deg: 初期姿勢から見た胴体のロール・ピッチ・ヨー (num_envs, 3) [deg]。
        base_lin_vel: 胴体の線速度 (num_envs, 3)、本体座標系 [m/s]。
    """

    proprioception: Go2Proprioception
    base_pos_m: torch.Tensor
    base_euler_deg: torch.Tensor
    base_lin_vel: torch.Tensor

    def with_dof_pos(self, dof_pos: torch.Tensor) -> "Go2State":
        """関節角だけを dof_pos (num_envs, 12) [rad] に置き換えた新しい状態を返す。"""
        return replace(self, proprioception=replace(self.proprioception, dof_pos=dof_pos))

    def reset_where(self, mask: torch.Tensor, initial: "Go2State") -> "Go2State":
        """mask (num_envs,) が True の環境の行だけを initial の値に置き換えた新しい状態を返す。元の状態は変えない。"""
        return Go2State(
            proprioception=self.proprioception.reset_where(mask, initial.proprioception),
            base_pos_m=_replace_rows(self.base_pos_m, initial.base_pos_m, mask),
            base_euler_deg=_replace_rows(self.base_euler_deg, initial.base_euler_deg, mask),
            base_lin_vel=_replace_rows(self.base_lin_vel, initial.base_lin_vel, mask),
        )


def initial_go2_state(
    base_pos_m: torch.Tensor, projected_gravity: torch.Tensor, default_dof_pos: torch.Tensor
) -> Go2State:
    """リセット直後（既定姿勢で静止）の状態を、全環境に共通の 1 行 (1, ...) で返す。

    reset_where() の initial に渡すと、各環境の行にブロードキャストされる。

    Args:
        base_pos_m: リセット時の胴体の位置 (3,) [m]。
        projected_gravity: リセット時の姿勢で本体座標系に表した重力方向 (3,)。
        default_dof_pos: 既定姿勢の関節角 (12,) [rad]。
    """
    def zero_row_like(values: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(values).unsqueeze(0)

    return Go2State(
        proprioception=Go2Proprioception(
            base_ang_vel=zero_row_like(base_pos_m),
            projected_gravity=projected_gravity.unsqueeze(0),
            dof_pos=default_dof_pos.unsqueeze(0),
            dof_vel=zero_row_like(default_dof_pos),
        ),
        base_pos_m=base_pos_m.unsqueeze(0),
        base_euler_deg=zero_row_like(base_pos_m),
        base_lin_vel=zero_row_like(base_pos_m),
    )


def sample_reset_dof_pos(
    default_dof_pos: torch.Tensor, num_envs: int, noise_rad: float, generator: torch.Generator
) -> torch.Tensor:
    """リセット時の関節角 (num_envs, 12) [rad] を、既定姿勢に一様ノイズを足して引く。

    Args:
        default_dof_pos: 既定姿勢の関節角 (12,) [rad]。
        num_envs: 並列環境の数。
        noise_rad: ノイズの幅 [rad]。関節ごとに独立に [-noise_rad, noise_rad] から引く。0 なら既定姿勢そのもの。
        generator: ノイズを引く乱数生成器。テンソルはこの生成器のデバイスに置く。
    """
    shape = (num_envs, default_dof_pos.shape[0])
    uniform = torch.rand(shape, generator=generator, device=generator.device)
    return default_dof_pos + (2.0 * uniform - 1.0) * noise_rad
