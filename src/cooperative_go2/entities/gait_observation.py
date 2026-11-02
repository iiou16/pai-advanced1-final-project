"""歩行方策の観測ベクトルの構成。

学習済みの方策は観測の順序と正規化に依存する。順序を変えるとチェックポイントの互換性が失われる。
観測は自己受容の状態（Go2Proprioception）・速度指令・前回の行動だけから作るため、
シミュレーションでも ROS 2 の実行系でも同じ関数で作れる。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.gait_config import GaitObservationScales
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE

# 観測ベクトルを先頭から順に (成分名, 次元数) で並べたもの。関節の成分は GO2_JOINT_NAMES の順
GAIT_OBSERVATION_LAYOUT: tuple[tuple[str, int], ...] = (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_command", VELOCITY_COMMAND_SIZE),
    ("joint_pos_offset", NUM_GO2_JOINTS),
    ("joint_vel", NUM_GO2_JOINTS),
    ("last_action", NUM_GO2_JOINTS),
)

GAIT_OBSERVATION_DIM = sum(size for _, size in GAIT_OBSERVATION_LAYOUT)


@dataclass(frozen=True)
class GaitObservationInputs:
    """観測の材料。

    Attributes:
        proprioception: 自己受容の状態。
        commands: 速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
        last_actions: 直前に方策が出した行動 (num_envs, 12)。エピソードの最初は 0。
        default_dof_pos: 既定姿勢の関節角 (12,) [rad]。
    """

    proprioception: Go2Proprioception
    commands: torch.Tensor
    last_actions: torch.Tensor
    default_dof_pos: torch.Tensor


def build_gait_observation(inputs: GaitObservationInputs, scales: GaitObservationScales) -> torch.Tensor:
    """GAIT_OBSERVATION_LAYOUT の順に正規化した観測 (num_envs, GAIT_OBSERVATION_DIM) を作る。

    Args:
        inputs: 観測の材料。
        scales: 成分ごとの正規化係数。速度指令の vx・vy には lin_vel、yaw_rate には ang_vel を掛ける。
    """
    proprioception = inputs.proprioception
    command_scales = torch.tensor(
        [scales.lin_vel, scales.lin_vel, scales.ang_vel],
        dtype=inputs.commands.dtype,
        device=inputs.commands.device,
    )
    return torch.cat(
        (
            proprioception.base_ang_vel * scales.ang_vel,
            proprioception.projected_gravity,
            inputs.commands * command_scales,
            (proprioception.dof_pos - inputs.default_dof_pos) * scales.dof_pos,
            proprioception.dof_vel * scales.dof_vel,
            inputs.last_actions,
        ),
        dim=-1,
    )
