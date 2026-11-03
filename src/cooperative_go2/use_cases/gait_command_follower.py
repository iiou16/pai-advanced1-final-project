"""ROS 2 の歩行方策ノードの中身。受け取った速度指令に、期限切れなら停止の指令に切り替えて歩行方策で追従する。"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.command_expiry import is_command_fresh
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.use_cases.gait_controller import GaitController


@dataclass(frozen=True)
class StampedVelocityCommand:
    """時刻付きの速度指令。

    Attributes:
        stamp_ns: 指令を決めた時点のシミュレーション時刻 [ns]。
        values: 速度指令 (1, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]（本体座標系）。
    """

    stamp_ns: int
    values: torch.Tensor


@dataclass(frozen=True)
class GaitFollowerOutput:
    """1 制御周期分の出力。

    Attributes:
        joint_targets: 目標関節角 (1, 12) [rad]、GO2_JOINT_NAMES の順。
        command_was_stopped: 速度指令が無いか期限切れで、停止の指令（すべて 0）に切り替えたか。
    """

    joint_targets: torch.Tensor
    command_was_stopped: bool


class GaitCommandFollower:
    """1 台の Go2 の歩行方策の推論を、受け取った速度指令の有効期限を確かめて行う。

    受け取った速度指令が有効期限より古いか、まだ 1 つも受け取っていなければ、停止の指令（すべて 0）を歩行方策に与える。
    高レベル方策のノードが止まると、ロボットはその場で立ち止まる。
    """

    def __init__(self, controller: GaitController, velocity_command_expiry_ns: int) -> None:
        """1 台分（num_envs = 1）の推論器と、速度指令の有効期限 [ns]（シミュレーション時間）を受け取る。"""
        self._controller = controller
        self._expiry_ns = velocity_command_expiry_ns
        self._stop_command = torch.zeros((1, VELOCITY_COMMAND_SIZE))

    def reset(self) -> None:
        """ロボットを初期姿勢に戻したときに呼ぶ。歩行方策の前回の行動を 0 に戻す。"""
        self._controller.reset_where(torch.ones(1, dtype=torch.bool))

    def compute(
        self, proprioception: Go2Proprioception, state_stamp_ns: int, command: StampedVelocityCommand | None
    ) -> GaitFollowerOutput:
        """時刻 state_stamp_ns の状態から、1 制御周期分の目標関節角を返す。

        Args:
            proprioception: 1 台分の自己受容の状態。
            state_stamp_ns: 状態を読んだ時点のシミュレーション時刻 [ns]。指令の期限の判定に使う。
            command: 最後に受け取った速度指令。まだ受け取っていなければ None。

        Returns:
            目標関節角と、停止の指令に切り替えたか。停止に切り替えても歩行方策は呼ぶ（前回の行動を更新し続けるため）。
        """
        fresh = command is not None and is_command_fresh(command.stamp_ns, state_stamp_ns, self._expiry_ns)
        values = command.values if fresh and command is not None else self._stop_command
        joint_targets = self._controller.compute_joint_targets(proprioception, values)
        return GaitFollowerOutput(joint_targets=joint_targets, command_was_stopped=not fresh)
