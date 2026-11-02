"""学習済みの歩行方策で、自己受容の状態と速度指令から 12 関節の目標角を出す推論器。"""

from typing import Protocol

import torch

from cooperative_go2.entities.gait_action import GaitActionHistory, to_joint_targets
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import GaitObservationInputs, build_gait_observation
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE


class InvalidVelocityCommandShapeError(ValueError):
    """推論器に渡した速度指令のテンソルの形が、推論器のロボットの数と指令の次元に合わないことを表す。"""


class GaitPolicy(Protocol):
    """観測から行動（既定姿勢からの 12 関節の残差）を出す推論専用の方策。パラメータは更新しない。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """観測 (num_envs, GAIT_OBSERVATION_DIM) から行動 (num_envs, 12) を返す。"""
        ...


class GaitController:
    """歩行方策 π_gait(o_gait, v_cmd) → q_target[12] を、学習時と同じ前処理・後処理で包んだ推論器。

    - 観測は GAIT_OBSERVATION_LAYOUT の順（角速度・重力方向・速度指令・関節角の差・関節角速度・前回の行動）で、
      学習時の設定の係数で正規化する。
    - 関節は入出力とも GO2_JOINT_NAMES の順（脚は FR → FL → RR → RL、各脚は hip → thigh → calf）。
    - 出力は既定姿勢からの残差ではなく、PD 制御にそのまま渡す目標関節角 [rad]。
    - 前回の行動を観測に含めるため状態を持つ。ロボットを初期姿勢に戻したら reset_where() で前回の行動を 0 にする。
    - 学習時の行動の遅延（env.simulate_action_latency）はロボット側の遅れとして扱い、ここでは模擬しない。
      出力は今回の行動から作った目標で、適用を 1 制御周期遅らせるかは目標を適用する側が決める。
    """

    def __init__(self, policy: GaitPolicy, config: GaitConfig, num_envs: int, device: torch.device | str) -> None:
        """推論に使う方策と、その方策を学習したときの設定を受け取る。前回の行動は 0 で始める。

        Args:
            policy: 学習済みの歩行方策。
            config: policy を学習したときの設定。観測の係数・行動のクリップ・行動の係数・既定姿勢を使う。
            num_envs: 同時に推論するロボットの数。
            device: 状態と出力を置くデバイス。
        """
        self._policy = policy
        self._config = config
        self._num_envs = num_envs
        self._default_dof_pos = torch.tensor(config.robot.default_joint_angles_rad.in_order(), device=device)
        self._actions = GaitActionHistory.zeros(num_envs, NUM_GO2_JOINTS, device)

    def reset_where(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True のロボットの前回の行動を 0 に戻す。初期姿勢に戻したロボットに使う。"""
        self._actions = self._actions.reset_where(mask)

    def compute_joint_targets(self, proprioception: Go2Proprioception, commands: torch.Tensor) -> torch.Tensor:
        """1 制御周期分の目標関節角 (num_envs, 12) [rad] を返し、今回の行動を前回の行動として覚える。

        Args:
            proprioception: 現在の自己受容の状態。
            commands: 速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]（本体座標系）。
                学習時の指令の範囲（command 節）の外の値は、方策が学習していない入力になる。

        Raises:
            InvalidVelocityCommandShapeError: commands の形が (num_envs, 3) でない。
        """
        expected_shape = (self._num_envs, VELOCITY_COMMAND_SIZE)
        if tuple(commands.shape) != expected_shape:
            raise InvalidVelocityCommandShapeError(f"速度指令の形は {expected_shape} が必要: {tuple(commands.shape)}")
        observation = build_gait_observation(
            GaitObservationInputs(
                proprioception=proprioception,
                commands=commands,
                last_actions=self._actions.current,
                default_dof_pos=self._default_dof_pos,
            ),
            self._config.observation,
        )
        with torch.no_grad():
            raw_actions = self._policy(observation)
        env_config = self._config.env
        self._actions = self._actions.advance(raw_actions, env_config.clip_actions)
        return to_joint_targets(self._actions.current, self._default_dof_pos, env_config.action_scale)
