"""歩行環境を、評価用のポート GaitRolloutEnvironment に合わせる。"""

import torch

from cooperative_go2.entities.gait_termination import TerminationReason
from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.interface_adapters.gait_env import Go2GaitEnv
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP
from cooperative_go2.use_cases.evaluate_gait import GaitRolloutStep
from cooperative_go2.use_cases.simulation_errors import SimulationDivergedError


class Go2GaitRolloutEnvironment:
    """速度指令を固定して歩行方策を走らせる。学習と違い、物理計算が 1 環境でも発散したら止める。"""

    def __init__(self, env: Go2GaitEnv) -> None:
        """評価に使う環境を受け取る。速度指令は reset_with_command() と switch_command() で与える。"""
        self._env = env

    @property
    def num_envs(self) -> int:
        """並列環境の数。"""
        return self._env.num_envs

    @property
    def device(self) -> torch.device | str:
        """テンソルを置くデバイス。"""
        return self._env.device

    def reset_with_command(self, command: VelocityCommand) -> torch.Tensor:
        """全環境を初期姿勢に戻し、速度指令を command に固定する。初期観測を返す。"""
        self._env.fix_command(command)
        return self._env.reset()[POLICY_OBSERVATION_GROUP]

    def switch_command(self, command: VelocityCommand) -> torch.Tensor:
        """リセットせずに、全環境の速度指令を直ちに command に切り替える。新しい指令を含む観測を返す。"""
        self._env.fix_command(command)
        return self._env.get_observations()[POLICY_OBSERVATION_GROUP]

    def step(self, actions: torch.Tensor) -> GaitRolloutStep:
        """行動を適用して 1 制御ステップ進める。

        Raises:
            SimulationDivergedError: いずれかの環境で物理計算が発散した、または速度が有限でなくなった。
        """
        outcome = self._env.step_with_outcome(actions)
        state = outcome.pre_reset_state
        # 物理エンジンが異常として報告しない NaN・inf も、評価の集計に混ぜずに発散として止める
        non_finite = ~(
            torch.isfinite(state.base_lin_vel).all(dim=1) & torch.isfinite(state.proprioception.base_ang_vel).all(dim=1)
        )
        diverged = (outcome.termination_reasons == int(TerminationReason.SIM_DIVERGED)) | non_finite
        if bool(diverged.any()):
            raise SimulationDivergedError(f"物理計算が発散した環境: {torch.nonzero(diverged).flatten().tolist()}")
        return GaitRolloutStep(
            observation=outcome.observation,
            base_lin_vel=state.base_lin_vel,
            base_ang_vel=state.proprioception.base_ang_vel,
            fallen=outcome.termination_reasons == int(TerminationReason.FALL),
        )
