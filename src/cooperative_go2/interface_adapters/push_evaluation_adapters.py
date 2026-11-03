"""箱押しの評価（use_cases.evaluate_push）に、学習環境と規則ベースの制御をつなぐ。"""

from pathlib import Path

import torch

from cooperative_go2.entities.push_config import RuleBasedPushConfig
from cooperative_go2.entities.push_observation import PUSH_OBSERVATION_DIM
from cooperative_go2.entities.rule_based_push import RuleBasedPushInputs, compute_rule_based_push_commands
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.interface_adapters.config_loader import load_push_training_settings
from cooperative_go2.interface_adapters.gait_checkpoint import checkpoint_path
from cooperative_go2.interface_adapters.push_env import Go2PushEnv
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP
from cooperative_go2.interface_adapters.rsl_rl_gait import PolicySpecEnvironment, RslRlPolicy, load_rsl_rl_policy
from cooperative_go2.interface_adapters.rsl_rl_push import PUSH_RUN_CONFIG_FILENAME
from cooperative_go2.use_cases.evaluate_push import PushEnvironmentView, PushEvaluationStep


class Go2PushEvaluationEnvironment:
    """Go2PushEnv を、評価が使う環境（PushEvaluationEnvironment）として見せる。"""

    def __init__(self, env: Go2PushEnv) -> None:
        """リセット直後の学習環境を受け取る。"""
        self._env = env

    @property
    def num_envs(self) -> int:
        """並列環境の数。"""
        return self._env.num_envs

    @property
    def max_episode_steps(self) -> int:
        """1 試行の高レベル方策のステップ数の上限。"""
        return self._env.max_episode_length

    def view(self) -> PushEnvironmentView:
        """現在の Go2・箱・目標の位置と観測を返す。"""
        observation = self._env.get_observations()[POLICY_OBSERVATION_GROUP]
        return PushEnvironmentView(snapshot=self._env.task_snapshot, observation=observation)

    def step(self, commands: torch.Tensor) -> PushEvaluationStep:
        """速度指令 (num_envs, 3) で 1 ステップ進める。"""
        outcome = self._env.step_with_outcome(commands)
        return PushEvaluationStep(
            done=outcome.done, termination_reasons=outcome.termination_reasons, pre_reset=outcome.pre_reset
        )


class RuleBasedCommandSource:
    """規則ベースの制御（実験 E1 のベースライン）で速度指令を決める。"""

    def __init__(self, config: RuleBasedPushConfig, goal_radius_m: float) -> None:
        """制御のパラメータと、停止の判定に使う目標領域の半径 [m] を受け取る。"""
        self._config = config
        self._goal_radius_m = goal_radius_m

    def __call__(self, view: PushEnvironmentView) -> torch.Tensor:
        """Go2・箱・目標の位置から速度指令 (num_envs, 3) を返す。"""
        snapshot = view.snapshot
        inputs = RuleBasedPushInputs(
            robot_xy_m=snapshot.robot.position_xy_m,
            robot_yaw_rad=snapshot.robot.yaw_rad,
            box_xy_m=snapshot.box.position_xy_m,
            goal_xy_m=snapshot.goal_xy_m,
        )
        return compute_rule_based_push_commands(inputs, self._config, self._goal_radius_m).commands


class PolicyCommandSource:
    """学習済みの高レベル方策（実験 E2）で速度指令を決める。出力は決定的（分布の平均）。"""

    def __init__(self, policy: RslRlPolicy) -> None:
        """推論用に読み込んだ方策を受け取る。"""
        self._policy = policy

    def __call__(self, view: PushEnvironmentView) -> torch.Tensor:
        """観測から速度指令 (num_envs, 3) を返す。範囲の外の値は環境が切り詰める。"""
        with torch.no_grad():
            return self._policy(view.observation)


def load_push_policy(run_dir: Path, iteration: int, device: torch.device | str) -> RslRlPolicy:
    """箱押しの学習の出力先から、反復番号 iteration のチェックポイントを推論用の方策として読み込む。

    Raises:
        FileNotFoundError: 学習時の設定の写しが無い。
        ConfigValidationError: 設定の写しが現在の設定の形式に合わない。
        CheckpointNotFoundError: チェックポイントが無い。
        CheckpointLoadError: チェックポイントを読み込めない、またはネットワークの構成が設定と合わない。
    """
    settings = load_push_training_settings(run_dir / PUSH_RUN_CONFIG_FILENAME)
    spec = PolicySpecEnvironment(PUSH_OBSERVATION_DIM, VELOCITY_COMMAND_SIZE, device)
    return load_rsl_rl_policy(checkpoint_path(run_dir, iteration), settings.rsl_rl, spec)
