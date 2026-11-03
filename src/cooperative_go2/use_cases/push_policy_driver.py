"""ROS 2 の高レベル方策ノードの中身。学習時と同じ観測を組み立て、学習済みの方策で速度指令を決める。"""

from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.push_observation import PlanarBodyState, PushObservationInputs, build_push_observation
from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE


class NoActiveGoalError(RuntimeError):
    """目標を受け取る前に速度指令を求めたことを表す。"""


class PushPolicy(Protocol):
    """高レベル方策の観測 (num_envs, PUSH_OBSERVATION_DIM) から速度指令 (num_envs, 3) を出す推論専用の方策。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """速度指令を返す。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。"""
        ...


@dataclass(frozen=True)
class PushPolicyInputs:
    """1 台分の、速度指令を決める時点の状態。テンソルの先頭の次元は 1。

    Attributes:
        robot: Go2 の胴体の水平面の状態。
        robot_projected_gravity: Go2 の本体座標系で表した重力方向 (1, 3)。
        box: 箱の水平面の状態。
        box_contact: Go2 が箱に触れているか (1,)。
    """

    robot: PlanarBodyState
    robot_projected_gravity: torch.Tensor
    box: PlanarBodyState
    box_contact: torch.Tensor


@dataclass(frozen=True)
class PushPolicyDecision:
    """1 ステップの判断。

    Attributes:
        commands: 速度指令 (1, 3)。設定の範囲に切り詰めた値。
        box_goal_distance_m: 箱の中心から目標領域の中心までの距離 [m]。
        in_goal_duration_s: 箱が目標領域に入ったまま続いた時間 [s]。
        succeeded: 成功の条件（目標領域に入った状態を所定の時間続けた）を満たしたか。依頼の結果を返すための判定で、
            試行の記録には使わない（記録はシミュレーションノードが真値に対して judge_push_step で判定する）。
    """

    commands: torch.Tensor
    box_goal_distance_m: float
    in_goal_duration_s: float
    succeeded: bool


class PushPolicyDriver:
    """1 台の Go2 の高レベル方策を、学習環境（Go2PushEnv）と同じ前処理・後処理で動かす。

    - 観測は build_push_observation で組み立てる。前回の指令には、範囲に切り詰めた後の指令を使う。
    - 出力は設定の範囲（task.commands）に切り詰める。
    - 成功の判定（目標領域に入った状態を続けた時間）を、高レベル方策の周期で数える。
    前提: decide() を高レベル方策の周期（task.episode.high_level_period_s）ごとに 1 回呼ぶ。
    """

    def __init__(self, policy: PushPolicy, task: PushTaskConfig) -> None:
        """学習済みの方策と、その方策を学習したときの環境の設定を受け取る。"""
        self._policy = policy
        self._task = task
        limits = task.commands.as_ranges()
        self._command_low = torch.tensor([[limit.low for limit in limits]])
        self._command_high = torch.tensor([[limit.high for limit in limits]])
        self._goal_xy_m: torch.Tensor | None = None
        self._last_commands = torch.zeros((1, VELOCITY_COMMAND_SIZE))
        self._hold_steps = torch.zeros(1, dtype=torch.int64)

    def start(self, goal_xy_m: tuple[float, float]) -> None:
        """新しい目標で作業を始める。前回の指令と、目標領域に入っていた時間を 0 に戻す。

        Args:
            goal_xy_m: 目標領域の中心 (x, y) [m]（ワールド座標系）。
        """
        self._goal_xy_m = torch.tensor([goal_xy_m])
        self._last_commands = torch.zeros((1, VELOCITY_COMMAND_SIZE))
        self._hold_steps = torch.zeros(1, dtype=torch.int64)

    def decide(self, inputs: PushPolicyInputs) -> PushPolicyDecision:
        """速度指令を決め、成功の判定を 1 ステップ進める。

        Raises:
            NoActiveGoalError: start() で目標を受け取っていない。
        """
        if self._goal_xy_m is None:
            raise NoActiveGoalError("目標を受け取る前に速度指令を求めた")
        observation = build_push_observation(
            PushObservationInputs(
                robot=inputs.robot,
                robot_projected_gravity=inputs.robot_projected_gravity,
                box=inputs.box,
                goal_xy_m=self._goal_xy_m,
                box_contact=inputs.box_contact,
                last_commands=self._last_commands,
            ),
            self._task.observation,
        )
        with torch.no_grad():
            raw_commands = self._policy(observation)
        commands = torch.clamp(raw_commands, self._command_low, self._command_high)
        self._last_commands = commands
        box_xy = inputs.box.position_xy_m
        in_goal = is_in_goal_region(box_xy, self._goal_xy_m, self._task.success.goal_radius_m)
        self._hold_steps = advance_goal_hold_steps(self._hold_steps, in_goal)
        hold_steps = int(self._hold_steps[0])
        return PushPolicyDecision(
            commands=commands,
            box_goal_distance_m=float(torch.linalg.vector_norm(box_xy - self._goal_xy_m, dim=1)[0]),
            in_goal_duration_s=hold_steps * self._task.episode.high_level_period_s,
            succeeded=bool(has_held_goal(self._hold_steps, self._task.required_hold_steps())[0]),
        )
