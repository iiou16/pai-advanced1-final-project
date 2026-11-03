"""箱押しの方策（規則ベース・学習済み）を、並列環境ごとに 1 試行ずつ走らせて結果を集める評価の流れ。

各環境の最初の 1 試行だけを数える。早く終わる試行（成功や転倒）ほど多く数えてしまう偏りを避けるため。
"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.push_observation import PlanarBodyState, PushTaskSnapshot
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.push_trial import PlanarPose, PushEpisodeResult


class EvaluationDidNotFinishError(RuntimeError):
    """評価のステップ数の上限までに、すべての環境の試行が終わらなかったことを表す。"""


@dataclass(frozen=True)
class PushEnvironmentView:
    """方策が速度指令を決めるために見られるもの。

    Attributes:
        snapshot: Go2・箱・目標の位置（規則ベースの制御が使う）。
        observation: 高レベル方策の観測 (num_envs, PUSH_OBSERVATION_DIM)（学習済みの方策が使う）。
    """

    snapshot: PushTaskSnapshot
    observation: torch.Tensor


@dataclass(frozen=True)
class PushEvaluationStep:
    """評価の 1 ステップの結果。

    Attributes:
        done: 試行が終わった環境のマスク (num_envs,)。終わった環境は次の試行に入っている。
        termination_reasons: 環境ごとの終了理由 (num_envs,)。PushTerminationReason の値。
        pre_reset: リセットする前の Go2・箱・目標の位置。
    """

    done: torch.Tensor
    termination_reasons: torch.Tensor
    pre_reset: PushTaskSnapshot


class PushEvaluationEnvironment(Protocol):
    """評価が使う箱押しの環境。"""

    @property
    def num_envs(self) -> int:
        """並列環境の数。"""
        ...

    @property
    def max_episode_steps(self) -> int:
        """1 試行の高レベル方策のステップ数の上限（時間切れになるステップ数）。"""
        ...

    def view(self) -> PushEnvironmentView:
        """現在の状態を返す。"""
        ...

    def step(self, commands: torch.Tensor) -> PushEvaluationStep:
        """速度指令 (num_envs, 3) で 1 ステップ進め、終わった試行の環境をリセットする。"""
        ...


class PushCommandSource(Protocol):
    """環境の状態から速度指令 (num_envs, 3) を決めるもの（規則ベースの制御や学習済みの方策）。"""

    def __call__(self, view: PushEnvironmentView) -> torch.Tensor:
        """速度指令 (num_envs, 3) を返す。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。"""
        ...


def evaluate_push(
    environment: PushEvaluationEnvironment, source: PushCommandSource, clock: Callable[[], float]
) -> tuple[PushEpisodeResult, ...]:
    """各環境で 1 試行ずつ走らせ、環境の番号の順に結果を返す。

    前提: environment はリセット直後（全環境が試行の始め）であること。

    Args:
        environment: 評価する環境。
        source: 速度指令を決めるもの。
        clock: 実時間 [s] を返す関数（試行ごとの実時間の計測に使う）。

    Raises:
        EvaluationDidNotFinishError: max_episode_steps を超えても終わらない試行があった（時間切れの判定の不具合）。
    """
    num_envs = environment.num_envs
    starts = environment.view().snapshot
    start_time = clock()
    steps = torch.zeros(num_envs, dtype=torch.int64)
    results: dict[int, PushEpisodeResult] = {}
    for _ in range(environment.max_episode_steps):
        step = environment.step(source(environment.view()))
        steps += 1
        now = clock()
        for env_index in torch.nonzero(step.done.cpu()).flatten().tolist():
            if env_index not in results:
                result = _episode_result(env_index, starts, step, int(steps[env_index]))
                results[env_index] = dataclasses.replace(result, wall_time_s=now - start_time)
        if len(results) == num_envs:
            return tuple(results[env_index] for env_index in range(num_envs))
    unfinished = sorted(set(range(num_envs)) - results.keys())
    raise EvaluationDidNotFinishError(
        f"{environment.max_episode_steps} ステップを超えても試行が終わらない環境がある: {unfinished}"
    )


def _pose(body: PlanarBodyState, env_index: int) -> PlanarPose:
    position = body.position_xy_m[env_index]
    return PlanarPose(x_m=float(position[0]), y_m=float(position[1]), yaw_rad=float(body.yaw_rad[env_index]))


def _episode_result(
    env_index: int, start: PushTaskSnapshot, step: PushEvaluationStep, steps_taken: int
) -> PushEpisodeResult:
    # 実時間は呼び出し側で dataclasses.replace して入れる
    reason = PushTerminationReason(int(step.termination_reasons[env_index]))
    goal = start.goal_xy_m[env_index]
    return PushEpisodeResult(
        env_index=env_index,
        robot_start=_pose(start.robot, env_index),
        box_start=_pose(start.box, env_index),
        goal_x_m=float(goal[0]),
        goal_y_m=float(goal[1]),
        box_end=_pose(step.pre_reset.box, env_index),
        termination_reason=reason,
        high_level_steps=steps_taken,
        wall_time_s=0.0,
        # S1 には壁も他のロボットも無いため、箱以外との衝突は起こらない
        collisions=0,
        falls=int(reason == PushTerminationReason.FALL),
    )
