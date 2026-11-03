"""箱押しの 1 試行（エピソード）の結果と、複数の試行をまとめた評価指標。"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from cooperative_go2.entities.push_evaluation import goal_distance_m, progress_toward_goal_m, yaw_change_deg
from cooperative_go2.entities.push_termination import PushTerminationReason


class EmptyTrialSetError(ValueError):
    """集計する試行が 1 つも無いことを表す。"""


@dataclass(frozen=True)
class PlanarPose:
    """水平面の位置と向き。

    Attributes:
        x_m: x 座標 [m]（各環境の原点から見たワールド座標系）。
        y_m: y 座標 [m]。
        yaw_rad: ヨー角 [rad]。上から見て反時計回りが正。
    """

    x_m: float
    y_m: float
    yaw_rad: float


@dataclass(frozen=True)
class PushEpisodeResult:
    """箱押しの 1 試行の結果。

    Attributes:
        env_index: 試行を行った並列環境の番号。
        robot_start: 開始時の Go2 の胴体の位置と向き。
        box_start: 開始時の箱の位置と向き。
        goal_x_m: 目標領域の中心の x [m]。
        goal_y_m: 目標領域の中心の y [m]。
        box_end: 終了時（リセット前）の箱の位置と向き。
        termination_reason: 終了理由。
        high_level_steps: 高レベル方策のステップ数。
        wall_time_s: 開始から終了までの実時間 [s]。並列環境で同時に進めているため、他の環境の試行と重なる。
        collisions: 箱以外（壁・他のロボット）との衝突の回数。
        falls: 転倒の回数。転倒した時点で試行が終わるため 0 か 1。
    """

    env_index: int
    robot_start: PlanarPose
    box_start: PlanarPose
    goal_x_m: float
    goal_y_m: float
    box_end: PlanarPose
    termination_reason: PushTerminationReason
    high_level_steps: int
    wall_time_s: float
    collisions: int
    falls: int

    @property
    def success(self) -> bool:
        """成功で終わったか。"""
        return self.termination_reason == PushTerminationReason.SUCCESS

    def displacement_m(self) -> float:
        """箱が開始時から終了時までに動いた水平距離 [m]。"""
        return math.hypot(self.box_end.x_m - self.box_start.x_m, self.box_end.y_m - self.box_start.y_m)

    def progress_toward_goal_m(self) -> float:
        """箱が、開始位置から見た目標の方向へ進んだ量 [m]。目標から遠ざかれば負。

        Raises:
            UndefinedGoalDirectionError: 箱の開始位置が目標位置と一致している。
        """
        start, end, goal = self._as_tensors()
        return float(progress_toward_goal_m(start, end, goal)[0])

    def final_goal_distance_m(self) -> float:
        """終了時の箱の中心から目標の中心までの水平距離 [m]。"""
        _, end, goal = self._as_tensors()
        return float(goal_distance_m(end, goal)[0])

    def rotation_deg(self) -> float:
        """開始時から終了時までの箱のヨー角の変化の大きさ [deg]（0 以上 180 以下）。"""
        start_yaw = torch.tensor([self.box_start.yaw_rad])
        end_yaw = torch.tensor([self.box_end.yaw_rad])
        return float(yaw_change_deg(start_yaw, end_yaw)[0])

    def _as_tensors(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = torch.tensor([[self.box_start.x_m, self.box_start.y_m]], dtype=torch.float64)
        end = torch.tensor([[self.box_end.x_m, self.box_end.y_m]], dtype=torch.float64)
        goal = torch.tensor([[self.goal_x_m, self.goal_y_m]], dtype=torch.float64)
        return start, end, goal


@dataclass(frozen=True)
class PushTrialSummary:
    """複数の試行をまとめた評価指標。

    Attributes:
        num_trials: 試行の数。
        success_rate: 成功した試行の割合（0〜1）。
        mean_progress_toward_goal_m: 箱の目標方向への移動量の平均 [m]。
        mean_final_goal_distance_m: 最終目標距離の平均 [m]。
        mean_rotation_deg: 箱の回転角の平均 [deg]。
        mean_success_time_s: 成功した試行の所要時間（シミュレーション時間）の平均 [s]。成功が無ければ None。
        fall_rate: 転倒した試行の割合（0〜1）。
        total_collisions: 衝突の回数の合計。
    """

    num_trials: int
    success_rate: float
    mean_progress_toward_goal_m: float
    mean_final_goal_distance_m: float
    mean_rotation_deg: float
    mean_success_time_s: float | None
    fall_rate: float
    total_collisions: int


def summarize_push_trials(results: Sequence[PushEpisodeResult], high_level_period_s: float) -> PushTrialSummary:
    """試行の結果をまとめた評価指標を返す。

    Args:
        results: 試行の結果。1 つ以上。
        high_level_period_s: 高レベル方策の周期 [s]。所要時間の換算に使う。

    Raises:
        EmptyTrialSetError: results が空。
    """
    if not results:
        raise EmptyTrialSetError("集計する試行が無い")
    count = len(results)
    success_times = [result.high_level_steps * high_level_period_s for result in results if result.success]
    return PushTrialSummary(
        num_trials=count,
        success_rate=len(success_times) / count,
        mean_progress_toward_goal_m=sum(result.progress_toward_goal_m() for result in results) / count,
        mean_final_goal_distance_m=sum(result.final_goal_distance_m() for result in results) / count,
        mean_rotation_deg=sum(result.rotation_deg() for result in results) / count,
        mean_success_time_s=sum(success_times) / len(success_times) if success_times else None,
        fall_rate=sum(1 for result in results if result.falls > 0) / count,
        total_collisions=sum(result.collisions for result in results),
    )
