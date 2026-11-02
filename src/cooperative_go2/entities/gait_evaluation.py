"""歩行方策の評価シナリオ（速度指令の区間の並び）と、区間ごとの追従誤差・転倒の集計。"""

import math
import re
from dataclasses import dataclass

import torch

from cooperative_go2.entities.velocity_command import VelocityCommand

# シナリオ名は評価動画のファイル名に使うため、パスの区切りや空白を含めない
_SCENARIO_NAME_PATTERN = re.compile(r"[A-Za-z0-9_-]+")


class InvalidEvaluationScenarioError(ValueError):
    """評価シナリオが不変条件（有限の指令、正の長さ、名前の形式と重複、空でない区間など）を満たさないことを表す。"""


@dataclass(frozen=True)
class CommandSegment:
    """同じ速度指令を与え続ける区間。

    Attributes:
        command: 区間中に全環境へ与える速度指令。各成分は有限の値。
        duration_s: 区間の長さ [s]。正の有限の値。
    """

    command: VelocityCommand
    duration_s: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) for value in self.command.as_tuple()):
            raise InvalidEvaluationScenarioError(f"速度指令は有限の値が必要: {self.command}")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise InvalidEvaluationScenarioError(f"区間の長さは正の有限の値が必要: {self.duration_s}")


@dataclass(frozen=True)
class GaitEvaluationScenario:
    """1 つの評価シナリオ。区間を順に切り替えながら、初期姿勢から続けて走らせる。

    Attributes:
        name: シナリオ名。英数字・`_`・`-` だけからなる。評価動画のファイル名に使う。
        segments: 指令の区間の並び。1 つ以上。
    """

    name: str
    segments: tuple[CommandSegment, ...]

    def __post_init__(self) -> None:
        if _SCENARIO_NAME_PATTERN.fullmatch(self.name) is None:
            raise InvalidEvaluationScenarioError(f"シナリオ名は英数字・_・- だけにする: {self.name!r}")
        if not self.segments:
            raise InvalidEvaluationScenarioError(f"シナリオ {self.name} に区間が無い")

    def duration_s(self) -> float:
        """シナリオ全体の長さ [s]。"""
        return sum(segment.duration_s for segment in self.segments)


@dataclass(frozen=True)
class GaitEvaluationSuite:
    """評価シナリオの一式。

    Attributes:
        settle_time_s: 各区間の始めに、追従誤差の集計から除く時間 [s]。指令を切り替えた直後の過渡を除くため。
            0 以上で、どの区間の長さよりも短い。転倒は除かずに数える。
        scenarios: 評価シナリオ。1 つ以上で、名前は重複しない。
    """

    settle_time_s: float
    scenarios: tuple[GaitEvaluationScenario, ...]

    def __post_init__(self) -> None:
        if not math.isfinite(self.settle_time_s) or self.settle_time_s < 0:
            raise InvalidEvaluationScenarioError(f"settle_time_s は 0 以上の有限の値が必要: {self.settle_time_s}")
        if not self.scenarios:
            raise InvalidEvaluationScenarioError("評価シナリオが 1 つも無い")
        names = [scenario.name for scenario in self.scenarios]
        if len(set(names)) != len(names):
            raise InvalidEvaluationScenarioError(f"シナリオ名が重複している: {names}")
        for scenario in self.scenarios:
            for segment in scenario.segments:
                if segment.duration_s <= self.settle_time_s:
                    raise InvalidEvaluationScenarioError(
                        f"シナリオ {scenario.name} の区間の長さ {segment.duration_s} s が "
                        f"settle_time_s {self.settle_time_s} s 以下で、追従誤差を集計できない"
                    )


@dataclass(frozen=True)
class SegmentTrackingSummary:
    """1 つの区間で全環境を走らせた結果。

    Attributes:
        command: 区間中に与えた速度指令。
        duration_s: 区間の長さ [s]。
        num_envs: 並列に走らせた環境の数。
        num_tracking_samples: 追従誤差を集計した「環境 × 制御ステップ」の数。区間の始めの過渡と、転倒した環境の
            立ち直りの過渡を除いた数。
        mean_lin_vel_error_mps: 水平速度 (vx, vy) の指令との差のノルムの、集計したサンプルでの平均 [m/s]。
            集計したサンプルが無い（全環境が転倒し続けた）場合は None。
        mean_yaw_rate_error_radps: ヨー角速度の指令との差の絶対値の、集計したサンプルでの平均 [rad/s]。
            集計したサンプルが無い場合は None。
        num_falls: 区間中に全環境で転倒した回数の合計。転倒した環境は初期姿勢に戻って続けるため、1 環境で複数回数えうる。
        fall_rate: 区間中に 1 回以上転倒した環境の割合（0〜1）。
    """

    command: VelocityCommand
    duration_s: float
    num_envs: int
    num_tracking_samples: int
    mean_lin_vel_error_mps: float | None
    mean_yaw_rate_error_radps: float | None
    num_falls: int
    fall_rate: float


@dataclass(frozen=True)
class ScenarioSummary:
    """1 つのシナリオの結果。

    Attributes:
        name: シナリオ名。
        segments: 区間ごとの結果。シナリオの区間と同じ順。
        fall_rate: シナリオ中に 1 回以上転倒した環境の割合（0〜1）。
    """

    name: str
    segments: tuple[SegmentTrackingSummary, ...]
    fall_rate: float


class FallRecoveryExclusion:
    """転倒してリセットされた環境を、立ち直るまでの間だけ追従誤差の集計から外す。

    転倒したステップのリセット前の速度と、初期姿勢から立ち直る間の速度は、指令への追従の良し悪しではなく
    転倒の結果を表す。これを追従誤差に混ぜないため、転倒したステップとその後の exclusion_steps ステップを除く。
    """

    def __init__(self, num_envs: int, exclusion_steps: int, device: torch.device | str) -> None:
        """どの環境も除外していない状態で始める。exclusion_steps は転倒後に除くステップ数（0 以上）。"""
        self._exclusion_steps = exclusion_steps
        self._remaining = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def advance(self, fallen: torch.Tensor) -> torch.Tensor:
        """1 制御ステップ進め、このステップを追従誤差に含めてよい環境のマスク (num_envs,) を返す。

        Args:
            fallen: このステップで転倒した環境を True とするマスク (num_envs,)。
        """
        self._remaining = torch.where(fallen, self._exclusion_steps + 1, self._remaining)
        included = self._remaining == 0
        self._remaining = torch.clamp(self._remaining - 1, min=0)
        return included


class SegmentTrackingAccumulator:
    """1 つの区間について、制御ステップごとの速度と転倒を積算し、SegmentTrackingSummary にまとめる。"""

    def __init__(self, segment: CommandSegment, num_envs: int, device: torch.device | str) -> None:
        self._segment = segment
        self._num_envs = num_envs
        command = segment.command
        self._command_xy = torch.tensor([command.vx_mps, command.vy_mps], device=device)
        self._lin_vel_error_sum = torch.zeros((), device=device)
        self._yaw_rate_error_sum = torch.zeros((), device=device)
        self._num_samples = torch.zeros((), dtype=torch.int64, device=device)
        self._falls_per_env = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def add_tracking(self, base_lin_vel: torch.Tensor, base_ang_vel: torch.Tensor, included: torch.Tensor) -> None:
        """1 制御ステップ分の速度を、included の環境についてだけ追従誤差として積算する。

        Args:
            base_lin_vel: 胴体の線速度 (num_envs, 3)、本体座標系 [m/s]。
            base_ang_vel: 胴体の角速度 (num_envs, 3)、本体座標系 [rad/s]。
            included: 集計に含める環境を True とするマスク (num_envs,)。
        """
        lin_vel_error = torch.linalg.vector_norm(base_lin_vel[:, :2] - self._command_xy, dim=1)
        yaw_rate_error = torch.abs(base_ang_vel[:, 2] - self._segment.command.yaw_rate_radps)
        self._lin_vel_error_sum += torch.where(included, lin_vel_error, 0.0).sum()
        self._yaw_rate_error_sum += torch.where(included, yaw_rate_error, 0.0).sum()
        self._num_samples += included.sum()

    def add_falls(self, fallen: torch.Tensor) -> None:
        """1 制御ステップ分の転倒を積算する。fallen はこのステップで転倒した環境を True とするマスク (num_envs,)。"""
        self._falls_per_env += fallen.to(torch.int64)

    def summarize(self) -> SegmentTrackingSummary:
        """積算した値を集計する。集計したサンプルが無ければ、追従誤差の平均は None にする。"""
        num_samples = int(self._num_samples)
        return SegmentTrackingSummary(
            command=self._segment.command,
            duration_s=self._segment.duration_s,
            num_envs=self._num_envs,
            num_tracking_samples=num_samples,
            mean_lin_vel_error_mps=_mean_or_none(self._lin_vel_error_sum, num_samples),
            mean_yaw_rate_error_radps=_mean_or_none(self._yaw_rate_error_sum, num_samples),
            num_falls=int(self._falls_per_env.sum()),
            fall_rate=float((self._falls_per_env > 0).sum()) / self._num_envs,
        )


def _mean_or_none(total: torch.Tensor, count: int) -> float | None:
    # 全環境が転倒し続けた区間では追従誤差を定義できない。0 などで埋めると良い追従と区別できなくなる
    if count == 0:
        return None
    return float(total) / count
