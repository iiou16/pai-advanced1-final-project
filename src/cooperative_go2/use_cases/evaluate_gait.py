"""学習済みの歩行方策に評価シナリオの速度指令を与えて、区間ごとの追従誤差と転倒を測る流れ。"""

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_evaluation import (
    CommandSegment,
    FallRecoveryExclusion,
    GaitEvaluationScenario,
    GaitEvaluationSuite,
    ScenarioSummary,
    SegmentTrackingAccumulator,
    SegmentTrackingSummary,
)
from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.use_cases.gait_controller import GaitPolicy


class InvalidEvaluationRequestError(ValueError):
    """評価の依頼が成り立たないことを表す（シードの再利用、環境数、シナリオの長さの超過・不足、環境数の不一致など）。"""


@dataclass(frozen=True)
class GaitEvaluationRequest:
    """利用者が指定する評価の条件。

    Attributes:
        suite: 評価シナリオの一式。
        num_envs: 並列に走らせる環境の数。環境ごとにリセット時の初期姿勢が異なる。
        seed: 評価用の乱数シード。学習のシードとは別にする。
    """

    suite: GaitEvaluationSuite
    num_envs: int
    seed: int


@dataclass(frozen=True)
class PlannedSegment:
    """制御ステップ数に直した区間。

    Attributes:
        segment: 元の区間。
        num_steps: 区間の制御ステップ数。評価の settle_steps より大きい。
    """

    segment: CommandSegment
    num_steps: int


@dataclass(frozen=True)
class PlannedScenario:
    """制御ステップ数に直したシナリオ。"""

    name: str
    segments: tuple[PlannedSegment, ...]


@dataclass(frozen=True)
class GaitEvaluationPlan:
    """検証済みの評価の条件。

    Attributes:
        config: 評価に使う設定（学習時の設定の環境数だけを置き換えたもの）。
        scenarios: 評価シナリオ。依頼と同じ順。
        settle_steps: 追従誤差から除く過渡のステップ数。各区間の始めと、転倒した環境の立ち直りに使う。
        seed: 評価用の乱数シード。
    """

    config: GaitConfig
    scenarios: tuple[PlannedScenario, ...]
    settle_steps: int
    seed: int


def plan_gait_evaluation(training_config: GaitConfig, request: GaitEvaluationRequest) -> GaitEvaluationPlan:
    """学習時の設定と評価の依頼から、評価の条件を決める。

    区間の長さと過渡の時間は、最も近い制御ステップ数に丸める。シナリオは丸めた後のステップ数でエピソードの長さ
    以内に限る。時間切れで初期姿勢に戻ると、その瞬間が追従誤差に混ざるため。

    Raises:
        InvalidEvaluationRequestError: シードが学習と同じ、環境数が正でない、シナリオがエピソードの長さを超える、
            指令が学習時の範囲（command 節）の外にある、または丸めた結果、区間のステップ数が過渡のステップ数以下になる。
    """
    env_config = training_config.env
    if request.seed == training_config.training.seed:
        raise InvalidEvaluationRequestError(f"評価のシード {request.seed} が学習のシードと同じ")
    if request.num_envs <= 0:
        raise InvalidEvaluationRequestError(f"評価の環境数は正の整数が必要: {request.num_envs}")
    settle_steps = round(request.suite.settle_time_s / env_config.control_dt_s)
    scenarios = tuple(
        _plan_scenario(scenario, settle_steps, training_config) for scenario in request.suite.scenarios
    )
    evaluation_config = dataclasses.replace(
        training_config, env=dataclasses.replace(env_config, num_envs=request.num_envs)
    )
    return GaitEvaluationPlan(evaluation_config, scenarios, settle_steps, request.seed)


def _plan_scenario(
    scenario: GaitEvaluationScenario, settle_steps: int, training_config: GaitConfig
) -> PlannedScenario:
    env_config = training_config.env
    segments = []
    for segment in scenario.segments:
        _require_within_training_range(scenario.name, segment.command, training_config)
        num_steps = round(segment.duration_s / env_config.control_dt_s)
        if num_steps <= settle_steps:
            raise InvalidEvaluationRequestError(
                f"シナリオ {scenario.name} の区間 {segment.duration_s} s は制御周期 {env_config.control_dt_s} s に"
                f"丸めると {num_steps} ステップで、過渡の {settle_steps} ステップ以下になる"
            )
        segments.append(PlannedSegment(segment, num_steps))
    # 区間ごとに丸めたステップ数の合計で比べる。秒の合計がエピソード以内でも、丸めで上限を超えうるため
    total_steps = sum(planned.num_steps for planned in segments)
    if total_steps > training_config.max_episode_length_steps():
        raise InvalidEvaluationRequestError(
            f"シナリオ {scenario.name} の長さ {total_steps} ステップがエピソードの長さ "
            f"{training_config.max_episode_length_steps()} ステップを超える"
        )
    return PlannedScenario(scenario.name, tuple(segments))


def _require_within_training_range(
    scenario_name: str, command: VelocityCommand, training_config: GaitConfig
) -> None:
    # 学習していない指令で測った追従誤差は方策の性能を表さないため、評価の対象にしない
    command_config = training_config.command
    ranges = (command_config.lin_vel_x_mps, command_config.lin_vel_y_mps, command_config.yaw_rate_radps)
    for value, value_range in zip(command.as_tuple(), ranges, strict=True):
        if not value_range.low <= value <= value_range.high:
            raise InvalidEvaluationRequestError(
                f"シナリオ {scenario_name} の指令 {command} が学習時の範囲（command 節）の外にある"
            )


@dataclass(frozen=True)
class GaitRolloutStep:
    """1 制御ステップ後の状態。速度は本体座標系で、リセットする前の値。

    Attributes:
        observation: 次の行動を決めるための観測 (num_envs, GAIT_OBSERVATION_DIM)。
        base_lin_vel: 胴体の線速度 (num_envs, 3) [m/s]。
        base_ang_vel: 胴体の角速度 (num_envs, 3) [rad/s]。
        fallen: このステップで転倒した環境を True とするマスク (num_envs,)。
    """

    observation: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    fallen: torch.Tensor


class GaitRolloutEnvironment(Protocol):
    """速度指令を外から与えて歩行方策を走らせる環境。"""

    @property
    def num_envs(self) -> int:
        """並列環境の数。"""
        ...

    @property
    def device(self) -> torch.device | str:
        """テンソルを置くデバイス。"""
        ...

    def reset_with_command(self, command: VelocityCommand) -> torch.Tensor:
        """全環境を初期姿勢に戻し、速度指令を command に固定する。初期観測を返す。"""
        ...

    def switch_command(self, command: VelocityCommand) -> torch.Tensor:
        """リセットせずに、全環境の速度指令を直ちに command に切り替える。新しい指令を含む観測を返す。"""
        ...

    def step(self, actions: torch.Tensor) -> GaitRolloutStep:
        """行動を適用して 1 制御ステップ進める。転倒した環境は初期姿勢に戻して続ける。

        Raises:
            SimulationDivergedError: いずれかの環境で物理計算が発散した。
        """
        ...


class VideoRecorder(Protocol):
    """評価の様子を動画に撮る。start() から stop() までの各制御ステップが 1 フレームになる。"""

    def start(self) -> None:
        """録画を始める。"""
        ...

    def stop(self) -> None:
        """録画を終えてファイルに書き出す。"""
        ...


# シナリオ名を受け取り、そのシナリオの評価動画を撮るレコーダを返す
VideoRecorderFactory = Callable[[str], VideoRecorder]


class GaitEvaluator:
    """評価シナリオを順に走らせ、区間ごとの追従誤差と転倒を集計する。"""

    def __init__(
        self,
        environment: GaitRolloutEnvironment,
        policy: GaitPolicy,
        recorder_factory: VideoRecorderFactory | None,
    ) -> None:
        """評価に使う環境・方策・レコーダの作り方を受け取る。

        Args:
            environment: 評価に使う環境。plan.config と plan.seed で構築しておく。
            policy: 学習済みの歩行方策。
            recorder_factory: シナリオごとに評価動画を撮る場合のレコーダの作り方。None なら撮らない。
        """
        self._environment = environment
        self._policy = policy
        self._recorder_factory = recorder_factory

    def run(self, plan: GaitEvaluationPlan) -> tuple[ScenarioSummary, ...]:
        """plan のシナリオを順に走らせて集計する。各シナリオは全環境を初期姿勢に戻してから始める。

        追従誤差からは、各区間の始めの過渡（plan.settle_steps）と、転倒した環境の転倒したステップとその後の
        plan.settle_steps ステップ（立ち直りの過渡）を除く。転倒は除かずに数える。

        Returns:
            シナリオごとの結果。plan.scenarios と同じ順。

        Raises:
            InvalidEvaluationRequestError: 環境の数が plan と異なる。
            SimulationDivergedError: 評価中に物理計算が発散した。
        """
        if self._environment.num_envs != plan.config.env.num_envs:
            raise InvalidEvaluationRequestError(
                f"環境の数 {self._environment.num_envs} が評価の条件の {plan.config.env.num_envs} と異なる"
            )
        return tuple(self._run_recorded(scenario, plan.settle_steps) for scenario in plan.scenarios)

    def _run_recorded(self, scenario: PlannedScenario, settle_steps: int) -> ScenarioSummary:
        if self._recorder_factory is None:
            return self._run_scenario(scenario, settle_steps)
        recorder = self._recorder_factory(scenario.name)
        recorder.start()
        # 評価が例外で止まっても録画のファイルを閉じる。録画の終了自体が失敗した場合はその例外が優先され、
        # 評価の例外は __context__ に残る
        try:
            return self._run_scenario(scenario, settle_steps)
        finally:
            recorder.stop()

    def _run_scenario(self, scenario: PlannedScenario, settle_steps: int) -> ScenarioSummary:
        environment = self._environment
        fell_in_scenario = torch.zeros(environment.num_envs, dtype=torch.bool, device=environment.device)
        recovery = FallRecoveryExclusion(environment.num_envs, settle_steps, environment.device)
        segment_summaries: list[SegmentTrackingSummary] = []
        observation = environment.reset_with_command(scenario.segments[0].segment.command)
        for index, planned in enumerate(scenario.segments):
            if index > 0:
                observation = environment.switch_command(planned.segment.command)
            accumulator = SegmentTrackingAccumulator(planned.segment, environment.num_envs, environment.device)
            with torch.no_grad():
                for step_index in range(planned.num_steps):
                    result = environment.step(self._policy(observation))
                    accumulator.add_falls(result.fallen)
                    fell_in_scenario |= result.fallen
                    recovered = recovery.advance(result.fallen)
                    if step_index >= settle_steps:
                        accumulator.add_tracking(result.base_lin_vel, result.base_ang_vel, recovered)
                    observation = result.observation
            segment_summaries.append(accumulator.summarize())
        fall_rate = float(fell_in_scenario.sum()) / environment.num_envs
        return ScenarioSummary(scenario.name, tuple(segment_summaries), fall_rate)
