"""学習済みの歩行方策に固定の速度指令を与えて、追従誤差と転倒を測る流れ。"""

import dataclasses
from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_evaluation import GaitTrackingAccumulator, GaitTrackingSummary
from cooperative_go2.entities.velocity_command import VelocityCommand


class InvalidEvaluationRequestError(ValueError):
    """評価の依頼が、学習の設定と組み合わせて成り立たないことを表す（シードの再利用、長さの不足・超過など）。"""


class SimulationDivergedError(RuntimeError):
    """評価中に物理計算が発散し、評価の結果を信頼できないことを表す。"""


@dataclass(frozen=True)
class GaitEvaluationRequest:
    """利用者が指定する評価の条件。

    Attributes:
        command: 全環境に与える速度指令。
        duration_s: 走らせるシミュレーション時間 [s]。
        num_envs: 並列に走らせる環境の数。
        seed: 評価用の乱数シード。学習のシードとは別にする。
    """

    command: VelocityCommand
    duration_s: float
    num_envs: int
    seed: int


@dataclass(frozen=True)
class GaitEvaluationPlan:
    """検証済みの評価の条件。

    Attributes:
        config: 評価に使う設定（学習時の設定の環境数だけを置き換えたもの）。
        command: 全環境に与える速度指令。
        num_steps: 走らせる制御ステップ数。1 以上。
        seed: 評価用の乱数シード。
    """

    config: GaitConfig
    command: VelocityCommand
    num_steps: int
    seed: int


def plan_gait_evaluation(training_config: GaitConfig, request: GaitEvaluationRequest) -> GaitEvaluationPlan:
    """学習時の設定と評価の依頼から、評価の条件を決める。

    評価はエピソードの長さ以内に限る。時間切れで初期姿勢に戻ると、その瞬間が追従誤差に混ざるため。

    Raises:
        InvalidEvaluationRequestError: シードが学習と同じ、環境数が正でない、時間が 1 制御周期未満、
            または時間がエピソードの長さを超える。
    """
    env_config = training_config.env
    if request.seed == training_config.training.seed:
        raise InvalidEvaluationRequestError(f"評価のシード {request.seed} が学習のシードと同じ")
    if request.num_envs <= 0:
        raise InvalidEvaluationRequestError(f"評価の環境数は正の整数が必要: {request.num_envs}")
    if request.duration_s > env_config.episode_length_s:
        raise InvalidEvaluationRequestError(
            f"評価の時間 {request.duration_s} s がエピソードの長さ {env_config.episode_length_s} s を超える"
        )
    num_steps = round(request.duration_s / env_config.control_dt_s)
    if num_steps < 1:
        raise InvalidEvaluationRequestError(
            f"評価の時間 {request.duration_s} s が制御周期 {env_config.control_dt_s} s に満たない"
        )
    evaluation_config = dataclasses.replace(
        training_config, env=dataclasses.replace(env_config, num_envs=request.num_envs)
    )
    return GaitEvaluationPlan(evaluation_config, request.command, num_steps, request.seed)


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
    """速度指令を固定して歩行方策を走らせる環境。"""

    @property
    def num_envs(self) -> int:
        """並列環境の数。"""
        ...

    @property
    def device(self) -> torch.device | str:
        """テンソルを置くデバイス。"""
        ...

    def reset_with_command(self, command: VelocityCommand) -> torch.Tensor:
        """全環境を初期姿勢に戻し、以後の速度指令を固定する。初期観測を返す。"""
        ...

    def step(self, actions: torch.Tensor) -> GaitRolloutStep:
        """行動を適用して 1 制御ステップ進める。転倒した環境は初期姿勢に戻して続ける。

        Raises:
            SimulationDivergedError: いずれかの環境で物理計算が発散した。
        """
        ...


class GaitPolicy(Protocol):
    """観測から行動（既定姿勢からの 12 関節の残差）を出す推論専用の方策。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """観測 (num_envs, GAIT_OBSERVATION_DIM) から行動 (num_envs, 12) を返す。"""
        ...


class VideoRecorder(Protocol):
    """評価の様子を動画に撮る。start() から stop() までの各制御ステップが 1 フレームになる。"""

    def start(self) -> None:
        """録画を始める。"""
        ...

    def stop(self) -> None:
        """録画を終えてファイルに書き出す。"""
        ...


class GaitEvaluator:
    """1 つの速度指令で歩行方策を走らせ、追従誤差と転倒を集計する。"""

    def __init__(
        self, environment: GaitRolloutEnvironment, policy: GaitPolicy, recorder: VideoRecorder | None
    ) -> None:
        """評価に使う環境・方策・レコーダを受け取る。

        Args:
            environment: 評価に使う環境。plan.config と plan.seed で構築しておく。
            policy: 学習済みの歩行方策。
            recorder: 評価動画を撮る場合のレコーダ。None なら撮らない。
        """
        self._environment = environment
        self._policy = policy
        self._recorder = recorder

    def run(self, plan: GaitEvaluationPlan) -> GaitTrackingSummary:
        """plan.num_steps だけ走らせて集計する。

        Raises:
            SimulationDivergedError: 評価中に物理計算が発散した。
        """
        if self._recorder is None:
            return self._rollout(plan)
        self._recorder.start()
        # 評価が例外で止まっても録画のファイルを閉じる。例外はそのまま呼び出し元へ伝わる
        try:
            return self._rollout(plan)
        finally:
            self._recorder.stop()

    def _rollout(self, plan: GaitEvaluationPlan) -> GaitTrackingSummary:
        accumulator = GaitTrackingAccumulator(plan.command, self._environment.num_envs, self._environment.device)
        observation = self._environment.reset_with_command(plan.command)
        with torch.no_grad():
            for _ in range(plan.num_steps):
                result = self._environment.step(self._policy(observation))
                accumulator.add_step(result.base_lin_vel, result.base_ang_vel, result.fallen)
                observation = result.observation
        return accumulator.summarize()
