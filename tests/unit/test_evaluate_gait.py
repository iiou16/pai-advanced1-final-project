import dataclasses
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_evaluation import CommandSegment, GaitEvaluationScenario, GaitEvaluationSuite
from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_DIM
from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.interface_adapters.config_loader import load_gait_settings
from cooperative_go2.use_cases.evaluate_gait import (
    GaitEvaluationRequest,
    GaitEvaluator,
    GaitRolloutStep,
    InvalidEvaluationRequestError,
    plan_gait_evaluation,
)
from cooperative_go2.use_cases.simulation_errors import SimulationDivergedError

STOP = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=0.0)
TURN_IN_PLACE = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.0)
TRAINING_SEED = 1
EVALUATION_SEED = 1001
CONTROL_DT_S = 0.02


@pytest.fixture
def training_config(gait_config_path: Path) -> GaitConfig:
    config = load_gait_settings(gait_config_path).gait
    training = dataclasses.replace(config.training, seed=TRAINING_SEED)
    env = dataclasses.replace(config.env, control_dt_s=CONTROL_DT_S, episode_length_s=20.0)
    return dataclasses.replace(config, training=training, env=env)


def _suite(*scenarios: GaitEvaluationScenario, settle_time_s: float = 0.02) -> GaitEvaluationSuite:
    return GaitEvaluationSuite(settle_time_s=settle_time_s, scenarios=scenarios)


def _scenario(name: str, *segments: tuple[VelocityCommand, float]) -> GaitEvaluationScenario:
    return GaitEvaluationScenario(
        name=name, segments=tuple(CommandSegment(command, duration_s) for command, duration_s in segments)
    )


# 3 ステップ（うち始めの 1 ステップは過渡）の旋回
TURN_SUITE = _suite(_scenario("turn", (TURN_IN_PLACE, 0.06)))


def _request(suite: GaitEvaluationSuite = TURN_SUITE, num_envs: int = 2, seed: int = EVALUATION_SEED):
    return GaitEvaluationRequest(suite=suite, num_envs=num_envs, seed=seed)


def test_plan_converts_durations_to_steps_and_overrides_env_count(training_config: GaitConfig) -> None:
    suite = _suite(_scenario("change", (STOP, 10.0), (TURN_IN_PLACE, 4.0)), settle_time_s=1.0)

    plan = plan_gait_evaluation(training_config, _request(suite, num_envs=3))

    segments = plan.scenarios[0].segments
    assert [segment.num_steps for segment in segments] == [500, 200]
    assert plan.settle_steps == 50
    assert plan.config.env.num_envs == 3
    assert plan.config.reward == training_config.reward


def test_plan_accepts_scenario_equal_to_episode_length(training_config: GaitConfig) -> None:
    suite = _suite(_scenario("long", (STOP, 15.0), (TURN_IN_PLACE, 5.0)))

    plan = plan_gait_evaluation(training_config, _request(suite))

    assert sum(segment.num_steps for segment in plan.scenarios[0].segments) == 1000


@pytest.mark.parametrize(
    "request_overrides",
    [
        {"seed": TRAINING_SEED},
        {"num_envs": 0},
        {"suite": _suite(_scenario("long", (STOP, 15.0), (TURN_IN_PLACE, 5.02)))},
        # 秒の合計は 19.98 s でエピソード以内だが、1.11 s を 56 ステップに丸めるので合計 1008 ステップになる
        {"suite": _suite(_scenario("rounded", *[(STOP, 1.11)] * 18))},
        # 0.01 s は制御周期 0.02 s に丸めると 0 ステップになり、過渡（0 ステップ）以下になる
        {"suite": _suite(_scenario("short", (STOP, 0.01)), settle_time_s=0.0)},
        {"suite": _suite(_scenario("fast", (VelocityCommand(vx_mps=5.0, vy_mps=0.0, yaw_rate_radps=0.0), 1.0)))},
    ],
    ids=[
        "seed-reused", "zero-envs", "longer-than-episode", "longer-after-rounding", "rounds-to-zero-steps",
        "outside-training-range",
    ],  # fmt: skip
)
def test_plan_rejects_invalid_request(training_config: GaitConfig, request_overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidEvaluationRequestError):
        plan_gait_evaluation(training_config, _request(**request_overrides))  # type: ignore[arg-type]


class FakeRolloutEnvironment:
    """指令の半分の角速度でしか回れず、各シナリオの 2 ステップ目に環境 0 が転倒する環境。

    diverge_at で、通算そのステップ目に発散させられる。
    """

    def __init__(self, num_envs: int, diverge_at: int | None = None) -> None:
        self._num_envs = num_envs
        self._diverge_at = diverge_at
        self._command = STOP
        self._steps_since_reset = 0
        self.events: list[tuple[str, VelocityCommand]] = []
        self.received_actions: list[torch.Tensor] = []

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return "cpu"

    def reset_with_command(self, command: VelocityCommand) -> torch.Tensor:
        self.events.append(("reset", command))
        self._command = command
        self._steps_since_reset = 0
        return torch.zeros(self._num_envs, GAIT_OBSERVATION_DIM)

    def switch_command(self, command: VelocityCommand) -> torch.Tensor:
        self.events.append(("switch", command))
        self._command = command
        return torch.full((self._num_envs, GAIT_OBSERVATION_DIM), -1.0)

    def step(self, actions: torch.Tensor) -> GaitRolloutStep:
        self.received_actions.append(actions)
        self._steps_since_reset += 1
        if len(self.received_actions) == self._diverge_at:
            raise SimulationDivergedError("発散")
        fallen = torch.zeros(self._num_envs, dtype=torch.bool)
        fallen[0] = self._steps_since_reset == 2
        half_yaw_rate = self._command.yaw_rate_radps / 2
        return GaitRolloutStep(
            observation=torch.full((self._num_envs, GAIT_OBSERVATION_DIM), float(self._steps_since_reset)),
            base_lin_vel=torch.zeros(self._num_envs, 3),
            base_ang_vel=torch.tensor([[0.0, 0.0, half_yaw_rate]] * self._num_envs),
            fallen=fallen,
        )


class EchoPolicy:
    """観測の先頭の値をそのまま 12 関節の行動にする方策。観測が次のステップに渡っているかを確かめる。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :1].expand(-1, 12)


class FakeRecorder:
    def __init__(self, name: str, events: list[str]) -> None:
        self._name = name
        self._events = events

    def start(self) -> None:
        self._events.append(f"start:{self._name}")

    def stop(self) -> None:
        self._events.append(f"stop:{self._name}")


def _recorder_factory(events: list[str]):
    return lambda name: FakeRecorder(name, events)


def test_evaluator_summarizes_tracking_after_settle_time(training_config: GaitConfig) -> None:
    environment = FakeRolloutEnvironment(num_envs=2)
    plan = plan_gait_evaluation(training_config, _request())

    (summary,) = GaitEvaluator(environment, EchoPolicy(), recorder_factory=None).run(plan)

    (segment,) = summary.segments
    assert environment.events == [("reset", TURN_IN_PLACE)]
    # 3 ステップのうち 1 ステップ目は過渡。環境 0 は 2 ステップ目に転倒し、その後 1 ステップは立ち直りの過渡
    assert segment.num_tracking_samples == 2
    assert segment.mean_yaw_rate_error_radps == pytest.approx(0.5)
    assert segment.fall_rate == pytest.approx(0.5)
    assert summary.fall_rate == pytest.approx(0.5)


def test_falls_during_settle_time_are_counted(training_config: GaitConfig) -> None:
    # 区間は 3 ステップ、過渡は 2 ステップ。転倒は 2 ステップ目（過渡の中）に起きる
    suite = _suite(_scenario("turn", (TURN_IN_PLACE, 0.06)), settle_time_s=0.04)
    plan = plan_gait_evaluation(training_config, _request(suite))

    (summary,) = GaitEvaluator(FakeRolloutEnvironment(num_envs=2), EchoPolicy(), recorder_factory=None).run(plan)

    assert summary.segments[0].num_tracking_samples == 1
    assert summary.segments[0].num_falls == 1


def test_segments_switch_command_without_reset(training_config: GaitConfig) -> None:
    environment = FakeRolloutEnvironment(num_envs=1)
    suite = _suite(_scenario("change", (STOP, 0.06), (TURN_IN_PLACE, 0.06)))
    plan = plan_gait_evaluation(training_config, _request(suite, num_envs=1))

    (summary,) = GaitEvaluator(environment, EchoPolicy(), recorder_factory=None).run(plan)

    assert environment.events == [("reset", STOP), ("switch", TURN_IN_PLACE)]
    assert [segment.command for segment in summary.segments] == [STOP, TURN_IN_PLACE]
    # 切り替えの直後は、新しい指令を含む観測（先頭が -1）で行動を決める
    assert [actions[0, 0].item() for actions in environment.received_actions] == [0.0, 1.0, 2.0, -1.0, 4.0, 5.0]


def test_each_scenario_starts_from_reset(training_config: GaitConfig) -> None:
    environment = FakeRolloutEnvironment(num_envs=1)
    suite = _suite(_scenario("stop", (STOP, 0.06)), _scenario("turn", (TURN_IN_PLACE, 0.06)))
    plan = plan_gait_evaluation(training_config, _request(suite, num_envs=1))

    summaries = GaitEvaluator(environment, EchoPolicy(), recorder_factory=None).run(plan)

    assert [summary.name for summary in summaries] == ["stop", "turn"]
    assert environment.events == [("reset", STOP), ("reset", TURN_IN_PLACE)]


def test_each_scenario_is_recorded_separately(training_config: GaitConfig) -> None:
    events: list[str] = []
    suite = _suite(_scenario("stop", (STOP, 0.06)), _scenario("turn", (TURN_IN_PLACE, 0.06)))
    plan = plan_gait_evaluation(training_config, _request(suite))

    GaitEvaluator(FakeRolloutEnvironment(num_envs=2), EchoPolicy(), _recorder_factory(events)).run(plan)

    assert events == ["start:stop", "stop:stop", "start:turn", "stop:turn"]


def test_recording_is_stopped_even_when_simulation_diverges(training_config: GaitConfig) -> None:
    events: list[str] = []
    plan = plan_gait_evaluation(training_config, _request())
    evaluator = GaitEvaluator(FakeRolloutEnvironment(num_envs=2, diverge_at=2), EchoPolicy(), _recorder_factory(events))

    with pytest.raises(SimulationDivergedError):
        evaluator.run(plan)

    assert events == ["start:turn", "stop:turn"]


def test_evaluator_rejects_environment_with_different_env_count(training_config: GaitConfig) -> None:
    plan = plan_gait_evaluation(training_config, _request(num_envs=2))

    with pytest.raises(InvalidEvaluationRequestError):
        GaitEvaluator(FakeRolloutEnvironment(num_envs=3), EchoPolicy(), recorder_factory=None).run(plan)
