import dataclasses
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig
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

TURN_IN_PLACE = VelocityCommand(vx_mps=0.0, vy_mps=0.0, yaw_rate_radps=1.0)
TRAINING_SEED = 1
EVALUATION_SEED = 1001


@pytest.fixture
def training_config(gait_config_path: Path) -> GaitConfig:
    config = load_gait_settings(gait_config_path).gait
    training = dataclasses.replace(config.training, seed=TRAINING_SEED)
    env = dataclasses.replace(config.env, control_dt_s=0.02, episode_length_s=20.0)
    return dataclasses.replace(config, training=training, env=env)


def _request(**overrides: object) -> GaitEvaluationRequest:
    values: dict[str, object] = {"command": TURN_IN_PLACE, "duration_s": 0.06, "num_envs": 2, "seed": EVALUATION_SEED}
    values.update(overrides)
    return GaitEvaluationRequest(**values)  # type: ignore[arg-type]


def test_plan_converts_duration_to_steps_and_overrides_env_count(training_config: GaitConfig) -> None:
    plan = plan_gait_evaluation(training_config, _request(duration_s=10.0, num_envs=3))

    assert plan.num_steps == 500
    assert plan.config.env.num_envs == 3
    assert plan.config.reward == training_config.reward


def test_plan_accepts_duration_equal_to_episode_length(training_config: GaitConfig) -> None:
    plan = plan_gait_evaluation(training_config, _request(duration_s=20.0))

    assert plan.num_steps == 1000


@pytest.mark.parametrize(
    "overrides",
    [
        {"seed": TRAINING_SEED},
        {"num_envs": 0},
        {"duration_s": 0.001},
        {"duration_s": 20.02},
        {"duration_s": float("nan")},
        {"duration_s": float("inf")},
        {"command": VelocityCommand(vx_mps=float("nan"), vy_mps=0.0, yaw_rate_radps=0.0)},
    ],
    ids=[
        "seed-reused", "zero-envs", "rounds-to-zero-steps", "longer-than-episode",
        "nan-duration", "inf-duration", "nan-command",
    ],
)
def test_plan_rejects_invalid_request(training_config: GaitConfig, overrides: dict[str, object]) -> None:
    with pytest.raises(InvalidEvaluationRequestError):
        plan_gait_evaluation(training_config, _request(**overrides))


class FakeRolloutEnvironment:
    """指令の半分の角速度でしか回れず、2 ステップ目に環境 0 が転倒する環境。diverge_at で発散させられる。"""

    def __init__(self, num_envs: int, diverge_at: int | None = None) -> None:
        self._num_envs = num_envs
        self._diverge_at = diverge_at
        self.reset_command: VelocityCommand | None = None
        self.received_actions: list[torch.Tensor] = []

    @property
    def num_envs(self) -> int:
        return self._num_envs

    @property
    def device(self) -> str:
        return "cpu"

    def reset_with_command(self, command: VelocityCommand) -> torch.Tensor:
        self.reset_command = command
        return torch.zeros(self._num_envs, GAIT_OBSERVATION_DIM)

    def step(self, actions: torch.Tensor) -> GaitRolloutStep:
        self.received_actions.append(actions)
        step_index = len(self.received_actions)
        if step_index == self._diverge_at:
            raise SimulationDivergedError("発散")
        fallen = torch.zeros(self._num_envs, dtype=torch.bool)
        fallen[0] = step_index == 2
        return GaitRolloutStep(
            observation=torch.full((self._num_envs, GAIT_OBSERVATION_DIM), float(step_index)),
            base_lin_vel=torch.zeros(self._num_envs, 3),
            base_ang_vel=torch.tensor([[0.0, 0.0, 0.5]] * self._num_envs),
            fallen=fallen,
        )


class EchoPolicy:
    """観測の先頭の値をそのまま 12 関節の行動にする方策。観測が次のステップに渡っているかを確かめる。"""

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return observation[:, :1].expand(-1, 12)


class FakeRecorder:
    def __init__(self) -> None:
        self.events: list[str] = []

    def start(self) -> None:
        self.events.append("start")

    def stop(self) -> None:
        self.events.append("stop")


def test_evaluator_applies_command_and_summarizes_tracking(training_config: GaitConfig) -> None:
    environment = FakeRolloutEnvironment(num_envs=2)
    plan = plan_gait_evaluation(training_config, _request())

    summary = GaitEvaluator(environment, EchoPolicy(), recorder=None).run(plan)

    assert environment.reset_command == TURN_IN_PLACE
    assert summary.num_steps == 3
    assert summary.mean_yaw_rate_error_radps == pytest.approx(0.5)
    assert summary.fall_rate == pytest.approx(0.5)


def test_policy_receives_observation_from_previous_step(training_config: GaitConfig) -> None:
    environment = FakeRolloutEnvironment(num_envs=1)
    plan = plan_gait_evaluation(training_config, _request(num_envs=1))

    GaitEvaluator(environment, EchoPolicy(), recorder=None).run(plan)

    assert [actions[0, 0].item() for actions in environment.received_actions] == [0.0, 1.0, 2.0]


def test_recording_wraps_the_rollout(training_config: GaitConfig) -> None:
    recorder = FakeRecorder()
    plan = plan_gait_evaluation(training_config, _request())

    GaitEvaluator(FakeRolloutEnvironment(num_envs=2), EchoPolicy(), recorder).run(plan)

    assert recorder.events == ["start", "stop"]


def test_recording_is_stopped_even_when_simulation_diverges(training_config: GaitConfig) -> None:
    recorder = FakeRecorder()
    plan = plan_gait_evaluation(training_config, _request())
    evaluator = GaitEvaluator(FakeRolloutEnvironment(num_envs=2, diverge_at=2), EchoPolicy(), recorder)

    with pytest.raises(SimulationDivergedError):
        evaluator.run(plan)

    assert recorder.events == ["start", "stop"]


def test_evaluator_rejects_environment_with_different_env_count(training_config: GaitConfig) -> None:
    plan = plan_gait_evaluation(training_config, _request(num_envs=2))

    with pytest.raises(InvalidEvaluationRequestError):
        GaitEvaluator(FakeRolloutEnvironment(num_envs=3), EchoPolicy(), recorder=None).run(plan)
