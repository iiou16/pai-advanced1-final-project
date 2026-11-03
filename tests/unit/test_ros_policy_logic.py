"""ROS 2 の方策ノードの中身（速度指令の期限切れで止まる歩行、学習時と同じ前処理・後処理の高レベル方策）の検証。"""

from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_LAYOUT
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.push_observation import PUSH_OBSERVATION_LAYOUT, PlanarBodyState
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_training_settings
from cooperative_go2.use_cases.gait_command_follower import GaitCommandFollower, StampedVelocityCommand
from cooperative_go2.use_cases.gait_controller import GaitController
from cooperative_go2.use_cases.push_policy_driver import NoActiveGoalError, PushPolicyDriver, PushPolicyInputs

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPIRY_NS = 300


def _layout_slice(layout: tuple[tuple[str, int], ...], name: str) -> slice:
    start = 0
    for component, size in layout:
        if component == name:
            return slice(start, start + size)
        start += size
    raise KeyError(name)


class RecordingPolicy:
    """受け取った観測を記録し、決まった出力を返す方策。"""

    def __init__(self, output: torch.Tensor) -> None:
        self.output = output
        self.observations: list[torch.Tensor] = []

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.observations.append(observation)
        return self.output.expand(observation.shape[0], -1)


@pytest.fixture
def gait(gait_config_path: Path) -> GaitConfig:
    return load_gait_settings(gait_config_path).gait


@pytest.fixture
def task() -> PushTaskConfig:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single_wide.yaml").task


def _standing() -> Go2Proprioception:
    return Go2Proprioception(
        base_ang_vel=torch.zeros(1, 3),
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        dof_pos=torch.zeros(1, 12),
        dof_vel=torch.zeros(1, 12),
    )


def _commands_in_gait_observation(policy: RecordingPolicy) -> torch.Tensor:
    observation = policy.observations[-1]
    return observation[:, _layout_slice(GAIT_OBSERVATION_LAYOUT, "velocity_command")]


def test_gait_follower_passes_fresh_command(gait: GaitConfig) -> None:
    policy = RecordingPolicy(torch.zeros(1, 12))
    follower = GaitCommandFollower(GaitController(policy, gait, 1, "cpu"), EXPIRY_NS)
    command = StampedVelocityCommand(stamp_ns=1000, values=torch.tensor([[0.5, 0.0, 0.0]]))

    output = follower.compute(_standing(), 1000 + EXPIRY_NS, command)

    assert not output.command_was_stopped
    assert _commands_in_gait_observation(policy)[0, 0] > 0


@pytest.mark.parametrize("command_stamp_ns", [None, 1000], ids=["never_received", "expired"])
def test_gait_follower_stops_without_fresh_command(gait: GaitConfig, command_stamp_ns: int | None) -> None:
    policy = RecordingPolicy(torch.zeros(1, 12))
    follower = GaitCommandFollower(GaitController(policy, gait, 1, "cpu"), EXPIRY_NS)
    command = None
    if command_stamp_ns is not None:
        command = StampedVelocityCommand(stamp_ns=command_stamp_ns, values=torch.tensor([[0.5, 0.0, 1.0]]))

    output = follower.compute(_standing(), 1000 + EXPIRY_NS + 1, command)

    assert output.command_was_stopped
    assert torch.count_nonzero(_commands_in_gait_observation(policy)) == 0


def test_gait_follower_reset_clears_last_action(gait: GaitConfig) -> None:
    policy = RecordingPolicy(torch.full((1, 12), 0.5))
    follower = GaitCommandFollower(GaitController(policy, gait, 1, "cpu"), EXPIRY_NS)
    follower.compute(_standing(), 0, None)

    follower.reset()
    follower.compute(_standing(), 20, None)

    last_actions = policy.observations[-1][:, _layout_slice(GAIT_OBSERVATION_LAYOUT, "last_action")]
    assert torch.count_nonzero(last_actions) == 0


def _planar(x: float, y: float) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x, y]]),
        yaw_rad=torch.zeros(1),
        lin_vel_xy_mps=torch.zeros(1, 2),
        yaw_rate_radps=torch.zeros(1),
    )


def _push_inputs(box_x: float) -> PushPolicyInputs:
    return PushPolicyInputs(
        robot=_planar(0.0, 0.0),
        robot_projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        box=_planar(box_x, 0.0),
        box_contact=torch.tensor([False]),
    )


def test_push_driver_requires_goal(task: PushTaskConfig) -> None:
    driver = PushPolicyDriver(RecordingPolicy(torch.zeros(1, 3)), task)

    with pytest.raises(NoActiveGoalError):
        driver.decide(_push_inputs(1.0))


def test_push_driver_clamps_commands_and_feeds_clamped_command_back(task: PushTaskConfig) -> None:
    policy = RecordingPolicy(torch.tensor([[5.0, -5.0, 0.5]]))
    driver = PushPolicyDriver(policy, task)
    driver.start((3.0, 0.0))

    first = driver.decide(_push_inputs(1.0))
    driver.decide(_push_inputs(1.0))

    limits = task.commands
    assert first.commands == pytest.approx(torch.tensor([[limits.vx_mps.high, limits.vy_mps.low, 0.5]]))
    last_command = policy.observations[-1][:, _layout_slice(PUSH_OBSERVATION_LAYOUT, "last_command")]
    scales = task.observation
    expected = torch.tensor(
        [[limits.vx_mps.high * scales.lin_vel, limits.vy_mps.low * scales.lin_vel, 0.5 * scales.ang_vel]]
    )
    assert last_command == pytest.approx(expected)


def test_push_driver_restart_clears_last_command(task: PushTaskConfig) -> None:
    policy = RecordingPolicy(torch.tensor([[0.3, 0.0, 0.0]]))
    driver = PushPolicyDriver(policy, task)
    driver.start((3.0, 0.0))
    driver.decide(_push_inputs(1.0))

    driver.start((3.0, 0.0))
    driver.decide(_push_inputs(1.0))

    last_command = policy.observations[-1][:, _layout_slice(PUSH_OBSERVATION_LAYOUT, "last_command")]
    assert torch.count_nonzero(last_command) == 0


def test_push_driver_succeeds_after_holding_goal(task: PushTaskConfig) -> None:
    driver = PushPolicyDriver(RecordingPolicy(torch.zeros(1, 3)), task)
    driver.start((1.0, 0.0))

    decisions = [driver.decide(_push_inputs(1.0)) for _ in range(task.required_hold_steps())]

    assert not any(decision.succeeded for decision in decisions[:-1])
    assert decisions[-1].succeeded
    assert decisions[-1].in_goal_duration_s == pytest.approx(task.success.hold_time_s)


def test_push_driver_success_hold_resets_when_box_leaves_goal(task: PushTaskConfig) -> None:
    driver = PushPolicyDriver(RecordingPolicy(torch.zeros(1, 3)), task)
    driver.start((1.0, 0.0))
    driver.decide(_push_inputs(1.0))

    outside = driver.decide(_push_inputs(2.0))

    assert outside.in_goal_duration_s == 0.0
    assert outside.box_goal_distance_m == pytest.approx(1.0)
