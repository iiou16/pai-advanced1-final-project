import dataclasses

import pytest
import torch
from conftest import REPO_ROOT, TeamMarlEnvFactory

from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.team_marl_config import TeamMarlTaskConfig
from cooperative_go2.entities.team_marl_observation import TEAM_ACTOR_OBSERVATION_LAYOUT
from cooperative_go2.interface_adapters.config_loader import load_team_marl_run_settings
from cooperative_go2.use_cases.team_policy_driver import (
    NoActiveTaskError,
    PartnerReport,
    TeamPolicyDriver,
    TeamPolicyInputs,
    TeamTaskStatus,
)

PERIOD_NS = 100_000_000
RETREAT_XY = (3.8, 0.0)
RETREAT_RADIUS_M = 0.4
START_NS = 5_000_000_000


class RecordingPolicy:
    """決まった指令を返し、受け取った観測を記録する Actor。"""

    def __init__(self, command: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.command = torch.tensor([command])
        self.observations: list[torch.Tensor] = []

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.observations.append(observation.clone())
        return self.command.expand(observation.shape[0], -1)


def _slice(name: str) -> slice:
    start = 0
    for component, size in TEAM_ACTOR_OBSERVATION_LAYOUT:
        if component == name:
            return slice(start, start + size)
        start += size
    raise KeyError(name)


@pytest.fixture
def task() -> TeamMarlTaskConfig:
    return load_team_marl_run_settings(REPO_ROOT / "configs" / "push_multi.yaml").settings.task


def _row(state: PlanarBodyState, env: int) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=state.position_xy_m[env : env + 1].clone(),
        yaw_rad=state.yaw_rad[env : env + 1].clone(),
        lin_vel_xy_mps=state.lin_vel_xy_mps[env : env + 1].clone(),
        yaw_rate_radps=state.yaw_rate_radps[env : env + 1].clone(),
    )


def _planar(x_m: float, y_m: float) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x_m, y_m]]), yaw_rad=torch.zeros(1),
        lin_vel_xy_mps=torch.zeros(1, 2), yaw_rate_radps=torch.zeros(1),
    )  # fmt: skip


def _inputs(now_ns: int, box_x_m: float = 1.4, robot_x_m: float = 0.7, roll_deg: float = 0.0) -> TeamPolicyInputs:
    return TeamPolicyInputs(
        now_ns=now_ns,
        robot=_planar(robot_x_m, 0.25),
        robot_projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
        robot_roll_pitch_deg=torch.tensor([[roll_deg, 0.0]]),
        box=_planar(box_x_m, 0.0),
    )


def _report(stamp_ns: int, x_m: float) -> PartnerReport:
    return PartnerReport(stamp_ns=stamp_ns, position_xy_m=(x_m, -0.25), yaw_rad=0.0, command=(0.6, 0.0, 0.0))


def _started(task: TeamMarlTaskConfig, policy: RecordingPolicy) -> TeamPolicyDriver:
    driver = TeamPolicyDriver(policy, task, side_sign=1.0)
    driver.start(RETREAT_XY, RETREAT_RADIUS_M, START_NS)
    return driver


def test_observation_matches_the_training_env_for_the_same_state(make_team_marl_env: TeamMarlEnvFactory) -> None:
    # 学習環境で 1 ステップ進めたときの go2_a の観測と、同じ状態を ROS 2 の経路の材料で渡したときの観測が一致すること
    env, simulation = make_team_marl_env()
    simulation.step_hook = lambda: setattr(
        simulation, "robots", [simulation.robots[0], dataclasses.replace(
            simulation.robots[1], position_xy_m=simulation.robots[1].position_xy_m + torch.tensor([0.01, 0.0]))]
    )  # fmt: skip
    robots_at_start = [_row(robot, 0) for robot in simulation.robots]
    box_at_start = _row(simulation.box, 0)
    reset_observation = env.observations[0][0:1].clone()
    env.step((torch.zeros(2, 3), torch.zeros(2, 3)))
    policy = RecordingPolicy()
    driver = TeamPolicyDriver(policy, env.cfg, side_sign=env.cfg.agent_side_signs[0])
    driver.start(RETREAT_XY, RETREAT_RADIUS_M, START_NS)
    gravity = torch.tensor([[0.0, 0.0, -1.0]])
    no_tilt = torch.zeros(1, 2)

    driver.decide(TeamPolicyInputs(START_NS, robots_at_start[0], gravity, no_tilt, box_at_start))
    partner = robots_at_start[1]
    driver.observe_partner(
        PartnerReport(START_NS, (float(partner.position_xy_m[0, 0]), float(partner.position_xy_m[0, 1])),
                      float(partner.yaw_rad[0]), (0.0, 0.0, 0.0))
    )  # fmt: skip
    driver.decide(
        TeamPolicyInputs(START_NS + PERIOD_NS, _row(simulation.robots[0], 0), gravity, no_tilt, _row(simulation.box, 0))
    )

    assert torch.allclose(policy.observations[0], reset_observation)
    assert torch.allclose(policy.observations[1], env.observations[0][0:1])


def test_partner_status_from_the_same_decision_time_is_not_used(task: TeamMarlTaskConfig) -> None:
    policy = RecordingPolicy()
    driver = _started(task, policy)
    driver.observe_partner(_report(START_NS + PERIOD_NS, x_m=0.9))

    decision = driver.decide(_inputs(START_NS + PERIOD_NS))

    assert decision.partner_age_s is None
    assert policy.observations[0][0, _slice("partner_received")].item() == 0.0


def test_latest_partner_status_before_the_decision_is_used_with_its_age(task: TeamMarlTaskConfig) -> None:
    policy = RecordingPolicy()
    driver = _started(task, policy)
    driver.observe_partner(_report(START_NS, x_m=0.5))
    driver.observe_partner(_report(START_NS + PERIOD_NS, x_m=0.6))
    driver.observe_partner(_report(START_NS + 2 * PERIOD_NS, x_m=0.9))

    decision = driver.decide(_inputs(START_NS + 2 * PERIOD_NS))

    assert decision.partner_age_s == pytest.approx(0.1)
    observation = policy.observations[0]
    # 自分（x = 0.7、y = 0.25、+x 向き）から見た相手（x = 0.6、y = -0.25）
    assert observation[0, _slice("partner_rel_pos_xy")].tolist() == pytest.approx([-0.1, -0.5], abs=1e-6)
    assert observation[0, _slice("partner_age")].item() == pytest.approx(0.1 / task.communication.max_age_s)


def test_partner_status_from_before_the_task_is_ignored(task: TeamMarlTaskConfig) -> None:
    driver = _started(task, RecordingPolicy())
    driver.observe_partner(_report(START_NS - PERIOD_NS, x_m=0.5))

    assert driver.decide(_inputs(START_NS + PERIOD_NS)).partner_age_s is None


def test_starting_a_new_task_discards_previous_partner_status_and_command(task: TeamMarlTaskConfig) -> None:
    policy = RecordingPolicy(command=(0.6, 0.0, 0.0))
    driver = _started(task, policy)
    driver.observe_partner(_report(START_NS, x_m=0.5))
    driver.decide(_inputs(START_NS + PERIOD_NS))
    next_start = START_NS + 10 * PERIOD_NS

    driver.start(RETREAT_XY, RETREAT_RADIUS_M, next_start)
    decision = driver.decide(_inputs(next_start + PERIOD_NS))

    assert decision.partner_age_s is None
    assert policy.observations[-1][0, _slice("last_command")].abs().sum() == 0.0


def test_commands_are_clamped_to_the_trained_range(task: TeamMarlTaskConfig) -> None:
    driver = _started(task, RecordingPolicy(command=(2.0, -1.0, 3.0)))

    decision = driver.decide(_inputs(START_NS))

    limits = task.commands
    expected = [limits.vx_mps.high, limits.vy_mps.low, limits.yaw_rate_radps.high]
    assert decision.commands[0].tolist() == pytest.approx(expected)


def test_success_after_the_box_is_held_in_the_retreat_area(task: TeamMarlTaskConfig) -> None:
    driver = _started(task, RecordingPolicy())
    required = task.required_hold_steps()

    statuses = [driver.decide(_inputs(START_NS + step * PERIOD_NS, box_x_m=3.8)).status for step in range(required)]

    assert statuses[:-1] == [TeamTaskStatus.RUNNING] * (required - 1)
    assert statuses[-1] is TeamTaskStatus.SUCCESS


@pytest.mark.parametrize(
    ("inputs_kwargs", "expected"),
    [({"roll_deg": 15.0}, TeamTaskStatus.FALL), ({"robot_x_m": -5.5}, TeamTaskStatus.OUT_OF_BOUNDS)],
    ids=["fall", "out-of-bounds"],
)
def test_own_failure_ends_the_task(task: TeamMarlTaskConfig, inputs_kwargs: dict, expected: TeamTaskStatus) -> None:
    driver = _started(task, RecordingPolicy())

    assert driver.decide(_inputs(START_NS, **inputs_kwargs)).status is expected


def test_time_out_after_the_episode_length(task: TeamMarlTaskConfig) -> None:
    driver = _started(task, RecordingPolicy())
    steps = task.episode.max_episode_length_steps()

    statuses = [driver.decide(_inputs(START_NS + step * PERIOD_NS)).status for step in range(steps + 1)]

    assert statuses[-2] is TeamTaskStatus.RUNNING
    assert statuses[-1] is TeamTaskStatus.TIME_OUT


def test_deciding_before_a_task_is_rejected(task: TeamMarlTaskConfig) -> None:
    driver = TeamPolicyDriver(RecordingPolicy(), task, side_sign=1.0)

    with pytest.raises(NoActiveTaskError):
        driver.decide(_inputs(START_NS))


def test_partner_status_received_before_the_task_starts_is_used(task: TeamMarlTaskConfig) -> None:
    # タスク管理は 2 台に順に依頼を送るので、先に始めた相手の最初の状態が自分の作業の始めより先に届くことがある
    driver = TeamPolicyDriver(RecordingPolicy(), task, side_sign=1.0)
    driver.observe_partner(_report(START_NS, x_m=0.5))

    driver.start(RETREAT_XY, RETREAT_RADIUS_M, START_NS)
    decision = driver.decide(_inputs(START_NS + PERIOD_NS))

    assert decision.partner_age_s == pytest.approx(0.1)


def test_time_out_is_judged_from_the_elapsed_time_even_if_decisions_are_skipped(task: TeamMarlTaskConfig) -> None:
    driver = _started(task, RecordingPolicy())
    steps = task.episode.max_episode_length_steps()

    before = driver.decide(_inputs(START_NS + (steps - 1) * PERIOD_NS))
    after = driver.decide(_inputs(START_NS + steps * PERIOD_NS))

    assert before.status is TeamTaskStatus.RUNNING
    assert after.status is TeamTaskStatus.TIME_OUT
