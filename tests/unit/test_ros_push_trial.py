"""シミュレーションノードの試行の流れ（物理の進め方・関節目標の遅れ・終了の判定・時刻の扱い）の検証。"""

import dataclasses
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State
from cooperative_go2.entities.push_layout import PushLayout
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.interface_adapters.config_loader import load_push_training_settings
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, JointTargetTiming, TargetSelection
from cooperative_go2.use_cases.ros_push_trial import (
    InvalidPushTrialTimingError,
    NoJointTargetsAppliedError,
    PushTrialPorts,
    PushTrialRunner,
    PushTrialTiming,
    PushWorldState,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DT_NS = 20_000_000
GAIT_STEPS = 5
DEFAULT_TARGETS = torch.zeros(1, 12)
GOAL_XY = (2.0, 0.0)


def _pose(x: float, y: float) -> RigidBodyPose:
    upright = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    return RigidBodyPose(torch.tensor([[x, y, 0.3]]), upright, torch.zeros(1, 3), torch.zeros(1, 3))


class FakeWorld:
    """物理を持たず、テストが箱の位置と傾きを決めるシーン。適用された目標を記録する。"""

    def __init__(self) -> None:
        self.box_xy = (1.0, 0.0)
        self.roll_deg = 0.0
        self.applied: list[torch.Tensor] = []
        self.steps = 0
        # 試行の始めから数えた物理ステップの数 → そのステップの後の箱の位置
        self.box_path: dict[int, tuple[float, float]] = {}
        # 発散を報告する、試行の始めから数えた物理ステップの番号（1 始まり）
        self.diverged_steps: set[int] = set()
        self._trial_steps = 0

    def reset(self, layout: PushLayout, dof_pos: torch.Tensor) -> None:
        self.box_xy = tuple(layout.box_xy_m[0].tolist())
        self._trial_steps = 0

    def read(self) -> PushWorldState:
        go2 = Go2State(
            proprioception=Go2Proprioception(
                base_ang_vel=torch.zeros(1, 3),
                projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
                dof_pos=torch.zeros(1, 12),
                dof_vel=torch.zeros(1, 12),
            ),
            base_pos_m=torch.tensor([[0.0, 0.0, 0.3]]),
            base_euler_deg=torch.tensor([[self.roll_deg, 0.0, 0.0]]),
            base_lin_vel=torch.zeros(1, 3),
        )
        box = _pose(*self.box_xy)
        return PushWorldState(go2=go2, robot=_pose(0.0, 0.0), box=box, box_contact=torch.tensor([False]))

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        self.applied.append(joint_targets.clone())

    def step(self) -> None:
        self.steps += 1
        self._trial_steps += 1
        if self._trial_steps in self.box_path:
            self.box_xy = self.box_path[self._trial_steps]

    def diverged(self) -> bool:
        return self._trial_steps in self.diverged_steps


class EchoGaitPublisher:
    """状態を配信すると、歩行方策のノードが即座にその時刻を値に持つ目標を返したことにする。"""

    def __init__(self, scheduler: JointTargetScheduler) -> None:
        self.scheduler = scheduler
        self.stamps: list[int] = []
        # この時刻より後の状態には目標を返さない（歩行方策のノードが止まったことにする）。None なら常に返す
        self.silent_after_ns: int | None = None

    def publish(self, stamp_ns: int, state: PushWorldState) -> None:
        self.stamps.append(stamp_ns)
        if self.silent_after_ns is None or stamp_ns <= self.silent_after_ns:
            self.scheduler.receive(stamp_ns, torch.full((1, 12), float(stamp_ns)))


class RecordingPeers:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def reset_gait_policy(self) -> None:
        self.calls.append(("reset",))

    def start_task(self, start_stamp_ns: int, goal_xy_m: tuple[float, float], goal_radius_m: float) -> None:
        self.calls.append(("start", start_stamp_ns, goal_xy_m))

    def end_task(self) -> None:
        self.calls.append(("end",))


class NoWaitPacer:
    def wait_for_next_step(self) -> None:
        pass


@dataclasses.dataclass
class Harness:
    runner: PushTrialRunner
    world: FakeWorld
    publisher: EchoGaitPublisher
    peers: RecordingPeers


@pytest.fixture
def task() -> PushTaskConfig:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single_wide.yaml").task


@pytest.fixture
def harness(task: PushTaskConfig) -> Harness:
    scheduler = JointTargetScheduler(JointTargetTiming(DT_NS, latency_steps=1, expiry_ns=5 * DT_NS), DEFAULT_TARGETS)
    world = FakeWorld()
    publisher = EchoGaitPublisher(scheduler)
    peers = RecordingPeers()
    ports = PushTrialPorts(world, publisher, peers, NoWaitPacer(), wall_clock=lambda: 0.0)
    runner = PushTrialRunner(ports, scheduler, task, PushTrialTiming(DT_NS, GAIT_STEPS))
    return Harness(runner, world, publisher, peers)


def _layout(box_xy: tuple[float, float]) -> PushLayout:
    return PushLayout(box_xy_m=torch.tensor([box_xy]), goal_xy_m=torch.tensor([GOAL_XY]))


def test_box_held_in_goal_succeeds_after_hold_time(harness: Harness, task: PushTaskConfig) -> None:
    record = harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.SUCCESS
    assert record.result.high_level_steps == task.required_hold_steps()
    assert harness.world.steps == task.required_hold_steps() * GAIT_STEPS


def test_peers_are_reset_then_given_goal_then_released(harness: Harness) -> None:
    harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    start_ns = harness.publisher.stamps[0]
    assert harness.peers.calls == [("reset",), ("start", start_ns, GOAL_XY), ("end",)]


def test_targets_are_applied_one_control_period_after_their_state(harness: Harness) -> None:
    harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    applied = harness.world.applied
    stamps = harness.publisher.stamps
    assert torch.equal(applied[0], DEFAULT_TARGETS)
    # 物理ステップ k（k >= 1）には、その 1 周期前の状態（stamps[k - 1]）から作った目標を適用する
    for step in range(1, len(applied)):
        assert applied[step][0, 0].item() == pytest.approx(float(stamps[step - 1]))


def test_selection_counts_and_delays_are_recorded(harness: Harness) -> None:
    record = harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    total_steps = harness.world.steps
    assert record.target_selections[TargetSelection.EPISODE_START] == 1
    assert record.target_selections[TargetSelection.POLICY] == total_steps - 1
    assert record.target_delay_steps == {1: total_steps - 1}


def test_sim_time_keeps_increasing_across_trials(harness: Harness) -> None:
    harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))
    last_stamp_of_first = harness.publisher.stamps[-1]

    harness.runner.run_trial(1, _layout(GOAL_XY), torch.zeros(1, 12))

    stamps = harness.publisher.stamps
    assert all(later > earlier for earlier, later in zip(stamps, stamps[1:], strict=False))
    assert harness.peers.calls[3] == ("reset",)
    assert harness.peers.calls[4][1] == last_stamp_of_first + DT_NS


def test_tilted_robot_ends_trial_as_fall(harness: Harness) -> None:
    harness.world.roll_deg = 45.0

    record = harness.runner.run_trial(0, _layout((1.0, 0.0)), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.FALL
    assert record.result.falls == 1
    assert record.result.high_level_steps == 1


def test_trial_times_out_when_box_never_reaches_goal(harness: Harness, task: PushTaskConfig) -> None:
    record = harness.runner.run_trial(0, _layout((1.0, 0.0)), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.TIME_OUT
    assert record.result.high_level_steps == task.episode.max_episode_length_steps()


def test_box_leaving_arena_ends_trial_as_out_of_bounds(harness: Harness) -> None:
    harness.world.box_path = {GAIT_STEPS * 2: (100.0, 0.0)}

    record = harness.runner.run_trial(0, _layout((1.0, 0.0)), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.OUT_OF_BOUNDS
    assert record.result.high_level_steps == 2


def test_divergence_in_middle_of_high_level_period_ends_trial(harness: Harness) -> None:
    # 高レベル方策の 1 周期（5 物理ステップ）の途中の 2 ステップ目だけ発散を報告する
    harness.world.diverged_steps = {GAIT_STEPS + 2}

    record = harness.runner.run_trial(0, _layout((1.0, 0.0)), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.SIM_DIVERGED
    assert record.result.high_level_steps == 2


def test_success_hold_restarts_when_box_leaves_goal(harness: Harness, task: PushTaskConfig) -> None:
    required = task.required_hold_steps()
    # 必要な数の 1 つ手前（required - 1 周期）まで目標領域に入れ、required 周期目の終わりに外へ出し、
    # 次の周期の終わりにまた入れる。入れ直してから required 周期続けて初めて成功する
    leave_at = required * GAIT_STEPS
    harness.world.box_path = {leave_at: (1.0, 0.0), leave_at + GAIT_STEPS: GOAL_XY}

    record = harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    assert record.result.termination_reason is PushTerminationReason.SUCCESS
    assert record.result.high_level_steps == required + required


def test_second_trial_counts_high_level_steps_from_zero(harness: Harness, task: PushTaskConfig) -> None:
    first = harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))
    second = harness.runner.run_trial(1, _layout(GOAL_XY), torch.zeros(1, 12))

    assert first.result.high_level_steps == second.result.high_level_steps == task.required_hold_steps()


def test_gait_node_loss_holds_default_pose_and_is_counted_as_stale(harness: Harness) -> None:
    start_ns = DT_NS
    harness.publisher.silent_after_ns = start_ns + 10 * DT_NS

    record = harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))

    assert record.target_selections[TargetSelection.STALE] > 0
    assert torch.equal(harness.world.applied[-1], DEFAULT_TARGETS)
    assert max(record.target_delay_steps) <= 5


def test_trial_without_any_gait_target_is_an_error(harness: Harness) -> None:
    harness.publisher.silent_after_ns = 0

    with pytest.raises(NoJointTargetsAppliedError):
        harness.runner.run_trial(0, _layout(GOAL_XY), torch.zeros(1, 12))


@pytest.mark.parametrize("control_dt_ns,gait_steps", [(0, 5), (DT_NS, 0)], ids=["zero_dt", "zero_gait_steps"])
def test_invalid_trial_timing_is_rejected(control_dt_ns: int, gait_steps: int) -> None:
    with pytest.raises(InvalidPushTrialTimingError):
        PushTrialTiming(control_dt_ns, gait_steps)
