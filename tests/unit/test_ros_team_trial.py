"""シミュレーションノードの通路の開通の試行の流れ（真値の判定・打ち切り・タスク管理の結果の扱い）の検証。"""

import math

import pytest
import torch

from cooperative_go2.entities.corridor import CorridorConfig
from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, JointTargetTiming
from cooperative_go2.use_cases.ros_team_trial import (
    DIVERGED_BEFORE_RESULT,
    NoTaskResultError,
    NoTeamJointTargetsAppliedError,
    TaskResult,
    TeamTrialPorts,
    TeamTrialRecord,
    TeamTrialRunner,
    TeamTrialSetup,
    TeamWorldState,
)

DT_NS = 20_000_000
JUDGE_STEPS = 5
ROBOTS = ("go2_a", "go2_b")
# 通路 x ∈ [1, 2]・幅 1.4、通行領域 x ∈ [0.3, 2.7]。箱は通路の中、退避領域は (3.8, 0)・半径 0.4
SETUP = TeamTrialSetup(
    robot_names=ROBOTS,
    corridor=CorridorConfig(entrance_x_m=1.0, exit_x_m=2.0, center_y_m=0.0, width_m=1.4, apron_m=0.7),
    box_half_size_xy_m=(0.2, 0.5),
    retreat_xy_m=(3.8, 0.0),
    retreat_radius_m=0.4,
    verify_hold_s=1.0,
    fall_tilt_deg=10.0,
    control_dt_ns=DT_NS,
    judge_steps=JUDGE_STEPS,
    max_trial_ns=4_000_000_000,
)
BOX_IN_CORRIDOR = (1.4, 0.0)
BOX_IN_RETREAT = (3.8, 0.0)


def pose(x: float, y: float, yaw_rad: float = 0.0) -> RigidBodyPose:
    quat = torch.tensor([[math.cos(yaw_rad / 2), 0.0, 0.0, math.sin(yaw_rad / 2)]])
    return RigidBodyPose(torch.tensor([[x, y, 0.3]]), quat, torch.zeros(1, 3), torch.zeros(1, 3))


class FakeTeamWorld:
    """物理を持たず、テストが箱の位置・向きとロボットの傾きを台本で決めるシーン。"""

    def __init__(self) -> None:
        self.box = (*BOX_IN_CORRIDOR, 0.0)
        self.roll_deg = [0.0, 0.0]
        # 試行の始めから数えた物理ステップの数 → そのステップの後の箱の (x, y, yaw)
        self.box_path: dict[int, tuple[float, float, float]] = {}
        # そのステップの後にロボット 1 を傾ける角度 [deg]
        self.tilt_path: dict[int, float] = {}
        self.diverged_steps: set[int] = set()
        self._steps = 0

    def reset(self, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]) -> None:
        self.box = (*BOX_IN_CORRIDOR, 0.0)
        self._steps = 0

    def read(self) -> TeamWorldState:
        go2 = tuple(
            Go2State(
                proprioception=Go2Proprioception(
                    base_ang_vel=torch.zeros(1, 3),
                    projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
                    dof_pos=torch.zeros(1, 12),
                    dof_vel=torch.zeros(1, 12),
                ),
                base_pos_m=torch.tensor([[0.0, 0.0, 0.3]]),
                base_euler_deg=torch.tensor([[roll, 0.0, 0.0]]),
                base_lin_vel=torch.zeros(1, 3),
            )
            for roll in self.roll_deg
        )
        return TeamWorldState(go2=go2, robots=(pose(0.7, 0.25), pose(0.7, -0.25)), box=pose(*self.box))

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        assert len(joint_targets) == len(ROBOTS)

    def step(self) -> None:
        self._steps += 1
        if self._steps in self.box_path:
            self.box = self.box_path[self._steps]
        if self._steps in self.tilt_path:
            self.roll_deg[1] = self.tilt_path[self._steps]

    def diverged(self) -> bool:
        return self._steps in self.diverged_steps


class EchoGaitPublisher:
    """状態を配信すると、歩行方策のノードが即座に目標を返したことにする。silent のロボットは目標を返さない。"""

    def __init__(self, schedulers: tuple[JointTargetScheduler, ...], silent: frozenset[int] = frozenset()) -> None:
        self._schedulers = schedulers
        self._silent = silent

    def publish(self, stamp_ns: int, state: TeamWorldState) -> None:
        for index, scheduler in enumerate(self._schedulers):
            if index not in self._silent:
                scheduler.receive(stamp_ns, torch.zeros(1, 12))


class ScriptedPeers:
    """タスク管理の結果を、何回目の問い合わせで返すかを台本で決める。None なら返さない。"""

    def __init__(self, result: TaskResult | None, after_polls: int) -> None:
        self._result = result
        self._after_polls = after_polls
        self._polls = 0
        self.ended = False

    def reset_gait_policies(self) -> None:
        pass

    def start_task(self, start_stamp_ns: int, retreat_xy_m: tuple[float, float], retreat_radius_m: float) -> None:
        self._polls = 0

    def find_task_result_or_none(self) -> TaskResult | None:
        self._polls += 1
        return self._result if self._polls > self._after_polls else None

    def find_first_push_stamps_ns(self) -> dict[str, int]:
        return {}

    def end_task(self) -> None:
        self.ended = True


class NoWaitPacer:
    def wait_for_next_step(self) -> None:
        pass


def run(world: FakeTeamWorld, peers: ScriptedPeers, silent: frozenset[int] = frozenset()) -> TeamTrialRecord:
    timing = JointTargetTiming(control_dt_ns=DT_NS, latency_steps=1, expiry_ns=5 * DT_NS)
    schedulers = tuple(JointTargetScheduler(timing, torch.zeros(1, 12)) for _ in ROBOTS)
    ports = TeamTrialPorts(
        world=world,
        publisher=EchoGaitPublisher(schedulers, silent),
        peers=peers,
        pacer=NoWaitPacer(),
        wall_clock=lambda: 0.0,
    )
    runner = TeamTrialRunner(ports, schedulers, SETUP)
    zeros = tuple(torch.zeros(1, 12) for _ in ROBOTS)
    return runner.run_trial(0, zeros, tuple(torch.zeros(1, 2) for _ in ROBOTS))


SUCCESS = TaskResult(success=True, termination_reason="SUCCESS")


def test_box_held_in_retreat_area_counts_as_success() -> None:
    world = FakeTeamWorld()
    world.box_path = {10: (*BOX_IN_RETREAT, 0.0)}

    record = run(world, ScriptedPeers(SUCCESS, after_polls=30))

    # 0.2 s に入り、判定の周期（0.1 s）ごとに測って 1.0 s 続いた 1.2 s に成功とみなす
    assert record.success is True
    assert record.success_time_s == pytest.approx(1.2)
    assert record.corridor_opened is True
    assert record.open_time_s == pytest.approx(0.2)


def test_leaving_retreat_area_before_hold_time_resets_the_hold() -> None:
    world = FakeTeamWorld()
    world.box_path = {10: (*BOX_IN_RETREAT, 0.0), 40: (3.0, 0.0), 60: (*BOX_IN_RETREAT, 0.0)}

    record = run(world, ScriptedPeers(SUCCESS, after_polls=15))

    # 0.2〜0.8 s は 0.6 s で出たので数えず、1.2 s に入り直してから試行を終えた 1.6 s までは 1.0 s 続いていない
    assert record.end_time_s == pytest.approx(1.6)
    assert record.success is False
    assert record.success_time_s is None


def test_task_result_is_recorded_separately_from_ground_truth() -> None:
    world = FakeTeamWorld()

    record = run(world, ScriptedPeers(TaskResult(success=True, termination_reason="SUCCESS"), after_polls=3))

    assert record.task_success is True
    assert record.task_reason == "SUCCESS"
    assert record.success is False
    assert record.end_time_s == pytest.approx(0.4)


def test_missing_task_result_is_an_error_instead_of_a_failed_trial() -> None:
    world = FakeTeamWorld()
    peers = ScriptedPeers(None, after_polls=0)

    # タスク管理は自分の時間切れで必ず結果を返すので、届かないのはノードが動いていないこと（評価の失敗に混ぜない）
    with pytest.raises(NoTaskResultError):
        run(world, peers)
    assert peers.ended is True


def test_divergence_stops_the_trial() -> None:
    world = FakeTeamWorld()
    world.diverged_steps = {7}

    record = run(world, ScriptedPeers(None, after_polls=0))

    # 発散したステップを含む判定の周期（0.2 s）の終わりで止める
    assert record.diverged is True
    assert record.task_reason == DIVERGED_BEFORE_RESULT
    assert record.end_time_s == pytest.approx(0.2)


def test_box_yaw_is_folded_because_box_front_and_back_are_alike() -> None:
    world = FakeTeamWorld()
    world.box_path = {5: (*BOX_IN_CORRIDOR, math.pi), 10: (*BOX_IN_CORRIDOR, math.pi - 0.2)}

    record = run(world, ScriptedPeers(SUCCESS, after_polls=4))

    # 180° 回った箱は回っていないのと同じで、π - 0.2 rad は 0.2 rad（約 11.5°）回ったとみなす
    assert record.box_max_abs_yaw_deg == pytest.approx(math.degrees(0.2), abs=0.01)


def test_tilted_robot_counts_as_fall() -> None:
    world = FakeTeamWorld()
    world.tilt_path = {8: 12.0}

    record = run(world, ScriptedPeers(SUCCESS, after_polls=4))

    assert record.falls == 1


def test_robot_without_any_joint_target_is_an_error() -> None:
    with pytest.raises(NoTeamJointTargetsAppliedError):
        run(FakeTeamWorld(), ScriptedPeers(SUCCESS, after_polls=4), silent=frozenset({1}))
