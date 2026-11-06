"""ROS 2 の実行系で、シミュレーションノードが通路の開通の試行（1 台・2 台）を 1 つずつ進める流れ。

物理ステップを進めるのはこの流れだけで、歩行方策・協調・タスク管理は別のプロセスが ROS 2 の通信で動かす。
物理は方策の応答を待たずに制御周期ごとに進め、ロボットごとに届いた関節目標を JointTargetScheduler で選んで適用する。
試行ごとにタスク管理へ通路の開通を依頼し、その結果とは別に、シミュレーションの真値で通路の開通・箱が退避領域に
入ったまま続いたか・転倒を判定して記録する（タスク管理の判定の誤りを評価指標に混ぜないため）。
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol

import torch

from cooperative_go2.entities.command_expiry import ns_to_seconds
from cooperative_go2.entities.corridor import BoxFootprint, CorridorConfig, is_corridor_open
from cooperative_go2.entities.go2_state import Go2State
from cooperative_go2.entities.planar_frame import yaw_from_quat_wxyz
from cooperative_go2.entities.push_success import is_in_goal_region
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, TargetSelection

ROLL_COLUMN, PITCH_COLUMN = 0, 1
# タスク管理の結果が届く前に物理計算が発散して試行を止めたときの、タスク管理の終わった理由の代わりの記録
DIVERGED_BEFORE_RESULT = "DIVERGED_BEFORE_RESULT"


class NoTeamJointTargetsAppliedError(RuntimeError):
    """試行の間に、いずれかのロボットの歩行方策の関節目標を一度も適用できなかったことを表す。

    そのロボットの歩行方策のノードが動いていない、または状態や目標が届いていない。既定姿勢のまま終わった試行を、
    制御の成績として記録しないために止める。
    """


class NoTaskResultError(RuntimeError):
    """試行を打ち切る時間（max_trial_ns）までに、タスク管理の結果が届かなかったことを表す。

    タスク管理は自分の時間切れで必ず結果を返すので、届かないのはタスク管理か協調のノードが動いていないことを表す。
    評価の失敗の試行として記録せず、実行を止める。
    """


@dataclass(frozen=True)
class TeamWorldState:
    """ある時点の、全ロボットと箱の状態（シミュレーションの真値）。テンソルの先頭の次元は 1。

    Attributes:
        go2: ロボットごとの Go2 の状態（関節・傾きなど）。シーンの robots と同じ順。
        robots: ロボットごとの胴体の位置・姿勢・速度。
        box: 箱の位置・姿勢・速度。
    """

    go2: tuple[Go2State, ...]
    robots: tuple[RigidBodyPose, ...]
    box: RigidBodyPose


class SimulatedTeamWorld(Protocol):
    """通路のシーン（並列環境 1 つ）。物理ステップを進められるのはこの操作だけ。テンソルは CPU に置く。"""

    def reset(self, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]) -> None:
        """各ロボットを開始位置 robot_xy_m (1, 2)・関節角 dof_pos (1, 12) で、箱を初期位置に置く。"""
        ...

    def read(self) -> TeamWorldState:
        """現在の状態を返す。"""
        ...

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        """各ロボットの PD 制御の目標関節角 (1, 12) [rad] を設定する。"""
        ...

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。"""
        ...

    def diverged(self) -> bool:
        """直前の物理ステップで物理計算が発散したか。"""
        ...


class TeamStatePublisher(Protocol):
    """シミュレーション時刻と状態を、方策のノードへ配信する。"""

    def publish(self, stamp_ns: int, state: TeamWorldState) -> None:
        """時刻 stamp_ns [ns] の状態を配信する。時刻（/clock）を先に配信する。"""
        ...


@dataclass(frozen=True)
class TaskResult:
    """タスク管理が返した、通路の開通の依頼の結果。

    Attributes:
        success: 全ロボットが成功を返した。
        termination_reason: 終わった理由（SUCCESS、FAILED(...)、TIMEOUT など）。
    """

    success: bool
    termination_reason: str


class TeamTaskPeers(Protocol):
    """試行の始めと終わりに、方策・協調・タスク管理のノードとやり取りする。"""

    def reset_gait_policies(self) -> None:
        """全ロボットの歩行方策のノードに、ロボットを初期姿勢に戻したことを伝え、受け取られるまで待つ。"""
        ...

    def start_task(self, start_stamp_ns: int, retreat_xy_m: tuple[float, float], retreat_radius_m: float) -> None:
        """タスク管理に、時刻 start_stamp_ns [ns] に始めた試行の通路の開通を依頼し、受け付けられるまで待つ。"""
        ...

    def find_task_result_or_none(self) -> TaskResult | None:
        """依頼の結果が届いていれば返す。まだなら None。

        タスク管理が依頼を結果を返さずに終えた（中断・取り消し）ときは、試行の失敗ではなく実行の誤りとして例外を送出する。
        """
        ...

    def find_first_push_stamps_ns(self) -> dict[str, int]:
        """今の試行で、各ロボットが初めて PUSH を配信した時刻 [ns]（まだ押していないロボットは含めない）。"""
        ...

    def end_task(self) -> None:
        """依頼がまだ終わっていなければ取り消す。"""
        ...


class StepPacer(Protocol):
    """物理ステップの実時間の間隔を整える。"""

    def wait_for_next_step(self) -> None:
        """次の物理ステップを進めてよい実時間まで待つ。"""
        ...


@dataclass(frozen=True)
class TeamTrialSetup:
    """試行の条件。

    Attributes:
        robot_names: ロボットの名前。シーンの robots と同じ順。
        corridor: 通路の形（開通の判定に使う）。
        box_half_size_xy_m: 箱の x 軸・y 軸方向の寸法の半分 [m]。
        retreat_xy_m: 退避領域の中心 [m]。
        retreat_radius_m: 退避領域の半径 [m]。
        verify_hold_s: 箱の中心が退避領域に入ったままこの時間 [s] 続いたら、真値で成功とみなす。
        fall_tilt_deg: ロールかピッチの絶対値がこの角度 [deg] を超えたら転倒とみなす。
        control_dt_ns: 歩行の制御周期（1 物理ステップ）のシミュレーション時間 [ns]。
        judge_steps: 真値の判定を行う物理ステップの間隔。
        max_trial_ns: 依頼の結果が届かなくても試行を打ち切る時間 [ns]。タスク管理の時間切れより長くする。
    """

    robot_names: tuple[str, ...]
    corridor: CorridorConfig
    box_half_size_xy_m: tuple[float, float]
    retreat_xy_m: tuple[float, float]
    retreat_radius_m: float
    verify_hold_s: float
    fall_tilt_deg: float
    control_dt_ns: int
    judge_steps: int
    max_trial_ns: int


@dataclass(frozen=True)
class TeamTrialRecord:
    """1 試行の結果（シミュレーションの真値とタスク管理の結果）と、ROS 2 の経路の計測値。

    Attributes:
        trial_index: 試行の番号。
        success: 真値で、箱の中心が退避領域に入ったまま verify_hold_s 続いた。
        task_success: タスク管理が成功を返した。結果が届く前に物理計算が発散したら False。
        task_reason: タスク管理が返した終わった理由。結果が届く前に物理計算が発散したら DIVERGED_BEFORE_RESULT。
        corridor_opened: 真値で、試行中に通路が開いた。
        open_time_s: 通路が初めて開いた時刻（試行の始めから）[s]。開かなければ None。
        success_time_s: 真値で成功とみなした時刻 [s]。成功しなければ None。
        end_time_s: 試行を終えた時刻 [s]。
        push_start_times_s: ロボットごとの、初めて PUSH を配信した時刻 [s]（押さなければ None）。
        box_dx_m: 箱の +x 方向の移動量 [m]。
        box_max_abs_yaw_deg: 試行中の、退避領域への向きから見た箱の回転の絶対値の最大 [deg]（前後を同じとみなす）。
        falls: 転倒したロボットの数。
        diverged: 物理計算が発散した。
        target_selections: 物理ステップごとに適用した目標の出所の数（全ロボットの合計）。
        target_delay_steps: 歩行方策の目標を適用したときの元の状態からの遅れ（制御周期の数）ごとの数（全ロボットの合計）。
        wall_time_s: 試行の実時間 [s]。
    """

    trial_index: int
    success: bool
    task_success: bool
    task_reason: str
    corridor_opened: bool
    open_time_s: float | None
    success_time_s: float | None
    end_time_s: float
    push_start_times_s: tuple[float | None, ...]
    box_dx_m: float
    box_max_abs_yaw_deg: float
    falls: int
    diverged: bool
    target_selections: dict[TargetSelection, int]
    target_delay_steps: dict[int, int]
    wall_time_s: float


@dataclass(frozen=True)
class TeamTrialPorts:
    """試行の流れが使う外部とのやり取り。"""

    world: SimulatedTeamWorld
    publisher: TeamStatePublisher
    peers: TeamTaskPeers
    pacer: StepPacer
    wall_clock: Callable[[], float]


@dataclass
class _Progress:
    """試行の途中の真値の判定の記録。"""

    start_box_x_m: float
    open_ns: int | None = None
    in_retreat_since_ns: int | None = None
    success_ns: int | None = None
    max_abs_yaw_rad: float = 0.0
    fallen: set[int] = field(default_factory=set)


class TeamTrialRunner:
    """試行を 1 つずつ進める。シミュレーション時刻は試行をまたいで増え続ける（/clock を巻き戻さない）。"""

    def __init__(
        self, ports: TeamTrialPorts, schedulers: tuple[JointTargetScheduler, ...], setup: TeamTrialSetup
    ) -> None:
        """外部とのやり取り、ロボットごとの関節目標の選び方（setup.robot_names と同じ順）、試行の条件を受け取る。"""
        self._ports = ports
        self._schedulers = schedulers
        self._setup = setup
        self._now_ns = 0

    def run_trial(
        self, trial_index: int, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]
    ) -> TeamTrialRecord:
        """各ロボットを開始位置・関節角で置いて 1 試行を走らせる。

        依頼の結果が届く・max_trial_ns に達する・物理計算が発散する、のいずれかまで物理を進める。

        Raises:
            NoTeamJointTargetsAppliedError: いずれかのロボットの関節目標を一度も適用できなかった。
            NoTaskResultError: 物理計算が発散しないまま、max_trial_ns までにタスク管理の結果が届かなかった。
        """
        ports, setup = self._ports, self._setup
        self._now_ns += setup.control_dt_ns
        start_ns = self._now_ns
        ports.world.reset(dof_pos, robot_xy_m)
        for scheduler in self._schedulers:
            scheduler.start_episode(start_ns)
        ports.peers.reset_gait_policies()
        start_state = ports.world.read()
        ports.publisher.publish(start_ns, start_state)
        ports.peers.start_task(start_ns, setup.retreat_xy_m, setup.retreat_radius_m)
        wall_start = ports.wall_clock()
        progress = _Progress(start_box_x_m=float(start_state.box.position_m[0, 0]))
        selections: dict[TargetSelection, int] = {}
        delays: dict[int, int] = {}
        policy_steps = [0] * len(self._schedulers)
        task_result: TaskResult | None = None
        diverged = False
        state = start_state
        while task_result is None and self._now_ns - start_ns < setup.max_trial_ns and not diverged:
            for _ in range(setup.judge_steps):
                ports.pacer.wait_for_next_step()
                targets = []
                for robot_index, scheduler in enumerate(self._schedulers):
                    selected = scheduler.select(self._now_ns)
                    selections[selected.selection] = selections.get(selected.selection, 0) + 1
                    if selected.selection is TargetSelection.POLICY:
                        policy_steps[robot_index] += 1
                    if selected.delay_steps is not None:
                        delays[selected.delay_steps] = delays.get(selected.delay_steps, 0) + 1
                    targets.append(selected.joint_targets)
                ports.world.apply_joint_targets(tuple(targets))
                ports.world.step()
                diverged = diverged or ports.world.diverged()
                self._now_ns += setup.control_dt_ns
                state = ports.world.read()
                ports.publisher.publish(self._now_ns, state)
            self._judge(state, progress)
            task_result = ports.peers.find_task_result_or_none()
        push_stamps = ports.peers.find_first_push_stamps_ns()
        ports.peers.end_task()
        if task_result is None and not diverged:
            raise NoTaskResultError(
                f"試行 {trial_index} で {ns_to_seconds(setup.max_trial_ns)} s 以内にタスク管理の結果が届かなかった"
            )
        for name, steps in zip(setup.robot_names, policy_steps, strict=True):
            if steps == 0:
                raise NoTeamJointTargetsAppliedError(f"試行 {trial_index} で {name} の関節目標を一度も適用できなかった")
        return self._record(trial_index, start_ns, state, progress, task_result, push_stamps, diverged,
                            selections, delays, ports.wall_clock() - wall_start)  # fmt: skip

    def _judge(self, state: TeamWorldState, progress: _Progress) -> None:
        setup = self._setup
        box_xy = state.box.position_m[:, :2]
        box_yaw = yaw_from_quat_wxyz(state.box.quat_wxyz)
        footprint = BoxFootprint(center_xy_m=box_xy, yaw_rad=box_yaw, half_size_xy_m=setup.box_half_size_xy_m)
        if progress.open_ns is None and bool(is_corridor_open(footprint, setup.corridor)[0]):
            progress.open_ns = self._now_ns
        retreat = torch.tensor([setup.retreat_xy_m], dtype=box_xy.dtype)
        if bool(is_in_goal_region(box_xy, retreat, setup.retreat_radius_m)[0]):
            if progress.in_retreat_since_ns is None:
                progress.in_retreat_since_ns = self._now_ns
            held_s = ns_to_seconds(self._now_ns - progress.in_retreat_since_ns)
            if progress.success_ns is None and held_s >= setup.verify_hold_s:
                progress.success_ns = self._now_ns
        else:
            progress.in_retreat_since_ns = None
        to_retreat = retreat - box_xy
        push_angle = float(torch.atan2(to_retreat[0, 1], to_retreat[0, 0]))
        # 箱は前後が同じ形なので、回転を [-π/2, π/2) に折り返してから測る
        yaw_error = math.remainder(float(box_yaw[0]) - push_angle, math.pi)
        progress.max_abs_yaw_rad = max(progress.max_abs_yaw_rad, abs(yaw_error))
        for robot_index, go2 in enumerate(state.go2):
            tilt = go2.base_euler_deg[0, [ROLL_COLUMN, PITCH_COLUMN]].abs().max()
            if float(tilt) > setup.fall_tilt_deg:
                progress.fallen.add(robot_index)

    def _record(
        self, trial_index: int, start_ns: int, state: TeamWorldState, progress: _Progress,
        task_result: TaskResult | None, push_stamps: dict[str, int], diverged: bool,
        selections: dict[TargetSelection, int], delays: dict[int, int], wall_time_s: float,
    ) -> TeamTrialRecord:  # fmt: skip
        def since_start(stamp_ns: int | None) -> float | None:
            return None if stamp_ns is None else round(ns_to_seconds(stamp_ns - start_ns), 3)

        return TeamTrialRecord(
            trial_index=trial_index,
            success=progress.success_ns is not None,
            task_success=task_result is not None and task_result.success,
            task_reason=DIVERGED_BEFORE_RESULT if task_result is None else task_result.termination_reason,
            corridor_opened=progress.open_ns is not None,
            open_time_s=since_start(progress.open_ns),
            success_time_s=since_start(progress.success_ns),
            end_time_s=round(ns_to_seconds(self._now_ns - start_ns), 3),
            push_start_times_s=tuple(since_start(push_stamps.get(name)) for name in self._setup.robot_names),
            box_dx_m=round(float(state.box.position_m[0, 0]) - progress.start_box_x_m, 4),
            box_max_abs_yaw_deg=round(math.degrees(progress.max_abs_yaw_rad), 2),
            falls=len(progress.fallen),
            diverged=diverged,
            target_selections=selections,
            target_delay_steps=delays,
            wall_time_s=wall_time_s,
        )

