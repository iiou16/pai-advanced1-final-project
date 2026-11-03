"""ROS 2 の実行系で、シミュレーションノードが 1 台の箱押しの試行を 1 つずつ進める流れ。

物理ステップを進めるのはこの流れだけで、歩行方策と高レベル方策は別のプロセスが ROS 2 の通信で動かす。
物理は方策の応答を待たずに制御周期ごとに進め、届いた関節目標を JointTargetScheduler で選んで適用する。
方策のノードが止まると目標が古くなり、ロボットは既定姿勢を保って止まる。
成功と終了は、学習環境と同じ判定（judge_push_step）を高レベル方策の周期でシミュレーションの真値に対して行う。
"""

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.command_expiry import ns_to_seconds
from cooperative_go2.entities.go2_state import Go2State
from cooperative_go2.entities.push_episode_judge import PushJudgeInputs, judge_push_step
from cooperative_go2.entities.push_layout import PushLayout
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.push_termination import PushTerminationReason, detect_divergence
from cooperative_go2.entities.push_trial import PlanarPose, PushEpisodeResult
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, TargetSelection


class InvalidPushTrialTimingError(ValueError):
    """試行の時間の条件が不正であることを表す。"""


class NoJointTargetsAppliedError(RuntimeError):
    """試行の間に、歩行方策の関節目標を一度も適用できなかったことを表す。

    歩行方策のノードが動いていない、または状態や目標が届いていない。既定姿勢のまま終わった試行を、方策の成績として
    記録しないために止める。
    """


@dataclass(frozen=True)
class PushWorldState:
    """1 台分の、ある時点の Go2 と箱の状態（シミュレーションの真値）。テンソルの先頭の次元は 1。

    Attributes:
        go2: Go2 の状態（関節・傾きなど）。
        robot: Go2 の胴体の位置・姿勢・速度。
        box: 箱の位置・姿勢・速度。
        box_contact: Go2 が箱に触れているか (1,)。
    """

    go2: Go2State
    robot: RigidBodyPose
    box: RigidBodyPose
    box_contact: torch.Tensor


class SimulatedPushWorld(Protocol):
    """1 台の箱押しのシーン（並列環境 1 つ）。物理ステップを進められるのはこの操作だけ。テンソルは CPU に置く。"""

    def reset(self, layout: PushLayout, dof_pos: torch.Tensor) -> None:
        """Go2 を初期位置・関節角 dof_pos (1, 12) で、箱と目標を layout (1 行) の位置に置く。"""
        ...

    def read(self) -> PushWorldState:
        """現在の状態を返す。"""
        ...

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (1, 12) [rad] を設定する。"""
        ...

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。"""
        ...

    def diverged(self) -> bool:
        """直前の物理ステップで物理計算が発散したか。"""
        ...


class WorldStatePublisher(Protocol):
    """シミュレーション時刻と状態を、方策のノードへ配信する。"""

    def publish(self, stamp_ns: int, state: PushWorldState) -> None:
        """時刻 stamp_ns [ns] の状態を配信する。時刻（/clock）を先に配信する。"""
        ...


class PushTaskPeers(Protocol):
    """試行の始めと終わりに、方策のノードとやり取りする。"""

    def reset_gait_policy(self) -> None:
        """歩行方策のノードに、ロボットを初期姿勢に戻したことを伝え、受け取られるまで待つ。"""
        ...

    def start_task(self, start_stamp_ns: int, goal_xy_m: tuple[float, float], goal_radius_m: float) -> None:
        """高レベル方策のノードに、時刻 start_stamp_ns [ns] に始めた試行の目標を依頼し、受け付けられるまで待つ。"""
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
class PushTrialTiming:
    """試行の時間の条件。

    Attributes:
        control_dt_ns: 歩行の制御周期（1 物理ステップ）のシミュレーション時間 [ns]。
        gait_steps_per_high_level_step: 高レベル方策の 1 周期の物理ステップの数。成功と終了をこの間隔で判定する。
    """

    control_dt_ns: int
    gait_steps_per_high_level_step: int

    def __post_init__(self) -> None:
        if self.control_dt_ns <= 0 or self.gait_steps_per_high_level_step < 1:
            raise InvalidPushTrialTimingError(
                f"制御周期は正、高レベル方策の 1 周期の物理ステップは 1 以上が必要: "
                f"{self.control_dt_ns} ns, {self.gait_steps_per_high_level_step}"
            )


@dataclass(frozen=True)
class PushTrialRecord:
    """1 試行の結果と、ROS 2 の経路の計測値。

    Attributes:
        result: 試行の結果（学習環境での評価と同じ形式）。
        target_selections: 物理ステップごとに適用した目標の出所の数。
        target_delay_steps: 歩行方策の目標を適用したときの、元の状態からの遅れ（制御周期の数）ごとの物理ステップの数。
        sim_time_s: 試行のシミュレーション時間 [s]。
    """

    result: PushEpisodeResult
    target_selections: Counter[TargetSelection]
    target_delay_steps: Counter[int]
    sim_time_s: float


@dataclass(frozen=True)
class PushTrialPorts:
    """試行の流れが使う外部とのやり取り。

    Attributes:
        world: シーン。
        publisher: 状態の配信。
        peers: 方策のノードとのやり取り。
        pacer: 実時間の間隔の調整。
        wall_clock: 実時間 [s] を返す関数。
    """

    world: SimulatedPushWorld
    publisher: WorldStatePublisher
    peers: PushTaskPeers
    pacer: StepPacer
    wall_clock: Callable[[], float]


class PushTrialRunner:
    """試行を 1 つずつ進める。シミュレーション時刻は試行をまたいで増え続ける（/clock を巻き戻さない）。"""

    def __init__(
        self, ports: PushTrialPorts, scheduler: JointTargetScheduler, task: PushTaskConfig, timing: PushTrialTiming
    ) -> None:
        """外部とのやり取り、関節目標の選び方、箱押しの設定（成功と終了の条件）、時間の条件を受け取る。"""
        self._ports = ports
        self._scheduler = scheduler
        self._task = task
        self._timing = timing
        self._now_ns = 0

    @property
    def now_ns(self) -> int:
        """現在のシミュレーション時刻 [ns]。"""
        return self._now_ns

    def run_trial(self, trial_index: int, layout: PushLayout, reset_dof_pos: torch.Tensor) -> PushTrialRecord:
        """配置 layout とリセット時の関節角 reset_dof_pos (1, 12) で 1 試行を走らせ、終わるまで物理を進める。

        試行の始めにシミュレーション時刻を 1 制御周期進め、前の試行の最後の状態と時刻が重ならないようにする。

        Args:
            trial_index: 試行の番号（結果の env_index に入れる）。
            layout: 箱と目標の配置（1 行）。
            reset_dof_pos: リセット時の関節角 (1, 12) [rad]。

        Raises:
            NoJointTargetsAppliedError: 試行の間に歩行方策の関節目標を一度も適用できなかった。
        """
        ports = self._ports
        self._now_ns += self._timing.control_dt_ns
        start_ns = self._now_ns
        ports.world.reset(layout, reset_dof_pos)
        self._scheduler.start_episode(start_ns)
        ports.peers.reset_gait_policy()
        start_state = ports.world.read()
        ports.publisher.publish(start_ns, start_state)
        goal = layout.goal_xy_m[0]
        ports.peers.start_task(start_ns, (float(goal[0]), float(goal[1])), self._task.success.goal_radius_m)
        wall_start = ports.wall_clock()
        outcome = self._run_until_terminated(layout.goal_xy_m)
        ports.peers.end_task()
        if outcome.selections[TargetSelection.POLICY] == 0:
            raise NoJointTargetsAppliedError(f"試行 {trial_index} で歩行方策の関節目標を一度も適用できなかった")
        result = PushEpisodeResult(
            env_index=trial_index,
            robot_start=_pose(start_state.robot.to_planar()),
            box_start=_pose(start_state.box.to_planar()),
            goal_x_m=float(goal[0]),
            goal_y_m=float(goal[1]),
            box_end=_pose(outcome.final_state.box.to_planar()),
            termination_reason=outcome.reason,
            high_level_steps=outcome.high_level_steps,
            wall_time_s=ports.wall_clock() - wall_start,
            # 1 台の箱押しのシーンには壁も他のロボットも無いため、箱以外との衝突は起こらない
            collisions=0,
            falls=int(outcome.reason == PushTerminationReason.FALL),
        )
        return PushTrialRecord(
            result=result,
            target_selections=outcome.selections,
            target_delay_steps=outcome.delays,
            sim_time_s=ns_to_seconds(self._now_ns - start_ns),
        )

    def _run_until_terminated(self, goal_xy_m: torch.Tensor) -> "_TrialOutcome":
        ports = self._ports
        timing = self._timing
        selections: Counter[TargetSelection] = Counter()
        delays: Counter[int] = Counter()
        hold_steps = torch.zeros(1, dtype=torch.int64)
        high_level_steps = 0
        while True:
            diverged = False
            for _ in range(timing.gait_steps_per_high_level_step):
                ports.pacer.wait_for_next_step()
                selected = self._scheduler.select(self._now_ns)
                selections[selected.selection] += 1
                if selected.delay_steps is not None:
                    delays[selected.delay_steps] += 1
                ports.world.apply_joint_targets(selected.joint_targets)
                ports.world.step()
                diverged = diverged or ports.world.diverged()
                self._now_ns += timing.control_dt_ns
                state = ports.world.read()
                ports.publisher.publish(self._now_ns, state)
            high_level_steps += 1
            judgement = judge_push_step(self._judge_inputs(state, goal_xy_m, high_level_steps, hold_steps, diverged),
                                        self._task)  # fmt: skip
            hold_steps = judgement.hold_steps
            reason = PushTerminationReason(int(judgement.reasons[0]))
            if reason != PushTerminationReason.NONE:
                return _TrialOutcome(reason, high_level_steps, state, selections, delays)

    def _judge_inputs(
        self, state: PushWorldState, goal_xy_m: torch.Tensor, high_level_steps: int, hold_steps: torch.Tensor,
        diverged: bool,
    ) -> PushJudgeInputs:  # fmt: skip
        robot = state.robot.to_planar()
        box = state.box.to_planar()
        diverged_mask = torch.tensor([diverged]) | detect_divergence(
            (robot.position_xy_m, box.position_xy_m, state.go2.base_pos_m),
            (robot.lin_vel_xy_mps, box.lin_vel_xy_mps),
            self._task.divergence,
        )
        return PushJudgeInputs(
            robot_xy_m=robot.position_xy_m,
            box_xy_m=box.position_xy_m,
            goal_xy_m=goal_xy_m,
            base_euler_deg=state.go2.base_euler_deg,
            episode_steps=torch.tensor([high_level_steps]),
            hold_steps=hold_steps,
            diverged=diverged_mask,
        )


@dataclass(frozen=True)
class _TrialOutcome:
    reason: PushTerminationReason
    high_level_steps: int
    final_state: PushWorldState
    selections: Counter[TargetSelection]
    delays: Counter[int]


def _pose(body: PlanarBodyState) -> PlanarPose:
    position = body.position_xy_m[0]
    return PlanarPose(x_m=float(position[0]), y_m=float(position[1]), yaw_rad=float(body.yaw_rad[0]))
