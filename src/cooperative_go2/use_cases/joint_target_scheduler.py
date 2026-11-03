"""シミュレーションノードが、受け取った関節目標のうち次の物理ステップに適用するものを選ぶ。

歩行方策は、学習時に「時刻 t の状態から作った目標を、t から 1 制御周期後の物理ステップに適用する」遅れで学習している
（env.simulate_action_latency、D19）。ROS 2 の経路でも同じ遅れになるよう、物理を進める直前の時刻 now から
latency_steps 制御周期より前の状態で作った目標のうち、最も新しいものを適用する。通信がそれより遅れて目標が届いて
いなければ、それより古い目標を使い、有効期限より古い目標しか無ければ既定姿勢を保つ（要件 5.5）。
"""

import threading
from dataclasses import dataclass
from enum import Enum

import torch

from cooperative_go2.entities.command_expiry import is_command_fresh


class TargetSelection(Enum):
    """適用した目標の出所。"""

    # 歩行方策の目標を適用した
    POLICY = "policy"
    # エピソードの始めで、まだ適用できる目標が無いので既定姿勢を保った（学習時と同じ）
    EPISODE_START = "episode_start"
    # 目標が届かないか有効期限より古いので既定姿勢を保った（停止）
    STALE = "stale"


@dataclass(frozen=True)
class SelectedJointTargets:
    """次の物理ステップに適用する目標。

    Attributes:
        joint_targets: 目標関節角 (1, 12) [rad]、GO2_JOINT_NAMES の順。
        selection: 目標の出所。
        delay_steps: 目標の元になった状態から、適用する物理ステップまでの制御周期の数。POLICY のときだけ値を持つ。
    """

    joint_targets: torch.Tensor
    selection: TargetSelection
    delay_steps: int | None


@dataclass(frozen=True)
class JointTargetTiming:
    """目標を選ぶ時間の条件。時刻はすべてシミュレーション時刻 [ns]。

    Attributes:
        control_dt_ns: 歩行の制御周期。
        latency_steps: 学習時の行動の遅れ（制御周期の数）。simulate_action_latency なら 1、無ければ 0。
        expiry_ns: 目標の有効期限。元になった状態の時刻からの経過時間がこれを超えた目標は使わない。
    """

    control_dt_ns: int
    latency_steps: int
    expiry_ns: int


class JointTargetScheduler:
    """受け取った関節目標を元の状態の時刻ごとに持ち、物理ステップごとに適用するものを選ぶ。

    receive() は ROS 2 の受信のスレッドから、select() と start_episode() はシミュレーションのスレッドから呼ぶ。
    """

    def __init__(self, timing: JointTargetTiming, default_joint_targets: torch.Tensor) -> None:
        """時間の条件と、目標が無いときに保つ既定姿勢の関節角 (1, 12) [rad] を受け取る。"""
        self._timing = timing
        self._default = default_joint_targets
        self._lock = threading.Lock()
        self._targets: dict[int, torch.Tensor] = {}
        self._episode_start_ns = 0

    def start_episode(self, start_ns: int) -> None:
        """時刻 start_ns に始めたエピソードの目標だけを受け付けるようにし、前のエピソードの目標を捨てる。"""
        with self._lock:
            self._episode_start_ns = start_ns
            self._targets.clear()

    def receive(self, state_stamp_ns: int, joint_targets: torch.Tensor) -> None:
        """時刻 state_stamp_ns の状態から作られた目標 (1, 12) を受け取る。今のエピソードより前の状態の目標は捨てる。"""
        with self._lock:
            if state_stamp_ns >= self._episode_start_ns:
                self._targets[state_stamp_ns] = joint_targets

    def select(self, now_ns: int) -> SelectedJointTargets:
        """時刻 now_ns から物理を 1 ステップ進めるときに適用する目標を選び、それより古い目標を捨てる。"""
        timing = self._timing
        latest_usable_ns = now_ns - timing.latency_steps * timing.control_dt_ns
        with self._lock:
            usable = [stamp for stamp in self._targets if stamp <= latest_usable_ns]
            if not usable:
                return self._hold_default(now_ns)
            stamp = max(usable)
            joint_targets = self._targets[stamp]
            for old in usable:
                if old < stamp:
                    del self._targets[old]
        if not is_command_fresh(stamp, now_ns, timing.expiry_ns):
            return SelectedJointTargets(self._default, TargetSelection.STALE, delay_steps=None)
        delay_steps = round((now_ns - stamp) / timing.control_dt_ns)
        return SelectedJointTargets(joint_targets, TargetSelection.POLICY, delay_steps=delay_steps)

    def _hold_default(self, now_ns: int) -> SelectedJointTargets:
        # エピソードの始めの latency_steps 周期は、学習時も既定姿勢を適用するので停止とは区別する
        within_start = now_ns - self._episode_start_ns < self._timing.latency_steps * self._timing.control_dt_ns
        selection = TargetSelection.EPISODE_START if within_start else TargetSelection.STALE
        return SelectedJointTargets(self._default, selection, delay_steps=None)
