"""2 台の規則ベース協調で、各ロボットが持つ作業の状態機械。

段階は INIT → APPROACH → ALIGN → READY → PUSH → VERIFY → DONE / FAILED。各ロボットが自分の状態機械を持ち、
相手とは作業の段階と準備状態だけを通信で共有する。

遷移に使う条件（押す位置に着いたか、向きが合ったか、など）は、位置や角度から判定した結果を真偽値で受け取る。
幾何の判定は協調の制御の側に置き、この状態機械は「どの条件でどの段階へ進むか」と時間切れだけを決める。
「チームの準備ができた」の判定も呼び出し元に任せる。2 台の協調では相手の状態が新しく READY か PUSH にあること、
1 台や通信なしの独立制御では自分の準備だけで真にすることで、同じ状態機械で条件をそろえて比べられる。

時刻はシミュレーション時刻 [s] を受け取る。状態は不変のデータで持ち、次の状態を返す純粋な関数で進める。
"""

from dataclasses import dataclass
from enum import IntEnum

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError


class InvalidCoordinationConfigError(InvalidConfigError):
    """協調の状態機械の設定値が不変条件（正の時間など）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidCoordinationConfigError)


class CoordinationTimeWentBackError(ValueError):
    """状態機械に、現在の段階に入った時刻より前の時刻が渡されたことを表す（時刻の渡し間違い）。"""


class TeamPhase(IntEnum):
    """作業の段階。値は ROS 2 のメッセージ AgentStatus の PHASE_* と同じにする。

    - INIT: 作業の依頼を待っている。
    - APPROACH: 割り当てられた押す位置へ向かっている。
    - ALIGN: 押す位置で、押す方向へ向きを合わせている。
    - READY: 押す準備ができ、チームの準備がそろうのを待っている。
    - PUSH: 押している。
    - VERIFY: 箱が退避領域に入り、その状態が続くかを確かめている。
    - DONE: 作業を終えた（成功）。
    - FAILED: 作業を終えた（失敗）。
    """

    INIT = 0
    APPROACH = 1
    ALIGN = 2
    READY = 3
    PUSH = 4
    VERIFY = 5
    DONE = 6
    FAILED = 7


TERMINAL_PHASES = frozenset({TeamPhase.DONE, TeamPhase.FAILED})
# 作業の依頼を受けてから、箱を退避領域へ移すまでの段階
_WORKING_PHASES = frozenset({TeamPhase.APPROACH, TeamPhase.ALIGN, TeamPhase.READY, TeamPhase.PUSH})


class FailureReason(IntEnum):
    """FAILED になった理由。"""

    FELL = 1
    TASK_TIMEOUT = 2
    APPROACH_TIMEOUT = 3
    ALIGN_TIMEOUT = 4
    READY_TIMEOUT = 5
    PUSH_TIMEOUT = 6


@dataclass(frozen=True)
class CoordinationTimeoutsConfig:
    """時間切れで FAILED にするまでの時間 [s]（シミュレーション時間）。どれも正の値。

    段階ごとの時間は、その段階に入り直すたびに測り直す。READY と ALIGN を行き来し続ける場合などに備えて、
    作業全体の時間も別に測る。

    Attributes:
        task_s: 作業の依頼を受けて（APPROACH に入って）から、DONE になるまでの上限。
        approach_s: APPROACH に続けていられる上限。
        align_s: ALIGN に続けていられる上限。
        ready_s: READY に続けていられる上限。相手の準備がいつまでもそろわない場合に止めるため。
        push_s: PUSH に続けていられる上限。
    """

    task_s: float
    approach_s: float
    align_s: float
    ready_s: float
    push_s: float

    def __post_init__(self) -> None:
        for name in ("task_s", "approach_s", "align_s", "ready_s", "push_s"):
            _check.positive(f"timeouts.{name}", getattr(self, name))


@dataclass(frozen=True)
class CoordinationConfig:
    """協調の状態機械の設定。

    Attributes:
        timeouts: 時間切れまでの時間。
        verify_hold_s: VERIFY で箱が退避領域に入ったまま続いたら DONE にする時間 [s]。正の値。
            押し終えた直後に箱が滑って退避領域から出ていく場合を成功に数えないため。
    """

    timeouts: CoordinationTimeoutsConfig
    verify_hold_s: float

    def __post_init__(self) -> None:
        _check.positive("verify_hold_s", self.verify_hold_s)


@dataclass(frozen=True)
class CoordinationObservation:
    """1 回の判断で使う条件。協調の制御が位置・角度・通信から判定して渡す。

    Attributes:
        task_assigned: 作業（箱を退避領域へ移す）の依頼を受けている。
        at_push_position: 割り当てられた押す位置の近くにいる。
        aligned: 押す位置にいて、押す方向へ向きが合っている。
        team_ready: チームの準備がそろっている。2 台の協調では、相手の状態が途絶えておらず、相手が READY か PUSH にあること。
            1 台や通信なしの独立制御では常に真にする（自分の準備だけで押し始める）。
        box_in_retreat_area: 箱の中心が退避領域に入っている（入っていれば通路は開いている）。
        fallen: このロボットが転倒した。
    """

    task_assigned: bool
    at_push_position: bool
    aligned: bool
    team_ready: bool
    box_in_retreat_area: bool
    fallen: bool


@dataclass(frozen=True)
class CoordinationState:
    """状態機械の状態。

    Attributes:
        phase: 現在の段階。
        phase_started_s: 現在の段階に入った時刻 [s]。
        task_started_s: 作業の依頼を受けて APPROACH に入った時刻 [s]。INIT のうちは None。
        failure_reason: FAILED になった理由。FAILED 以外では None。
    """

    phase: TeamPhase
    phase_started_s: float
    task_started_s: float | None
    failure_reason: FailureReason | None

    @staticmethod
    def initial(now_s: float) -> "CoordinationState":
        """作業の依頼を待つ（INIT の）状態を返す。"""
        return CoordinationState(phase=TeamPhase.INIT, phase_started_s=now_s, task_started_s=None, failure_reason=None)


def advance_coordination(
    state: CoordinationState, observation: CoordinationObservation, now_s: float, config: CoordinationConfig
) -> CoordinationState:
    """条件と時刻から、次の状態を返す。1 回の呼び出しで進む段階は 1 つまで。

    次の順に判定し、最初に当てはまったものを採る。

    1. DONE・FAILED のままなら変えない。
    2. 転倒したら FAILED（FELL）。
    3. 作業全体の時間切れ、または段階ごとの時間切れなら FAILED（どの時間切れかを理由に残す）。
    4. 箱が退避領域に入っていれば、作業中のどの段階（APPROACH・ALIGN・READY・PUSH）からでも VERIFY。
       相手が先に VERIFY に入ってチームの準備が崩れたときや、自分が押していないうちに箱が入ったときにも、
       待ち続けて時間切れにせず、作業の結果を確かめる段階へ進むため。
    5. 段階ごとの遷移:
       - INIT: 依頼を受けたら APPROACH。
       - APPROACH: 押す位置に着いたら ALIGN。
       - ALIGN: 向きが合ったら READY。押す位置から外れたら APPROACH。
       - READY: 向きが外れたら ALIGN。チームの準備がそろったら PUSH（全員が同じ条件で一斉に押し始める）。
       - PUSH: チームの準備が崩れたら（相手の状態が途絶えた、相手が押すのをやめた）
         押すのをやめて READY で待つ。向きが外れたら ALIGN。
       - VERIFY: 箱が退避領域に入ったまま verify_hold_s 続いたら DONE。退避領域から出たら APPROACH からやり直す。

    Args:
        state: 現在の状態。
        observation: この時刻の条件。
        now_s: 現在のシミュレーション時刻 [s]。state を作った時刻以降であること。
        config: 時間切れと VERIFY の時間。

    Raises:
        CoordinationTimeWentBackError: now_s が現在の段階に入った時刻より前。
    """
    if now_s < state.phase_started_s:
        raise CoordinationTimeWentBackError(f"時刻が戻っている: 現在の段階に入った時刻 {state.phase_started_s} s より前の {now_s} s")
    if state.phase in TERMINAL_PHASES:
        return state
    if observation.fallen:
        return _fail(state, FailureReason.FELL, now_s)
    timeout_reason = _timeout_reason(state, now_s, config.timeouts)
    if timeout_reason is not None:
        return _fail(state, timeout_reason, now_s)
    next_phase = _next_phase(state, observation, now_s, config)
    if next_phase is state.phase:
        return state
    task_started_s = now_s if next_phase is TeamPhase.APPROACH and state.task_started_s is None else state.task_started_s
    return CoordinationState(
        phase=next_phase, phase_started_s=now_s, task_started_s=task_started_s, failure_reason=None
    )


def _timeout_reason(
    state: CoordinationState, now_s: float, timeouts: CoordinationTimeoutsConfig
) -> FailureReason | None:
    if state.task_started_s is not None and now_s - state.task_started_s > timeouts.task_s:
        return FailureReason.TASK_TIMEOUT
    # INIT は依頼を待つだけ、VERIFY は verify_hold_s で必ず抜けるので、段階ごとの時間切れを持たない
    phase_limits = {
        TeamPhase.APPROACH: (timeouts.approach_s, FailureReason.APPROACH_TIMEOUT),
        TeamPhase.ALIGN: (timeouts.align_s, FailureReason.ALIGN_TIMEOUT),
        TeamPhase.READY: (timeouts.ready_s, FailureReason.READY_TIMEOUT),
        TeamPhase.PUSH: (timeouts.push_s, FailureReason.PUSH_TIMEOUT),
    }
    if state.phase not in phase_limits:
        return None
    limit_s, reason = phase_limits[state.phase]
    return reason if now_s - state.phase_started_s > limit_s else None


def _next_phase(
    state: CoordinationState, observation: CoordinationObservation, now_s: float, config: CoordinationConfig
) -> TeamPhase:
    phase = state.phase
    if phase in _WORKING_PHASES and observation.box_in_retreat_area:
        return TeamPhase.VERIFY
    if phase is TeamPhase.INIT:
        return TeamPhase.APPROACH if observation.task_assigned else phase
    if phase is TeamPhase.APPROACH:
        return TeamPhase.ALIGN if observation.at_push_position else phase
    if phase is TeamPhase.ALIGN:
        if not observation.at_push_position:
            return TeamPhase.APPROACH
        return TeamPhase.READY if observation.aligned else phase
    if phase is TeamPhase.READY:
        if not observation.aligned:
            return TeamPhase.ALIGN
        return TeamPhase.PUSH if observation.team_ready else phase
    if phase is TeamPhase.PUSH:
        if not observation.team_ready:
            return TeamPhase.READY
        return TeamPhase.ALIGN if not observation.aligned else phase
    if phase is TeamPhase.VERIFY:
        if not observation.box_in_retreat_area:
            return TeamPhase.APPROACH
        return TeamPhase.DONE if now_s - state.phase_started_s >= config.verify_hold_s else phase
    raise ValueError(f"遷移を定めていない段階: {phase!r}")


def _fail(state: CoordinationState, reason: FailureReason, now_s: float) -> CoordinationState:
    return CoordinationState(
        phase=TeamPhase.FAILED, phase_started_s=now_s, task_started_s=state.task_started_s, failure_reason=reason
    )

