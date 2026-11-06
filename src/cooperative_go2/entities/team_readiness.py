"""2 台の協調で、通信で受け取った相手の状態から「チームの準備ができた」を判定する。

判定の結果は、協調の状態機械の条件（CoordinationObservation.team_ready）として渡す。
1 台や通信なしの独立制御ではこの判定を使わず、team_ready を常に真にする。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError
from cooperative_go2.entities.coordination_state_machine import CoordinationState, TeamPhase


class InvalidTeamReadinessConfigError(InvalidConfigError):
    """チームの準備の判定の設定値が不変条件（正の時間）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidTeamReadinessConfigError)
# 押し始めの合図として、相手がこの段階にいればチームの準備ができたとみなす
_PARTNER_PHASES_READY_TO_START = frozenset({TeamPhase.READY, TeamPhase.PUSH})
# 押している間に相手がこの段階に戻ったら、相手は押す位置を直している（作業を続けている）とみなす
PARTNER_REPOSITIONING_PHASES = frozenset({TeamPhase.APPROACH, TeamPhase.ALIGN, TeamPhase.READY})


@dataclass(frozen=True)
class TeamReadinessConfig:
    """相手の状態から「チームの準備ができた」を判定するための時間 [s]（シミュレーション時間）。どれも正の値。

    Attributes:
        partner_stale_after_s: 相手の状態がこの時間より古ければ、途絶えたとみなす。
        push_start_grace_s: 自分が PUSH に入ってからこの時間までは、相手がまだ READY でも準備ができたとみなす。
            相手が同じ合図で PUSH に入るまでの通信の遅れを待つため。この時間を過ぎたら、相手も PUSH にいることを求める。
    """

    partner_stale_after_s: float
    push_start_grace_s: float

    def __post_init__(self) -> None:
        _check.positive("team_readiness.partner_stale_after_s", self.partner_stale_after_s)
        _check.positive("team_readiness.push_start_grace_s", self.push_start_grace_s)


@dataclass(frozen=True)
class PartnerStatus:
    """通信で受け取った相手の作業の状態。

    Attributes:
        phase: 相手の段階。
        stamp_s: 相手がこの状態を配信したシミュレーション時刻 [s]。
    """

    phase: TeamPhase
    stamp_s: float


@dataclass(frozen=True)
class TeamReadinessInputs:
    """「チームの準備ができた」を判定する材料。

    Attributes:
        own: 自分の状態機械の状態。
        partner: 最後に受け取った相手の状態。まだ一度も受け取っていなければ None。
        now_s: 現在のシミュレーション時刻 [s]。
        keep_pushing_while_partner_repositions: 押している間に相手が押す位置を直し始めても（相手の状態が届いていて、
            段階が PARTNER_REPOSITIONING_PHASES のどれか）、準備はできているとみなして押し続けるか。
            False なら相手も PUSH にいることを求め、相手が押すのをやめたら止まって待つ。
    """

    own: CoordinationState
    partner: PartnerStatus | None
    now_s: float
    keep_pushing_while_partner_repositions: bool


def is_team_ready_with_partner(inputs: TeamReadinessInputs, config: TeamReadinessConfig) -> bool:
    """2 台の協調で、相手の状態から「チームの準備ができた」（CoordinationObservation.team_ready）を判定する。

    - 相手の状態をまだ受け取っていない、または partner_stale_after_s より古ければ、準備はできていない（途絶えた）。
      押している最中でも止まって待つ。
    - 自分が PUSH にいる間:
      - keep_pushing_while_partner_repositions なら、相手が PUSH にいるか、押す位置を直している
        （APPROACH・ALIGN・READY）間は準備ができているとみなす。相手が押す位置を直している間に 2 台とも止まると、
        箱も 2 台も動かず、相手が戻れない位置関係が変わらないため。相手が失敗した（FAILED）などのときは止まる。
      - そうでなければ、相手も PUSH にいることを求める。ただし PUSH に入ってから push_start_grace_s までは、
        相手が同じ合図で PUSH に入る途中なので READY でもよい。
    - それ以外（押し始める前）は、相手が READY か PUSH にいれば準備ができた（2 台が同じ合図で押し始める）。

    1 台や通信なしの独立制御では、この関数を使わず team_ready を常に真にする。
    """
    partner = inputs.partner
    if partner is None or inputs.now_s - partner.stamp_s > config.partner_stale_after_s:
        return False
    own = inputs.own
    if own.phase is TeamPhase.PUSH:
        if partner.phase is TeamPhase.PUSH:
            return True
        if inputs.keep_pushing_while_partner_repositions:
            return partner.phase in PARTNER_REPOSITIONING_PHASES
        within_grace = inputs.now_s - own.phase_started_s <= config.push_start_grace_s
        return within_grace and partner.phase is TeamPhase.READY
    return partner.phase in _PARTNER_PHASES_READY_TO_START


def is_partner_repositioning(partner: PartnerStatus | None, now_s: float, config: TeamReadinessConfig) -> bool:
    """相手の状態が届いていて（partner_stale_after_s 以内）、相手が押す位置を直している（APPROACH・ALIGN・READY）か。

    相手の状態をまだ受け取っていない、または途絶えたなら False（直しているかどうか分からない）。
    """
    if partner is None or now_s - partner.stamp_s > config.partner_stale_after_s:
        return False
    return partner.phase in PARTNER_REPOSITIONING_PHASES
