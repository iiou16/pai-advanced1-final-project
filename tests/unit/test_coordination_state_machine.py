import dataclasses
from collections.abc import Callable

import pytest

from cooperative_go2.entities.coordination_state_machine import (
    CoordinationConfig,
    CoordinationObservation,
    CoordinationState,
    CoordinationTimeoutsConfig,
    CoordinationTimeWentBackError,
    FailureReason,
    InvalidCoordinationConfigError,
    TeamPhase,
    advance_coordination,
)
from cooperative_go2.entities.team_readiness import (
    InvalidTeamReadinessConfigError,
    PartnerStatus,
    TeamReadinessConfig,
    TeamReadinessInputs,
    is_team_ready_with_partner,
)

TIMEOUTS = CoordinationTimeoutsConfig(task_s=60.0, approach_s=20.0, align_s=10.0, ready_s=15.0, push_s=30.0)
CONFIG = CoordinationConfig(timeouts=TIMEOUTS, verify_hold_s=1.0)
READINESS = TeamReadinessConfig(partner_stale_after_s=0.5, push_start_grace_s=0.5)
# どの条件も満たさない観測。テストでは必要な条件だけを真にする
NOTHING = CoordinationObservation(
    task_assigned=False,
    at_push_position=False,
    aligned=False,
    team_ready=False,
    box_in_retreat_area=False,
    fallen=False,
)


def in_phase(phase: TeamPhase, phase_started_s: float = 0.0, task_started_s: float | None = 0.0) -> CoordinationState:
    return CoordinationState(
        phase=phase, phase_started_s=phase_started_s, task_started_s=task_started_s, failure_reason=None
    )


def observe(**conditions: bool) -> CoordinationObservation:
    return dataclasses.replace(NOTHING, **conditions)


@pytest.mark.parametrize(
    ("phase", "conditions", "expected"),
    [
        (TeamPhase.INIT, {"task_assigned": True}, TeamPhase.APPROACH),
        (TeamPhase.INIT, {}, TeamPhase.INIT),
        (TeamPhase.APPROACH, {"at_push_position": True}, TeamPhase.ALIGN),
        (TeamPhase.APPROACH, {}, TeamPhase.APPROACH),
        (TeamPhase.ALIGN, {"at_push_position": True, "aligned": True}, TeamPhase.READY),
        (TeamPhase.ALIGN, {"at_push_position": True}, TeamPhase.ALIGN),
        (TeamPhase.ALIGN, {"aligned": True}, TeamPhase.APPROACH),
        (TeamPhase.READY, {"aligned": True, "team_ready": True}, TeamPhase.PUSH),
        (TeamPhase.READY, {"aligned": True}, TeamPhase.READY),
        (TeamPhase.READY, {"team_ready": True}, TeamPhase.ALIGN),
        # 箱が退避領域に入ったら、チームの準備や向きに関係なく確かめる段階へ進む
        (TeamPhase.PUSH, {"box_in_retreat_area": True}, TeamPhase.VERIFY),
        # 相手の状態が途絶えた・相手が押すのをやめたら、押すのをやめて待つ
        (TeamPhase.PUSH, {"aligned": True}, TeamPhase.READY),
        (TeamPhase.PUSH, {"team_ready": True}, TeamPhase.ALIGN),
        (TeamPhase.PUSH, {"aligned": True, "team_ready": True}, TeamPhase.PUSH),
        (TeamPhase.VERIFY, {}, TeamPhase.APPROACH),
    ],
    ids=[
        "init-task-assigned",
        "init-waiting",
        "approach-arrived",
        "approach-moving",
        "align-aligned",
        "align-turning",
        "align-left-push-position",
        "ready-team-ready",
        "ready-waiting-for-team",
        "ready-lost-alignment",
        "push-box-in-retreat-area",
        "push-team-not-ready",
        "push-lost-alignment",
        "push-continuing",
        "verify-box-left-retreat-area",
    ],
)
def test_phase_transitions(phase: TeamPhase, conditions: dict[str, bool], expected: TeamPhase) -> None:
    state = in_phase(phase, task_started_s=None if phase is TeamPhase.INIT else 0.0)

    advanced = advance_coordination(state, observe(**conditions), now_s=0.5, config=CONFIG)

    assert advanced.phase is expected


def test_verify_waits_until_hold_time_has_passed() -> None:
    state = in_phase(TeamPhase.VERIFY, phase_started_s=10.0)

    still_verifying = advance_coordination(state, observe(box_in_retreat_area=True), now_s=10.9, config=CONFIG)
    done = advance_coordination(state, observe(box_in_retreat_area=True), now_s=11.0, config=CONFIG)

    assert still_verifying.phase is TeamPhase.VERIFY
    assert done.phase is TeamPhase.DONE


def test_transition_records_new_phase_start_time() -> None:
    state = in_phase(TeamPhase.APPROACH, phase_started_s=1.0)

    advanced = advance_coordination(state, observe(at_push_position=True), now_s=3.5, config=CONFIG)

    assert advanced.phase_started_s == 3.5


def test_staying_in_phase_keeps_its_start_time() -> None:
    state = in_phase(TeamPhase.APPROACH, phase_started_s=1.0)

    advanced = advance_coordination(state, NOTHING, now_s=3.5, config=CONFIG)

    assert advanced == state


def test_task_start_time_is_set_when_task_is_assigned() -> None:
    state = CoordinationState.initial(now_s=0.0)

    advanced = advance_coordination(state, observe(task_assigned=True), now_s=2.0, config=CONFIG)

    assert advanced.task_started_s == 2.0


def test_task_start_time_is_kept_when_restarting_approach() -> None:
    state = in_phase(TeamPhase.VERIFY, phase_started_s=30.0, task_started_s=2.0)

    advanced = advance_coordination(state, NOTHING, now_s=30.5, config=CONFIG)

    assert advanced.phase is TeamPhase.APPROACH
    assert advanced.task_started_s == 2.0


@pytest.mark.parametrize(
    ("phase", "limit_s", "reason"),
    [
        (TeamPhase.APPROACH, TIMEOUTS.approach_s, FailureReason.APPROACH_TIMEOUT),
        (TeamPhase.ALIGN, TIMEOUTS.align_s, FailureReason.ALIGN_TIMEOUT),
        (TeamPhase.READY, TIMEOUTS.ready_s, FailureReason.READY_TIMEOUT),
        (TeamPhase.PUSH, TIMEOUTS.push_s, FailureReason.PUSH_TIMEOUT),
    ],
    ids=["approach", "align", "ready", "push"],
)
def test_phase_timeout_fails_after_limit(phase: TeamPhase, limit_s: float, reason: FailureReason) -> None:
    # 作業全体の時間切れに掛からないよう、作業は段階と同時に始めたことにする
    state = in_phase(phase, phase_started_s=5.0, task_started_s=5.0)

    at_limit = advance_coordination(state, NOTHING, now_s=5.0 + limit_s, config=CONFIG)
    beyond_limit = advance_coordination(state, NOTHING, now_s=5.0 + limit_s + 0.01, config=CONFIG)

    assert at_limit.phase is not TeamPhase.FAILED
    assert beyond_limit.phase is TeamPhase.FAILED
    assert beyond_limit.failure_reason is reason


@pytest.mark.parametrize("phase", [TeamPhase.INIT, TeamPhase.VERIFY], ids=["init", "verify"])
def test_phases_without_their_own_timeout_do_not_fail_by_phase_time(phase: TeamPhase) -> None:
    # INIT は依頼を待つだけ。VERIFY は箱が入ったままなら DONE、出たら APPROACH に移るので、段階の時間では止めない
    task_started_s = None if phase is TeamPhase.INIT else 100.0
    state = in_phase(phase, phase_started_s=100.0, task_started_s=task_started_s)

    advanced = advance_coordination(state, NOTHING, now_s=150.0, config=CONFIG)

    assert advanced.phase is not TeamPhase.FAILED


def test_task_timeout_fails_even_when_phase_time_is_short() -> None:
    # READY と ALIGN を行き来して段階の時間が測り直され続けても、作業全体の時間で止める
    state = in_phase(TeamPhase.READY, phase_started_s=60.0, task_started_s=0.0)

    advanced = advance_coordination(state, observe(aligned=True), now_s=60.5, config=CONFIG)

    assert advanced.phase is TeamPhase.FAILED
    assert advanced.failure_reason is FailureReason.TASK_TIMEOUT


def test_timeout_takes_precedence_over_transition() -> None:
    state = in_phase(TeamPhase.APPROACH, phase_started_s=0.0, task_started_s=0.0)

    advanced = advance_coordination(state, observe(at_push_position=True), now_s=20.5, config=CONFIG)

    assert advanced.failure_reason is FailureReason.APPROACH_TIMEOUT


def test_fall_fails_from_any_active_phase() -> None:
    state = in_phase(TeamPhase.PUSH, phase_started_s=1.0)

    advanced = advance_coordination(
        state, observe(fallen=True, aligned=True, team_ready=True, box_in_retreat_area=True), now_s=2.0, config=CONFIG
    )

    assert advanced.phase is TeamPhase.FAILED
    assert advanced.failure_reason is FailureReason.FELL


@pytest.mark.parametrize("phase", [TeamPhase.DONE, TeamPhase.FAILED], ids=["done", "failed"])
def test_terminal_phases_do_not_change(phase: TeamPhase) -> None:
    state = in_phase(phase, phase_started_s=1.0)

    advanced = advance_coordination(state, observe(fallen=True, task_assigned=True), now_s=500.0, config=CONFIG)

    assert advanced == state


def test_time_going_back_is_rejected() -> None:
    state = in_phase(TeamPhase.PUSH, phase_started_s=10.0)

    with pytest.raises(CoordinationTimeWentBackError):
        advance_coordination(state, NOTHING, now_s=9.9, config=CONFIG)


def test_phase_values_match_agent_status_message() -> None:
    # ROS 2 の AgentStatus.msg の PHASE_INIT=0 〜 PHASE_FAILED=7 と同じ値で送受信する
    assert [phase.value for phase in TeamPhase] == list(range(8))
    assert [phase.name for phase in TeamPhase] == [
        "INIT", "APPROACH", "ALIGN", "READY", "PUSH", "VERIFY", "DONE", "FAILED",
    ]


@pytest.mark.parametrize("field", ["task_s", "approach_s", "align_s", "ready_s", "push_s"])
def test_timeouts_must_be_positive(field: str) -> None:
    with pytest.raises(InvalidCoordinationConfigError):
        dataclasses.replace(TIMEOUTS, **{field: 0.0})


def test_verify_hold_must_be_positive() -> None:
    with pytest.raises(InvalidCoordinationConfigError):
        CoordinationConfig(timeouts=TIMEOUTS, verify_hold_s=0.0)


@pytest.mark.parametrize("field", ["partner_stale_after_s", "push_start_grace_s"])
def test_team_readiness_times_must_be_positive(field: str) -> None:
    with pytest.raises(InvalidTeamReadinessConfigError):
        dataclasses.replace(READINESS, **{field: 0.0})


@pytest.mark.parametrize(
    "phase", [TeamPhase.APPROACH, TeamPhase.ALIGN, TeamPhase.READY, TeamPhase.PUSH], ids=lambda phase: phase.name
)
def test_box_in_retreat_area_leads_to_verify_from_any_working_phase(phase: TeamPhase) -> None:
    advanced = advance_coordination(in_phase(phase), observe(box_in_retreat_area=True), now_s=0.5, config=CONFIG)

    assert advanced.phase is TeamPhase.VERIFY


def test_box_in_retreat_area_before_task_is_assigned_keeps_waiting() -> None:
    state = CoordinationState.initial(now_s=0.0)

    advanced = advance_coordination(state, observe(box_in_retreat_area=True), now_s=0.5, config=CONFIG)

    assert advanced.phase is TeamPhase.INIT


def readiness(own: CoordinationState, partner: PartnerStatus | None, now_s: float) -> bool:
    return is_team_ready_with_partner(TeamReadinessInputs(own=own, partner=partner, now_s=now_s), READINESS)


@pytest.mark.parametrize(
    ("partner_phase", "expected"),
    [
        (TeamPhase.READY, True),
        (TeamPhase.PUSH, True),
        (TeamPhase.ALIGN, False),
        (TeamPhase.VERIFY, False),
        (TeamPhase.FAILED, False),
    ],
    ids=lambda value: value.name if isinstance(value, TeamPhase) else str(value),
)
def test_team_is_ready_to_start_when_partner_is_ready_or_pushing(partner_phase: TeamPhase, expected: bool) -> None:
    own = in_phase(TeamPhase.READY, phase_started_s=1.0)

    assert readiness(own, PartnerStatus(phase=partner_phase, stamp_s=2.0), now_s=2.1) is expected


def test_team_is_not_ready_before_partner_status_arrives() -> None:
    assert readiness(in_phase(TeamPhase.READY), None, now_s=1.0) is False


def test_team_is_not_ready_when_partner_status_is_stale() -> None:
    own = in_phase(TeamPhase.READY, phase_started_s=0.0)
    partner = PartnerStatus(phase=TeamPhase.READY, stamp_s=1.0)

    # 0.5 s ちょうどの古さまでは有効で、それを超えたら途絶えたとみなす
    assert readiness(own, partner, now_s=1.5) is True
    assert readiness(own, partner, now_s=1.51) is False


def test_pushing_requires_partner_to_push_after_start_grace() -> None:
    own = in_phase(TeamPhase.PUSH, phase_started_s=10.0)

    assert readiness(own, PartnerStatus(phase=TeamPhase.PUSH, stamp_s=12.0), now_s=12.0) is True
    # 押し始めてから 0.5 s までは、相手が同じ合図で PUSH に入る途中として READY でも待つ
    assert readiness(own, PartnerStatus(phase=TeamPhase.READY, stamp_s=10.5), now_s=10.5) is True
    # それを過ぎて相手が READY に戻っていたら、1 台で押し続けない
    assert readiness(own, PartnerStatus(phase=TeamPhase.READY, stamp_s=10.6), now_s=10.6) is False


# 2 台の状態機械をつないだテスト。各ロボットは 0.1 s ごとに判断し、相手の状態は 1 回前の判断の結果を受け取る
# （通信の遅れ 1 回分）。位置・向き・箱の条件は時刻の関数として台本で与える
TICK_S = 0.1
ConditionScript = Callable[[int, float], dict[str, bool]]
# 相手 index の状態を、ロボット index が最後に受け取れる時刻（通信が切れた時刻）。None なら切れない
LinkCutScript = Callable[[int], float | None]


def run_pair(
    conditions: ConditionScript, duration_s: float, link_cut_s: LinkCutScript = lambda _robot: None
) -> list[list[CoordinationState]]:
    states = [CoordinationState.initial(now_s=0.0), CoordinationState.initial(now_s=0.0)]
    published: list[PartnerStatus | None] = [None, None]
    received: list[PartnerStatus | None] = [None, None]
    history: list[list[CoordinationState]] = [[], []]
    for tick in range(1, round(duration_s / TICK_S) + 1):
        now_s = round(tick * TICK_S, 6)
        for robot in (0, 1):
            cut_s = link_cut_s(robot)
            if cut_s is None or now_s <= cut_s:
                received[robot] = published[1 - robot]
        next_states = []
        for robot in (0, 1):
            team_ready = readiness(states[robot], received[robot], now_s)
            observation = observe(team_ready=team_ready, **conditions(robot, now_s))
            next_states.append(advance_coordination(states[robot], observation, now_s, CONFIG))
        states = next_states
        published = [PartnerStatus(phase=state.phase, stamp_s=now_s) for state in states]
        for robot in (0, 1):
            history[robot].append(states[robot])
    return history


def first_time_in(history: list[CoordinationState], phase: TeamPhase) -> float:
    for tick, state in enumerate(history, start=1):
        if state.phase is phase:
            return round(tick * TICK_S, 6)
    raise AssertionError(f"{phase.name} に入らなかった: 最後は {history[-1].phase.name}")


def first_time_after(history: list[CoordinationState], start_s: float, predicate: Callable[[TeamPhase], bool]) -> float:
    for tick, state in enumerate(history, start=1):
        now_s = round(tick * TICK_S, 6)
        if now_s > start_s and predicate(state.phase):
            return now_s
    raise AssertionError(f"{start_s} s より後に条件を満たさなかった: 最後は {history[-1].phase.name}")


def staged_conditions(aligned_from_s: tuple[float, float], box_in_from_s: tuple[float, float] | None) -> ConditionScript:
    """依頼は最初から受け、0.5 s で押す位置に着き、ロボットごとの時刻に向きが合い、箱が退避領域に入る台本。"""

    def conditions(robot: int, now_s: float) -> dict[str, bool]:
        return {
            "task_assigned": True,
            "at_push_position": now_s >= 0.5,
            "aligned": now_s >= aligned_from_s[robot],
            "box_in_retreat_area": box_in_from_s is not None and now_s >= box_in_from_s[robot],
        }

    return conditions


def test_pair_starts_pushing_together_after_both_are_aligned() -> None:
    history = run_pair(staged_conditions(aligned_from_s=(1.0, 2.0), box_in_from_s=None), duration_s=4.0)

    push_start_a = first_time_in(history[0], TeamPhase.PUSH)
    push_start_b = first_time_in(history[1], TeamPhase.PUSH)
    # 先に向きが合った A は、B の準備がそろう（2.0 s）まで押し始めず、両者は 1 回の判断の差で押し始める
    assert push_start_a > 2.0
    assert abs(push_start_a - push_start_b) <= TICK_S + 1e-9


def test_pair_stops_pushing_when_partner_status_stops_arriving() -> None:
    cut_s = 5.0
    history = run_pair(
        staged_conditions(aligned_from_s=(1.0, 1.0), box_in_from_s=None),
        duration_s=8.0,
        link_cut_s=lambda robot: cut_s if robot == 1 else None,
    )

    def stopped_pushing(phase: TeamPhase) -> bool:
        return phase is not TeamPhase.PUSH

    assert history[1][round(cut_s / TICK_S) - 1].phase is TeamPhase.PUSH
    # B は A の状態が途絶えたと判断して止まり、A は B が READY に戻ったのを見て 1 台で押し続けない
    b_stop_s = first_time_after(history[1], cut_s, stopped_pushing)
    a_stop_s = first_time_after(history[0], cut_s, stopped_pushing)
    assert b_stop_s <= cut_s + READINESS.partner_stale_after_s + TICK_S + 1e-9
    assert a_stop_s <= b_stop_s + 2 * TICK_S + 1e-9


def test_pair_both_finish_when_box_reaches_retreat_area_at_different_times() -> None:
    # A が先に箱を退避領域に入ったと判断し、B は 0.3 s 遅れて判断する（判定の境界でのずれ）
    history = run_pair(staged_conditions(aligned_from_s=(1.0, 1.0), box_in_from_s=(6.0, 6.3)), duration_s=10.0)

    assert history[0][-1].phase is TeamPhase.DONE
    assert history[1][-1].phase is TeamPhase.DONE


def test_pair_stops_pushing_when_partner_falls() -> None:
    fall_s = 5.0
    aligned = staged_conditions(aligned_from_s=(1.0, 1.0), box_in_from_s=None)

    def conditions(robot: int, now_s: float) -> dict[str, bool]:
        return {**aligned(robot, now_s), "fallen": robot == 0 and now_s >= fall_s}

    history = run_pair(conditions, duration_s=8.0)

    assert history[0][-1].failure_reason is FailureReason.FELL
    b_stop_s = first_time_after(history[1], fall_s, lambda phase: phase is not TeamPhase.PUSH)
    assert b_stop_s <= fall_s + 2 * TICK_S + 1e-9
