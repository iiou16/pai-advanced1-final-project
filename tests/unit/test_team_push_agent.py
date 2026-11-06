import dataclasses
from pathlib import Path

import pytest

from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.push_trial import PlanarPose
from cooperative_go2.entities.team_readiness import PartnerStatus
from cooperative_go2.entities.team_task_config import TeamRuleBasedConfig
from cooperative_go2.interface_adapters.config_loader import load_team_run_settings
from cooperative_go2.use_cases.team_push_agent import (
    AgentNotStartedError,
    CoordinationMode,
    RetreatArea,
    TeamPushAgent,
    TeamPushAgentInputs,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
RETREAT = RetreatArea(center_xy_m=(3.8, 0.0), radius_m=0.4)
BOX = PlanarPose(x_m=1.4, y_m=0.0, yaw_rad=0.0)
# 左（go2_a、横のずれ +0.25）の押す位置 (0.8, 0.25) にいて +x を向いている
LEFT_AT_PUSH_POSITION = PlanarPose(x_m=0.8, y_m=0.25, yaw_rad=0.0)


@pytest.fixture
def team_config(monkeypatch: pytest.MonkeyPatch) -> TeamRuleBasedConfig:
    """リポジトリの 2 台の作業の設定。シーンの設定ファイルを相対パスで読むため、リポジトリのルートで読む。"""
    monkeypatch.chdir(REPO_ROOT)
    return load_team_run_settings(Path("configs/team_rule_based.yaml")).settings.team


def inputs(
    now_s: float, partner: PartnerStatus | None, robot: PlanarPose = LEFT_AT_PUSH_POSITION
) -> TeamPushAgentInputs:
    return TeamPushAgentInputs(now_s=now_s, robot=robot, box=BOX, partner=partner, task_assigned=True, fallen=False)


def run_until_ready(agent: TeamPushAgent, partner: PartnerStatus | None) -> float:
    """依頼を受けてから、押す位置で向きが合って READY になるまで判断を進め、そのときの時刻を返す。"""
    agent.start(RETREAT, now_s=0.0)
    for tick in range(1, 10):
        now_s = tick * 0.1
        agent.decide(inputs(now_s, partner))
        if agent.state.phase is TeamPhase.READY:
            return now_s
    raise AssertionError(f"READY にならなかった: {agent.state.phase.name}")


def test_deciding_before_start_is_rejected(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.COOPERATIVE)

    with pytest.raises(AgentNotStartedError):
        agent.decide(inputs(0.1, None))


def test_cooperative_agent_waits_for_partner_before_pushing(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.COOPERATIVE)
    now_s = run_until_ready(agent, partner=None)

    waiting = agent.decide(inputs(now_s + 0.1, partner=PartnerStatus(phase=TeamPhase.ALIGN, stamp_s=now_s)))
    pushing = agent.decide(inputs(now_s + 0.2, partner=PartnerStatus(phase=TeamPhase.READY, stamp_s=now_s + 0.1)))

    assert waiting.state.phase is TeamPhase.READY
    assert waiting.commands[0].tolist() == [0.0, 0.0, 0.0]
    assert pushing.state.phase is TeamPhase.PUSH
    assert float(pushing.commands[0, 0]) > 0.0


def test_independent_agent_pushes_without_partner(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.INDEPENDENT)
    now_s = run_until_ready(agent, partner=None)

    decision = agent.decide(inputs(now_s + 0.1, partner=None))

    assert decision.state.phase is TeamPhase.PUSH


def test_decision_reports_intent_for_partner(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=1, mode=CoordinationMode.COOPERATIVE)
    agent.start(RETREAT, now_s=0.0)

    decision = agent.decide(inputs(0.1, None, robot=PlanarPose(x_m=0.0, y_m=-0.25, yaw_rad=0.0)))

    # 右（go2_b、横のずれ -0.25）の押す位置と、+x の押す方向を相手に知らせる
    assert decision.push_position_xy_m == pytest.approx((0.8, -0.25))
    assert decision.push_dir_xy == pytest.approx((1.0, 0.0))
    assert decision.box_in_retreat_area is False


def test_box_in_retreat_area_is_reported(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.INDEPENDENT)
    agent.start(RETREAT, now_s=0.0)
    box_in_retreat = dataclasses.replace(BOX, x_m=3.7)

    decision = agent.decide(
        TeamPushAgentInputs(
            now_s=0.1, robot=LEFT_AT_PUSH_POSITION, box=box_in_retreat, partner=None, task_assigned=True, fallen=False
        )
    )

    assert decision.box_in_retreat_area is True


def test_start_resets_state_machine(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.INDEPENDENT)
    run_until_ready(agent, partner=None)

    agent.start(RETREAT, now_s=5.0)

    assert agent.state.phase is TeamPhase.INIT
    assert agent.state.phase_started_s == 5.0


def run_until_pushing(agent: TeamPushAgent) -> float:
    """相手が READY のまま、自分が PUSH に入るまで判断を進め、そのときの時刻を返す。"""
    now_s = run_until_ready(agent, partner=None)
    now_s += 0.1
    agent.decide(inputs(now_s, PartnerStatus(phase=TeamPhase.READY, stamp_s=now_s - 0.1)))
    assert agent.state.phase is TeamPhase.PUSH
    return now_s


def test_cooperative_agent_keeps_pushing_on_center_line_while_partner_repositions(
    team_config: TeamRuleBasedConfig,
) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.COOPERATIVE)
    now_s = run_until_pushing(agent)

    decision = agent.decide(inputs(now_s + 1.0, PartnerStatus(phase=TeamPhase.ALIGN, stamp_s=now_s + 0.9)))

    # 押し続け、押す位置（意図）は箱の中央の線（横のずれ 0）へ寄る
    assert decision.state.phase is TeamPhase.PUSH
    assert decision.push_position_xy_m == pytest.approx((0.8, 0.0))
    assert float(decision.commands[0, 0]) > 0.0


def test_waiting_cooperative_agent_stops_while_partner_repositions(team_config: TeamRuleBasedConfig) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.COOPERATIVE_WAIT)
    now_s = run_until_pushing(agent)

    decision = agent.decide(inputs(now_s + 1.0, PartnerStatus(phase=TeamPhase.ALIGN, stamp_s=now_s + 0.9)))

    assert decision.state.phase is TeamPhase.READY
    assert decision.commands[0].tolist() == [0.0, 0.0, 0.0]


@pytest.mark.parametrize(
    "mode", [CoordinationMode.COOPERATIVE, CoordinationMode.COOPERATIVE_WAIT], ids=lambda mode: mode.value
)
def test_cooperative_agents_stop_when_partner_status_is_stale(
    team_config: TeamRuleBasedConfig, mode: CoordinationMode
) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=mode)
    now_s = run_until_pushing(agent)

    # 相手の最後の状態から 0.5 s を超えた
    decision = agent.decide(inputs(now_s + 1.0, PartnerStatus(phase=TeamPhase.PUSH, stamp_s=now_s + 0.4)))

    assert decision.state.phase is TeamPhase.READY


def test_push_line_does_not_jump_to_center_when_partner_status_lags_at_push_start(
    team_config: TeamRuleBasedConfig,
) -> None:
    agent = TeamPushAgent(team_config, robot_index=0, mode=CoordinationMode.COOPERATIVE)
    now_s = run_until_pushing(agent)

    # 押し始めた次の周期でも、相手の状態は 1 周期遅れてまだ READY に見える
    decision = agent.decide(inputs(now_s + 0.1, PartnerStatus(phase=TeamPhase.READY, stamp_s=now_s)))

    assert decision.state.phase is TeamPhase.PUSH
    assert decision.push_position_xy_m == pytest.approx((0.8, 0.25))
    assert float(decision.commands[0, 2]) == pytest.approx(0.0, abs=1e-6)
