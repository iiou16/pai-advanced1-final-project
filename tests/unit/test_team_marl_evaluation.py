import dataclasses
import math

import pytest
import torch
from conftest import REPO_ROOT, TEAM_NUM_ENVS, TeamMarlEnvFactory

from cooperative_go2.interface_adapters.config_loader import load_team_marl_run_settings
from cooperative_go2.interface_adapters.skrl_team import TeamPolicy
from cooperative_go2.interface_adapters.team_marl_evaluation import evaluate_team_policies


@pytest.fixture
def policies() -> tuple[TeamPolicy, ...]:
    ppo = load_team_marl_run_settings(REPO_ROOT / "configs" / "push_multi.yaml").settings.skrl
    torch.manual_seed(0)
    return (TeamPolicy(ppo, "cpu"), TeamPolicy(ppo, "cpu"))


def test_box_held_in_the_retreat_area_is_a_success_and_others_time_out(
    make_team_marl_env: TeamMarlEnvFactory, policies: tuple[TeamPolicy, ...]
) -> None:
    env, simulation = make_team_marl_env()
    physics_steps = {"count": 0}

    def move_box_into_retreat_in_env_1() -> None:
        physics_steps["count"] += 1
        if physics_steps["count"] == env.gait_steps_per_high_level_step * 3:
            box_xy = torch.tensor([[1.4, 0.0], [3.8, 0.0]])
            simulation.box = dataclasses.replace(simulation.box, position_xy_m=box_xy)

    simulation.step_hook = move_box_into_retreat_in_env_1

    rows = evaluate_team_policies(env, policies)

    assert len(rows) == TEAM_NUM_ENVS
    assert [row.success for row in rows] == [False, True]
    assert rows[0].termination_reason == "TIME_OUT"
    assert rows[0].end_time_s == pytest.approx(env.cfg.episode.episode_length_s)
    # 3 ステップ目に入ったと観測し、そこから 1 s たった 13 ステップ目の観測まで入ったままなので成功する
    assert rows[1].end_time_s == pytest.approx(1.3)
    assert rows[1].corridor_opened and not rows[0].corridor_opened
    assert rows[1].box_dx_m == pytest.approx(2.4, abs=1e-5)


def test_first_box_contact_time_is_recorded_per_robot(
    make_team_marl_env: TeamMarlEnvFactory, policies: tuple[TeamPolicy, ...]
) -> None:
    env, simulation = make_team_marl_env()
    physics_steps = {"count": 0}

    def go2_a_touches_the_box_in_the_second_step() -> None:
        physics_steps["count"] += 1
        if physics_steps["count"] == env.gait_steps_per_high_level_step * 2:
            simulation.contacts = (torch.ones(TEAM_NUM_ENVS, dtype=torch.bool), simulation.contacts[1])

    simulation.step_hook = go2_a_touches_the_box_in_the_second_step

    rows = evaluate_team_policies(env, policies)

    assert rows[0].first_box_contact_times_s == "0.2|-"
    assert rows[0].first_box_contact_spread_s is None


def test_rotation_falls_and_contact_spread_are_recorded(
    make_team_marl_env: TeamMarlEnvFactory, policies: tuple[TeamPolicy, ...]
) -> None:
    env, simulation = make_team_marl_env()
    gait_steps = env.gait_steps_per_high_level_step
    physics_steps = {"count": 0}

    def scripted_episode() -> None:
        physics_steps["count"] += 1
        count = physics_steps["count"]
        if count == gait_steps * 2:
            # 箱を押す方向（+x）から 170° 回す。箱は 180° 回しても同じ形なので、回転は 10° と数える
            simulation.box = dataclasses.replace(simulation.box, yaw_rad=torch.full((TEAM_NUM_ENVS,), math.pi - 0.17))
            simulation.contacts = (torch.ones(TEAM_NUM_ENVS, dtype=torch.bool), simulation.contacts[1])
        if count == gait_steps * 3:
            simulation.contacts = (simulation.contacts[0], torch.ones(TEAM_NUM_ENVS, dtype=torch.bool))
        if count == gait_steps * 4:
            simulation.euler_deg[1] = torch.tensor([[20.0, 0.0, 0.0]] * TEAM_NUM_ENVS)

    simulation.step_hook = scripted_episode

    rows = evaluate_team_policies(env, policies)

    row = rows[0]
    assert row.termination_reason == "FALL"
    assert row.falls == 1
    assert row.box_max_abs_yaw_deg == pytest.approx(math.degrees(0.17), abs=0.1)
    assert row.first_box_contact_times_s == "0.2|0.3"
    assert row.first_box_contact_spread_s == pytest.approx(0.1)
