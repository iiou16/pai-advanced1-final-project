import pytest
import torch

from cooperative_go2.entities.go2_state import initial_go2_state


def test_initial_state_is_standing_still_in_default_pose() -> None:
    default_dof_pos = torch.linspace(-1.0, 1.0, 12)

    state = initial_go2_state(
        base_pos_m=torch.tensor([0.0, 0.0, 0.42]),
        projected_gravity=torch.tensor([0.0, 0.0, -1.0]),
        default_dof_pos=default_dof_pos,
    )

    assert torch.equal(state.proprioception.dof_pos, default_dof_pos.unsqueeze(0))
    assert torch.equal(state.proprioception.dof_vel, torch.zeros(1, 12))
    assert torch.equal(state.base_lin_vel, torch.zeros(1, 3))
    assert state.base_pos_m[0, 2].item() == pytest.approx(0.42)


def test_reset_where_replaces_only_masked_rows(make_go2_state) -> None:
    moving = make_go2_state(num_envs=3, base_lin_vel=torch.ones(3, 3), dof_vel=torch.ones(3, 12))
    initial = initial_go2_state(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]), torch.zeros(12))

    reset = moving.reset_where(torch.tensor([False, True, False]), initial)

    assert torch.equal(reset.base_lin_vel[:, 0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(reset.proprioception.dof_vel[:, 0], torch.tensor([1.0, 0.0, 1.0]))
    assert torch.equal(reset.proprioception.projected_gravity[1], torch.tensor([0.0, 0.0, -1.0]))
    # 元の状態は変えない
    assert torch.equal(moving.base_lin_vel, torch.ones(3, 3))


def test_reset_of_all_envs_broadcasts_initial_state_to_every_row(make_go2_state) -> None:
    moving = make_go2_state(num_envs=4, dof_vel=torch.ones(4, 12))
    initial = initial_go2_state(torch.zeros(3), torch.tensor([0.0, 0.0, -1.0]), torch.zeros(12))

    reset = moving.reset_where(torch.ones(4, dtype=torch.bool), initial)

    assert reset.proprioception.dof_vel.shape == (4, 12)
    assert torch.equal(reset.proprioception.dof_vel, torch.zeros(4, 12))
