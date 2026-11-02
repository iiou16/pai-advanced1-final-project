import torch

from cooperative_go2.entities.gait_config import GaitObservationScales
from cooperative_go2.entities.gait_observation import (
    GAIT_OBSERVATION_DIM,
    GAIT_OBSERVATION_LAYOUT,
    GaitObservationInputs,
    build_gait_observation,
)
from cooperative_go2.entities.go2 import GO2_JOINT_NAMES
from cooperative_go2.entities.go2_state import Go2Proprioception

SCALES = GaitObservationScales(lin_vel=2.0, ang_vel=0.25, dof_pos=1.0, dof_vel=0.05)


def _inputs() -> GaitObservationInputs:
    return GaitObservationInputs(
        proprioception=Go2Proprioception(
            base_ang_vel=torch.tensor([[1.0, 2.0, 4.0]]),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]]),
            dof_pos=torch.full((1, 12), 0.3),
            dof_vel=torch.full((1, 12), 10.0),
        ),
        commands=torch.tensor([[0.5, -0.5, 1.0]]),
        last_actions=torch.arange(12, dtype=torch.float32).unsqueeze(0),
        default_dof_pos=torch.full((12,), 0.1),
    )


def test_joint_order_is_fixed() -> None:
    # 学習済みの方策の互換性を守るため、関節の順序を固定する
    assert GO2_JOINT_NAMES == (
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    )  # fmt: skip


def test_observation_layout_is_fixed() -> None:
    assert GAIT_OBSERVATION_LAYOUT == (
        ("base_ang_vel", 3),
        ("projected_gravity", 3),
        ("velocity_command", 3),
        ("joint_pos_offset", 12),
        ("joint_vel", 12),
        ("last_action", 12),
    )
    assert GAIT_OBSERVATION_DIM == 45


def test_components_are_scaled_and_placed_at_fixed_positions() -> None:
    observation = build_gait_observation(_inputs(), SCALES)

    assert observation.shape == (1, 45)
    row = observation[0]
    assert torch.allclose(row[0:3], torch.tensor([0.25, 0.5, 1.0]))
    assert torch.allclose(row[3:6], torch.tensor([0.0, 0.0, -1.0]))
    # vx・vy は lin_vel、yaw_rate は ang_vel で正規化する
    assert torch.allclose(row[6:9], torch.tensor([1.0, -1.0, 0.25]))
    assert torch.allclose(row[9:21], torch.full((12,), 0.2))
    assert torch.allclose(row[21:33], torch.full((12,), 0.5))
    assert torch.allclose(row[33:45], torch.arange(12, dtype=torch.float32))
