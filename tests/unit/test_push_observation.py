import math

import torch

from cooperative_go2.entities.push_config import PushObservationScales
from cooperative_go2.entities.push_observation import (
    PUSH_OBSERVATION_DIM,
    PUSH_OBSERVATION_LAYOUT,
    PlanarBodyState,
    PushObservationInputs,
    build_push_observation,
)

SCALES = PushObservationScales(position=0.5, lin_vel=2.0, ang_vel=0.25)


def _body(x: float, y: float, yaw: float, vx: float = 0.0, vy: float = 0.0, yaw_rate: float = 0.0) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x, y]]),
        yaw_rad=torch.tensor([yaw]),
        lin_vel_xy_mps=torch.tensor([[vx, vy]]),
        yaw_rate_radps=torch.tensor([yaw_rate]),
    )


def _split(observation: torch.Tensor) -> dict[str, torch.Tensor]:
    sizes = [size for _, size in PUSH_OBSERVATION_LAYOUT]
    parts = torch.split(observation, sizes, dim=-1)
    return {name: part[0] for (name, _), part in zip(PUSH_OBSERVATION_LAYOUT, parts, strict=True)}


def _inputs(robot: PlanarBodyState, box: PlanarBodyState, **overrides: torch.Tensor) -> PushObservationInputs:
    return PushObservationInputs(
        robot=robot,
        robot_projected_gravity=overrides.get("robot_projected_gravity", torch.tensor([[0.0, 0.0, -1.0]])),
        box=box,
        goal_xy_m=overrides.get("goal_xy_m", torch.tensor([[0.0, 0.0]])),
        box_contact=overrides.get("box_contact", torch.tensor([False])),
        last_commands=overrides.get("last_commands", torch.zeros(1, 3)),
    )


def test_observation_has_layout_dimension_for_each_env() -> None:
    robot = PlanarBodyState(
        position_xy_m=torch.zeros(4, 2),
        yaw_rad=torch.zeros(4),
        lin_vel_xy_mps=torch.zeros(4, 2),
        yaw_rate_radps=torch.zeros(4),
    )
    inputs = PushObservationInputs(
        robot=robot,
        robot_projected_gravity=torch.zeros(4, 3),
        box=robot,
        goal_xy_m=torch.zeros(4, 2),
        box_contact=torch.zeros(4, dtype=torch.bool),
        last_commands=torch.zeros(4, 3),
    )

    observation = build_push_observation(inputs, SCALES)

    assert observation.shape == (4, PUSH_OBSERVATION_DIM)


def test_projected_gravity_is_passed_through_without_scaling() -> None:
    # ロボットが右に 30° ロールした姿勢。本体座標系で重力は y 成分を持つ
    roll = math.radians(30.0)
    gravity = torch.tensor([[0.0, -math.sin(roll), -math.cos(roll)]])
    inputs = _inputs(_body(0.0, 0.0, 0.0), _body(1.0, 0.0, 0.0), robot_projected_gravity=gravity)

    parts = _split(build_push_observation(inputs, SCALES))

    assert torch.allclose(parts["robot_projected_gravity"], gravity[0])


def test_positions_are_relative_to_robot_and_rotated_into_robot_frame() -> None:
    # ロボットは (1, 1) で +y を向く。箱はワールドの +y に 2 m（真正面）、目標はワールドの -x に 3 m（左）
    robot = _body(1.0, 1.0, math.pi / 2)
    box = _body(1.0, 3.0, math.pi / 2)

    parts = _split(build_push_observation(_inputs(robot, box, goal_xy_m=torch.tensor([[-2.0, 1.0]])), SCALES))

    assert torch.allclose(parts["box_rel_pos_xy"], torch.tensor([2.0, 0.0]) * SCALES.position, atol=1e-6)
    assert torch.allclose(parts["goal_rel_pos_xy"], torch.tensor([0.0, 3.0]) * SCALES.position, atol=1e-6)


def test_velocities_are_rotated_into_robot_frame_and_scaled() -> None:
    # ロボットは +y を向き、ワールドの +y に進んでいる＝前進。箱はワールドの +x に動いている＝ロボットの右
    robot = _body(0.0, 0.0, math.pi / 2, vx=0.0, vy=0.5, yaw_rate=0.4)
    box = _body(1.0, 0.0, 0.0, vx=0.3, vy=0.0, yaw_rate=-0.8)

    parts = _split(build_push_observation(_inputs(robot, box), SCALES))

    assert torch.allclose(parts["robot_lin_vel_xy"], torch.tensor([0.5, 0.0]) * SCALES.lin_vel, atol=1e-6)
    assert torch.allclose(parts["robot_yaw_rate"], torch.tensor([0.4 * SCALES.ang_vel]))
    assert torch.allclose(parts["box_lin_vel_xy"], torch.tensor([0.0, -0.3]) * SCALES.lin_vel, atol=1e-6)
    assert torch.allclose(parts["box_yaw_rate"], torch.tensor([-0.8 * SCALES.ang_vel]))


def test_box_orientation_is_continuous_across_half_turn() -> None:
    # 相対ヨー角が +π の手前と -π の手前（ほぼ同じ向き）で、観測はほぼ同じ値になる
    robot = _body(0.0, 0.0, 0.0)
    just_below = _split(build_push_observation(_inputs(robot, _body(1.0, 0.0, math.pi - 1e-3)), SCALES))
    just_above = _split(build_push_observation(_inputs(robot, _body(1.0, 0.0, -math.pi + 1e-3)), SCALES))

    assert torch.allclose(just_below["box_rel_yaw_sin_cos"], just_above["box_rel_yaw_sin_cos"], atol=1e-2)


def test_box_orientation_is_relative_to_robot_heading() -> None:
    robot = _body(0.0, 0.0, math.pi / 2)
    box = _body(1.0, 0.0, math.pi / 2)

    parts = _split(build_push_observation(_inputs(robot, box), SCALES))

    # 同じ向きなら相対ヨー角は 0（sin 0、cos 1）
    assert torch.allclose(parts["box_rel_yaw_sin_cos"], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_contact_and_last_command_are_encoded() -> None:
    robot = _body(0.0, 0.0, 0.0)
    inputs = _inputs(
        robot, _body(1.0, 0.0, 0.0), box_contact=torch.tensor([True]), last_commands=torch.tensor([[0.4, -0.2, 1.0]])
    )

    parts = _split(build_push_observation(inputs, SCALES))

    assert parts["box_contact"].tolist() == [1.0]
    expected_command = torch.tensor([0.4 * SCALES.lin_vel, -0.2 * SCALES.lin_vel, 1.0 * SCALES.ang_vel])
    assert torch.allclose(parts["last_command"], expected_command)
