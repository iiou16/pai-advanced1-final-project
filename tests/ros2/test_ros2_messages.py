"""ROS 2 のメッセージとの変換の契約（時刻・関節の名前での対応づけ・座標系）の検証。"""

import pytest
import torch
from go2_interfaces.msg import AgentStatus
from trajectory_msgs.msg import JointTrajectoryPoint

from cooperative_go2.entities.coordination_state_machine import TeamPhase

from cooperative_go2.entities.go2 import GO2_JOINT_NAMES
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.interface_adapters.ros2_messages import (
    JointNameMismatchError,
    make_joint_state,
    make_joint_trajectory,
    make_odometry,
    make_twist_stamped,
    ns_to_time,
    read_joint_state,
    read_joint_trajectory,
    read_odometry,
    read_twist_stamped,
    time_to_ns,
)
from cooperative_go2.interface_adapters.ros2_team_messages import (
    AgentStatusReport,
    UnknownTeamPhaseError,
    make_agent_status,
    read_agent_status,
)

STAMP_NS = 12_340_000_000 + 20_000_000


def test_time_round_trip_is_exact() -> None:
    stamp = ns_to_time(STAMP_NS)

    assert (stamp.sec, stamp.nanosec) == (12, 360_000_000)
    assert time_to_ns(stamp) == STAMP_NS


def test_joint_trajectory_is_reordered_by_name() -> None:
    targets = torch.arange(12, dtype=torch.float32).unsqueeze(0)
    message = make_joint_trajectory(STAMP_NS, targets)
    message.joint_names = list(reversed(message.joint_names))
    message.points[0].positions = list(reversed(message.points[0].positions))

    read = read_joint_trajectory(message)

    assert read.stamp_ns == STAMP_NS
    assert torch.equal(read.joint_targets, targets)


@pytest.mark.parametrize(
    "names",
    [list(GO2_JOINT_NAMES[:-1]), [*GO2_JOINT_NAMES[:-1], GO2_JOINT_NAMES[0]], [*GO2_JOINT_NAMES[:-1], "unknown"]],
    ids=["missing", "duplicated", "unknown"],
)
def test_joint_state_with_wrong_names_is_rejected(names: list[str]) -> None:
    message = make_joint_state(STAMP_NS, torch.zeros(1, 12), torch.zeros(1, 12))
    message.name = names

    with pytest.raises(JointNameMismatchError):
        read_joint_state(message)


def test_joint_state_with_wrong_value_count_is_rejected() -> None:
    message = make_joint_state(STAMP_NS, torch.zeros(1, 12), torch.zeros(1, 12))
    message.velocity = message.velocity[:-1]

    with pytest.raises(JointNameMismatchError):
        read_joint_state(message)


def test_joint_trajectory_with_multiple_points_is_rejected() -> None:
    message = make_joint_trajectory(STAMP_NS, torch.zeros(1, 12))
    message.points = [message.points[0], JointTrajectoryPoint()]

    with pytest.raises(ValueError):
        read_joint_trajectory(message)


def test_odometry_round_trip_keeps_body_frame_velocities() -> None:
    pose = RigidBodyPose(
        position_m=torch.tensor([[1.0, -2.0, 0.3]]),
        quat_wxyz=torch.tensor([[0.9238795, 0.0, 0.0, 0.3826834]]),
        lin_vel_body_mps=torch.tensor([[0.5, 0.1, 0.0]]),
        ang_vel_body_radps=torch.tensor([[0.0, 0.0, 0.8]]),
    )

    read = read_odometry(make_odometry(STAMP_NS, "map", "go2_a/base_link", pose))

    assert read.stamp_ns == STAMP_NS
    for name in ("position_m", "quat_wxyz", "lin_vel_body_mps", "ang_vel_body_radps"):
        assert torch.equal(getattr(read.pose, name), getattr(pose, name))


def test_twist_round_trip_uses_vx_vy_and_yaw_rate() -> None:
    commands = torch.tensor([[0.4, -0.2, 0.9]])

    read = read_twist_stamped(make_twist_stamped(STAMP_NS, "go2_a/base_link", commands))

    assert read.stamp_ns == STAMP_NS
    assert torch.equal(read.values, commands)


def test_agent_status_round_trip() -> None:
    report = AgentStatusReport(
        agent_id="go2_b", stamp_ns=STAMP_NS, phase=TeamPhase.PUSH, push_position_xy_m=(0.8, -0.25), push_dir_xy=(1.0, 0.0)
    )

    message = make_agent_status(report, "map")

    assert message.ready is True
    assert message.phase == 4
    assert read_agent_status(message) == report
    assert read_agent_status(message).to_partner_status().stamp_s == pytest.approx(12.36)


def test_agent_status_with_unknown_phase_is_rejected() -> None:
    message = AgentStatus()
    message.phase = 9

    with pytest.raises(UnknownTeamPhaseError):
        read_agent_status(message)
