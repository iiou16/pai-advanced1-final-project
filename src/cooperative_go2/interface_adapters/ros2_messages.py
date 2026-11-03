"""ROS 2 のメッセージと、方策・シミュレーションが使う型との変換。

- 時刻はシミュレーション時刻の整数ナノ秒（entities.command_expiry）で扱う。
- 関節の値は名前で対応づける。送り手の並びに頼らず、GO2_JOINT_NAMES の順に並べ直して返す。
- Odometry の速度は ROS 2 の慣例どおり子の座標系（本体座標系）で表す。
- テンソルは 1 台分（先頭の次元が 1）で CPU に置く。
"""

from dataclasses import dataclass

import torch
from builtin_interfaces.msg import Time
from geometry_msgs.msg import TransformStamped, TwistStamped
from go2_interfaces.msg import ContactState
from nav_msgs.msg import Odometry
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from cooperative_go2.entities.command_expiry import NANOSECONDS_PER_SECOND
from cooperative_go2.entities.go2 import GO2_JOINT_NAMES
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose
from cooperative_go2.use_cases.gait_command_follower import StampedVelocityCommand


class JointNameMismatchError(ValueError):
    """メッセージの関節名が Go2 の 12 関節とちょうど一致しない（不足・余分・重複・値の数の不一致）ことを表す。"""


@dataclass(frozen=True)
class StampedJointState:
    """時刻付きの関節の状態。

    Attributes:
        stamp_ns: シミュレーション時刻 [ns]。
        dof_pos: 関節角 (1, 12) [rad]、GO2_JOINT_NAMES の順。
        dof_vel: 関節角速度 (1, 12) [rad/s]、GO2_JOINT_NAMES の順。
    """

    stamp_ns: int
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor


@dataclass(frozen=True)
class StampedPose:
    """時刻付きの剛体の状態。

    Attributes:
        stamp_ns: シミュレーション時刻 [ns]。
        pose: 位置・姿勢・本体座標系の速度。
    """

    stamp_ns: int
    pose: RigidBodyPose


@dataclass(frozen=True)
class StampedJointTargets:
    """時刻付きの目標関節角。

    Attributes:
        stamp_ns: 目標の元になった関節の状態の時刻 [ns]。
        joint_targets: 目標関節角 (1, 12) [rad]、GO2_JOINT_NAMES の順。
    """

    stamp_ns: int
    joint_targets: torch.Tensor


@dataclass(frozen=True)
class StampedContact:
    """時刻付きの接触状態。

    Attributes:
        stamp_ns: シミュレーション時刻 [ns]。
        in_contact: 触れているか (1,)。
    """

    stamp_ns: int
    in_contact: torch.Tensor


def ns_to_time(stamp_ns: int) -> Time:
    """シミュレーション時刻 [ns] を builtin_interfaces/Time にする。"""
    seconds, nanoseconds = divmod(stamp_ns, NANOSECONDS_PER_SECOND)
    return Time(sec=seconds, nanosec=nanoseconds)


def time_to_ns(stamp: Time) -> int:
    """builtin_interfaces/Time をシミュレーション時刻 [ns] にする。"""
    return stamp.sec * NANOSECONDS_PER_SECOND + stamp.nanosec


def make_clock(stamp_ns: int) -> Clock:
    """/clock に配信するメッセージを作る。"""
    return Clock(clock=ns_to_time(stamp_ns))


def make_joint_state(stamp_ns: int, dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> JointState:
    """関節角と関節角速度 (1, 12)（GO2_JOINT_NAMES の順）から JointState を作る。"""
    message = JointState()
    message.header.stamp = ns_to_time(stamp_ns)
    message.name = list(GO2_JOINT_NAMES)
    message.position = _row_to_floats(dof_pos)
    message.velocity = _row_to_floats(dof_vel)
    return message


def read_joint_state(message: JointState) -> StampedJointState:
    """JointState を、GO2_JOINT_NAMES の順に並べ直した関節の状態にする。

    Raises:
        JointNameMismatchError: 関節名が Go2 の 12 関節とちょうど一致しない、または値の数が関節の数と違う。
    """
    order = _go2_joint_order(message.name)
    return StampedJointState(
        stamp_ns=time_to_ns(message.header.stamp),
        dof_pos=_reordered(message.position, order, "position"),
        dof_vel=_reordered(message.velocity, order, "velocity"),
    )


def make_joint_trajectory(stamp_ns: int, joint_targets: torch.Tensor) -> JointTrajectory:
    """目標関節角 (1, 12) から、点が 1 つの JointTrajectory を作る。header.stamp は元になった状態の時刻。"""
    message = JointTrajectory()
    message.header.stamp = ns_to_time(stamp_ns)
    message.joint_names = list(GO2_JOINT_NAMES)
    point = JointTrajectoryPoint()
    point.positions = _row_to_floats(joint_targets)
    message.points = [point]
    return message


def read_joint_trajectory(message: JointTrajectory) -> StampedJointTargets:
    """点が 1 つの JointTrajectory を、GO2_JOINT_NAMES の順に並べ直した目標関節角にする。

    Raises:
        JointNameMismatchError: 関節名が Go2 の 12 関節とちょうど一致しない、または値の数が関節の数と違う。
        ValueError: 点がちょうど 1 つではない。
    """
    if len(message.points) != 1:
        raise ValueError(f"関節目標の点はちょうど 1 つが必要: {len(message.points)} 個")
    order = _go2_joint_order(message.joint_names)
    return StampedJointTargets(
        stamp_ns=time_to_ns(message.header.stamp),
        joint_targets=_reordered(message.points[0].positions, order, "positions"),
    )


def make_odometry(stamp_ns: int, frame_id: str, child_frame_id: str, pose: RigidBodyPose) -> Odometry:
    """剛体の状態から Odometry を作る。速度は child_frame_id（本体座標系）で表す。"""
    message = Odometry()
    message.header.stamp = ns_to_time(stamp_ns)
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    position = _row_to_floats(pose.position_m)
    message.pose.pose.position.x, message.pose.pose.position.y, message.pose.pose.position.z = position
    _set_quaternion(message.pose.pose.orientation, pose.quat_wxyz)
    linear = message.twist.twist.linear
    linear.x, linear.y, linear.z = _row_to_floats(pose.lin_vel_body_mps)
    angular = message.twist.twist.angular
    angular.x, angular.y, angular.z = _row_to_floats(pose.ang_vel_body_radps)
    return message


def read_odometry(message: Odometry) -> StampedPose:
    """Odometry を剛体の状態にする。速度は本体座標系として読む。"""
    position = message.pose.pose.position
    orientation = message.pose.pose.orientation
    linear = message.twist.twist.linear
    angular = message.twist.twist.angular
    return StampedPose(
        stamp_ns=time_to_ns(message.header.stamp),
        pose=RigidBodyPose(
            position_m=torch.tensor([[position.x, position.y, position.z]]),
            quat_wxyz=torch.tensor([[orientation.w, orientation.x, orientation.y, orientation.z]]),
            lin_vel_body_mps=torch.tensor([[linear.x, linear.y, linear.z]]),
            ang_vel_body_radps=torch.tensor([[angular.x, angular.y, angular.z]]),
        ),
    )


def make_transform(stamp_ns: int, frame_id: str, child_frame_id: str, pose: RigidBodyPose) -> TransformStamped:
    """剛体の位置・姿勢から、frame_id → child_frame_id の座標変換を作る。"""
    message = TransformStamped()
    message.header.stamp = ns_to_time(stamp_ns)
    message.header.frame_id = frame_id
    message.child_frame_id = child_frame_id
    translation = message.transform.translation
    translation.x, translation.y, translation.z = _row_to_floats(pose.position_m)
    _set_quaternion(message.transform.rotation, pose.quat_wxyz)
    return message


def make_twist_stamped(stamp_ns: int, frame_id: str, commands: torch.Tensor) -> TwistStamped:
    """速度指令 (1, 3)（vx, vy, yaw_rate）から TwistStamped を作る。header.stamp は指令を決めた時刻。"""
    vx, vy, yaw_rate = _row_to_floats(commands)
    message = TwistStamped()
    message.header.stamp = ns_to_time(stamp_ns)
    message.header.frame_id = frame_id
    message.twist.linear.x = vx
    message.twist.linear.y = vy
    message.twist.angular.z = yaw_rate
    return message


def read_twist_stamped(message: TwistStamped) -> StampedVelocityCommand:
    """TwistStamped を速度指令 (1, 3)（vx, vy, yaw_rate）にする。それ以外の成分は使わない。"""
    twist = message.twist
    return StampedVelocityCommand(
        stamp_ns=time_to_ns(message.header.stamp),
        values=torch.tensor([[twist.linear.x, twist.linear.y, twist.angular.z]]),
    )


def make_contact_state(stamp_ns: int, in_contact: torch.Tensor) -> ContactState:
    """接触状態 (1,) から ContactState を作る。"""
    message = ContactState()
    message.header.stamp = ns_to_time(stamp_ns)
    message.in_contact = bool(in_contact[0])
    return message


def read_contact_state(message: ContactState) -> StampedContact:
    """ContactState を接触状態 (1,) にする。"""
    return StampedContact(stamp_ns=time_to_ns(message.header.stamp), in_contact=torch.tensor([message.in_contact]))


def _row_to_floats(values: torch.Tensor) -> list[float]:
    if values.shape[0] != 1:
        raise ValueError(f"1 台分（先頭の次元が 1）のテンソルが必要: 形 {tuple(values.shape)}")
    return [float(value) for value in values[0].tolist()]


def _set_quaternion(target: object, quat_wxyz: torch.Tensor) -> None:
    w, x, y, z = _row_to_floats(quat_wxyz)
    target.w, target.x, target.y, target.z = w, x, y, z  # type: ignore[attr-defined]


def _go2_joint_order(names: list[str]) -> list[int]:
    # GO2_JOINT_NAMES の各関節が、メッセージの何番目にあるか
    if len(names) != len(set(names)) or set(names) != set(GO2_JOINT_NAMES):
        raise JointNameMismatchError(f"Go2 の 12 関節とちょうど一致しない関節名: {list(names)}")
    index_by_name = {name: index for index, name in enumerate(names)}
    return [index_by_name[name] for name in GO2_JOINT_NAMES]


def _reordered(values: object, order: list[int], field_name: str) -> torch.Tensor:
    sequence = list(values)  # type: ignore[call-overload]
    if len(sequence) != len(order):
        raise JointNameMismatchError(f"{field_name} の値の数 {len(sequence)} が関節の数 {len(order)} と違う")
    return torch.tensor([[sequence[index] for index in order]])
