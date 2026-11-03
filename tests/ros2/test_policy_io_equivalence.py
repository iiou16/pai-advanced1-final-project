"""ROS 2 のメッセージを経由した観測で推論した出力が、学習環境と同じ観測で推論した出力と一致することの検証（P3-04）。

学習環境（Go2GaitEnv・Go2PushEnv）は、シミュレータから読んだ状態から直接観測を作る。
ROS 2 の経路では、シミュレーションノードが状態をメッセージにし、方策のノードがメッセージから観測を作る。
関節の順序（名前での対応づけ）と座標系（Odometry の速度は本体座標系）の取り違えがあれば、出力が食い違う。
"""

import math
import random
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_DIM
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.entities.planar_frame import yaw_from_quat_wxyz
from cooperative_go2.entities.push_observation import (
    PUSH_OBSERVATION_DIM,
    PlanarBodyState,
    PushObservationInputs,
    build_push_observation,
)
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.rigid_body_pose import RigidBodyPose, proprioception_from_pose
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_training_settings
from cooperative_go2.interface_adapters.ros2_messages import (
    make_contact_state,
    make_joint_state,
    make_odometry,
    read_contact_state,
    read_joint_state,
    read_odometry,
)
from cooperative_go2.use_cases.gait_controller import GaitController
from cooperative_go2.use_cases.push_policy_driver import PushPolicyDriver, PushPolicyInputs

REPO_ROOT = Path(__file__).resolve().parents[2]
STAMP_NS = 1_234_560_000
SEED = 7
YAW, PITCH, ROLL = 0.9, 0.08, -0.05
GOAL_XY = (2.4, 1.1)


class LinearPolicy:
    """シードで固定した線形の方策。入力の並びが変わると出力が変わる。受け取った観測を記録する。"""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        generator = torch.Generator().manual_seed(SEED)
        self._weight = torch.randn((input_dim, output_dim), generator=generator)
        self.observations: list[torch.Tensor] = []

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.observations.append(observation)
        return torch.tanh(observation @ self._weight) * 0.3


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    cos_r, sin_r = math.cos(roll), math.sin(roll)
    yaw_matrix = torch.tensor([[cos_y, -sin_y, 0.0], [sin_y, cos_y, 0.0], [0.0, 0.0, 1.0]])
    pitch_matrix = torch.tensor([[cos_p, 0.0, sin_p], [0.0, 1.0, 0.0], [-sin_p, 0.0, cos_p]])
    roll_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, cos_r, -sin_r], [0.0, sin_r, cos_r]])
    return yaw_matrix @ pitch_matrix @ roll_matrix


def _quat_from_rotation(yaw: float, pitch: float, roll: float) -> torch.Tensor:
    # ZYX の順の回転に対応する四元数（w, x, y, z）
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    return torch.tensor([[
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]])  # fmt: skip


@pytest.fixture
def gait() -> GaitConfig:
    return load_gait_settings(REPO_ROOT / "configs" / "gait.yaml").gait


@pytest.fixture
def task() -> PushTaskConfig:
    return load_push_training_settings(REPO_ROOT / "configs" / "push_single_wide.yaml").task


@pytest.fixture
def robot_truth() -> RigidBodyPose:
    """シミュレータが持つ Go2 の胴体の真値（ワールド座標系の速度）を、本体座標系の速度に直したもの。"""
    quat = _quat_from_rotation(YAW, PITCH, ROLL)
    lin_vel_world = torch.tensor([[0.35, -0.12, 0.03]])
    ang_vel_world = torch.tensor([[0.2, -0.1, 0.7]])
    return RigidBodyPose.from_world_velocities(torch.tensor([[0.4, -0.3, 0.29]]), quat, lin_vel_world, ang_vel_world)


def _shuffled_joint_state_message(dof_pos: torch.Tensor, dof_vel: torch.Tensor) -> object:
    # 送り手が別の順序で関節を並べても、名前で対応づけて同じ状態になることを確かめる
    message = make_joint_state(STAMP_NS, dof_pos, dof_vel)
    order = list(range(NUM_GO2_JOINTS))
    random.Random(SEED).shuffle(order)
    message.name = [message.name[index] for index in order]
    message.position = [message.position[index] for index in order]
    message.velocity = [message.velocity[index] for index in order]
    return message


def test_gait_policy_output_matches_training_observation(gait: GaitConfig, robot_truth: RigidBodyPose) -> None:
    generator = torch.Generator().manual_seed(SEED)
    dof_pos = torch.randn((1, NUM_GO2_JOINTS), generator=generator) * 0.3
    dof_vel = torch.randn((1, NUM_GO2_JOINTS), generator=generator)
    rotation = _rotation_matrix(YAW, PITCH, ROLL)
    # 学習環境（GenesisGo2Body.read_state）と同じ定義の自己受容の状態を、回転行列で独立に求める
    training_state = Go2Proprioception(
        base_ang_vel=(rotation.T @ torch.tensor([0.2, -0.1, 0.7])).unsqueeze(0),
        projected_gravity=(rotation.T @ torch.tensor([0.0, 0.0, -1.0])).unsqueeze(0),
        dof_pos=dof_pos,
        dof_vel=dof_vel,
    )
    joint_message = _shuffled_joint_state_message(dof_pos, dof_vel)
    odom_message = make_odometry(STAMP_NS, "map", "go2_a/base_link", robot_truth)
    joint_state = read_joint_state(joint_message)
    ros_state = proprioception_from_pose(read_odometry(odom_message).pose, joint_state.dof_pos, joint_state.dof_vel)
    commands = torch.tensor([[0.4, -0.1, 0.6]])
    training_controller = GaitController(LinearPolicy(GAIT_OBSERVATION_DIM, NUM_GO2_JOINTS), gait, 1, "cpu")
    ros_controller = GaitController(LinearPolicy(GAIT_OBSERVATION_DIM, NUM_GO2_JOINTS), gait, 1, "cpu")

    for _ in range(3):
        expected = training_controller.compute_joint_targets(training_state, commands)
        actual = ros_controller.compute_joint_targets(ros_state, commands)

        assert actual == pytest.approx(expected, abs=1e-5)


def test_push_policy_observation_matches_training_observation(task: PushTaskConfig, robot_truth: RigidBodyPose) -> None:
    box_quat = _quat_from_rotation(-0.4, 0.0, 0.0)
    box_truth = RigidBodyPose.from_world_velocities(
        torch.tensor([[1.3, 0.2, 0.15]]), box_quat, torch.tensor([[0.25, 0.05, 0.0]]), torch.tensor([[0.0, 0.0, -0.3]])
    )
    contact = torch.tensor([True])
    # 学習環境（GenesisPushSimulation.read_*_planar）と同じく、ワールド座標系の速度と姿勢から水平面の状態を作る
    training_inputs = PushObservationInputs(
        robot=PlanarBodyState(
            position_xy_m=torch.tensor([[0.4, -0.3]]),
            yaw_rad=yaw_from_quat_wxyz(robot_truth.quat_wxyz),
            lin_vel_xy_mps=torch.tensor([[0.35, -0.12]]),
            yaw_rate_radps=torch.tensor([0.7]),
        ),
        robot_projected_gravity=(_rotation_matrix(YAW, PITCH, ROLL).T @ torch.tensor([0.0, 0.0, -1.0])).unsqueeze(0),
        box=PlanarBodyState(
            position_xy_m=torch.tensor([[1.3, 0.2]]),
            yaw_rad=torch.tensor([-0.4]),
            lin_vel_xy_mps=torch.tensor([[0.25, 0.05]]),
            yaw_rate_radps=torch.tensor([-0.3]),
        ),
        goal_xy_m=torch.tensor([GOAL_XY]),
        box_contact=contact,
        last_commands=torch.zeros(1, 3),
    )
    expected = build_push_observation(training_inputs, task.observation)
    robot = read_odometry(make_odometry(STAMP_NS, "map", "go2_a/base_link", robot_truth)).pose
    box = read_odometry(make_odometry(STAMP_NS, "map", "box", box_truth)).pose
    policy = LinearPolicy(PUSH_OBSERVATION_DIM, 3)
    driver = PushPolicyDriver(policy, task)
    driver.start(GOAL_XY)

    driver.decide(
        PushPolicyInputs(
            robot=robot.to_planar(),
            robot_projected_gravity=robot.projected_gravity(),
            box=box.to_planar(),
            box_contact=read_contact_state(make_contact_state(STAMP_NS, contact)).in_contact,
        )
    )

    assert policy.observations[0] == pytest.approx(expected, abs=1e-5)
