"""2 台の協調学習の、Actor の観測と Critic の状態の構成。

Actor の観測は、ROS 2 の実行時にも協調のノードが受け取れる情報だけで作る。
自分の odom（位置・向き・速度・傾き）、箱の状態（/team/object_state）、退避領域（ClearCorridor の目標）、
自分が前に出した指令、通信で届いた相手の状態（/team/agent_status）に限り、箱との接触や相手の真の現在の状態は入れない。
Critic の状態は、学習時にだけ使える全体の真の状態（2 台の状態・接触・指令、箱、時間）で作り、Actor には渡さない。

学習済みの方策は観測の順序と正規化に依存する。順序を変えるとチェックポイントの互換性が失われる。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.partner_channel import PartnerReception
from cooperative_go2.entities.planar_frame import relative_xy_in_robot_frame, rotate_world_to_robot_xy, wrap_angle_rad
from cooperative_go2.entities.push_config import PushObservationScales
from cooperative_go2.entities.push_observation import PlanarBodyState

# Actor の観測を先頭から順に (成分名, 次元数) で並べたもの。ベクトル量はすべて自分のロボット座標系
TEAM_ACTOR_OBSERVATION_LAYOUT: tuple[tuple[str, int], ...] = (
    ("robot_lin_vel_xy", 2),
    ("robot_yaw_rate", 1),
    ("robot_projected_gravity", 3),
    ("box_rel_pos_xy", 2),
    ("box_rel_yaw_sin_cos", 2),
    ("box_lin_vel_xy", 2),
    ("box_yaw_rate", 1),
    ("goal_rel_pos_xy", 2),
    ("last_command", 3),
    ("side_sign", 1),
    ("partner_rel_pos_xy", 2),
    ("partner_rel_yaw_sin_cos", 2),
    ("partner_command", 3),
    ("partner_age", 1),
    ("partner_received", 1),
)
TEAM_ACTOR_OBSERVATION_DIM = sum(size for _, size in TEAM_ACTOR_OBSERVATION_LAYOUT)

# Critic の状態の、ロボット 1 台分の成分。ワールド座標系
TEAM_STATE_ROBOT_LAYOUT: tuple[tuple[str, int], ...] = (
    ("position_xy", 2),
    ("yaw_sin_cos", 2),
    ("lin_vel_xy", 2),
    ("yaw_rate", 1),
    ("projected_gravity", 3),
    ("last_command", 3),
    ("box_contact", 1),
)
# Critic の状態の、ロボットに依らない成分と、最後に付ける「この状態を使う Actor の側」。ワールド座標系
TEAM_STATE_SHARED_LAYOUT: tuple[tuple[str, int], ...] = (
    ("box_position_xy", 2),
    ("box_yaw_sin_cos", 2),
    ("box_lin_vel_xy", 2),
    ("box_yaw_rate", 1),
    ("goal_position_xy", 2),
    ("episode_progress", 1),
    ("own_side_sign", 1),
)


def team_state_dim(num_robots: int) -> int:
    """ロボット num_robots 台の Critic の状態の次元数を返す。"""
    robot_dim = sum(size for _, size in TEAM_STATE_ROBOT_LAYOUT)
    return num_robots * robot_dim + sum(size for _, size in TEAM_STATE_SHARED_LAYOUT)


@dataclass(frozen=True)
class TeamActorObservationInputs:
    """1 台の Actor の観測の材料。ROS 2 の実行時にも協調のノードが持てる情報だけを持つ。

    Attributes:
        robot: 自分の胴体の状態（odom）。
        robot_projected_gravity: 自分の本体座標系で表した重力方向の単位ベクトル (num_envs, 3)。水平なら (0, 0, -1)。
        box: 箱の状態（/team/object_state）。
        goal_xy_m: 退避領域の中心 (num_envs, 2)、ワールド座標系 [m]。
        last_commands: 自分が直前に出した速度指令 (num_envs, 3)。エピソードの最初は 0。
        side_sign: 押す方向から見た自分の側。左が 1.0、右が -1.0。
        partner: 通信で最後に届いた相手の状態。
    """

    robot: PlanarBodyState
    robot_projected_gravity: torch.Tensor
    box: PlanarBodyState
    goal_xy_m: torch.Tensor
    last_commands: torch.Tensor
    side_sign: float
    partner: PartnerReception


@dataclass(frozen=True)
class TeamRobotTruth:
    """Critic に渡す、1 台のロボットの真の状態。

    Attributes:
        body: 胴体の状態。
        projected_gravity: 本体座標系で表した重力方向の単位ベクトル (num_envs, 3)。
        last_commands: 直前に出した速度指令 (num_envs, 3)。
        box_contact: 箱に接触している環境のマスク (num_envs,)（シミュレーションの接触判定）。
    """

    body: PlanarBodyState
    projected_gravity: torch.Tensor
    last_commands: torch.Tensor
    box_contact: torch.Tensor


@dataclass(frozen=True)
class TeamStateInputs:
    """Critic の状態の材料（全体の真の状態）。

    Attributes:
        robots: ロボットごとの真の状態。シーンの robots と同じ順。
        box: 箱の状態。
        goal_xy_m: 退避領域の中心 (num_envs, 2)、ワールド座標系 [m]。
        episode_progress: エピソードの経過の割合 (num_envs,)。始めが 0、時間切れが 1。
    """

    robots: tuple[TeamRobotTruth, ...]
    box: PlanarBodyState
    goal_xy_m: torch.Tensor
    episode_progress: torch.Tensor


def _command_scales(scales: PushObservationScales, like: torch.Tensor) -> torch.Tensor:
    return torch.tensor([scales.lin_vel, scales.lin_vel, scales.ang_vel], dtype=like.dtype, device=like.device)


def _sin_cos(angle_rad: torch.Tensor) -> torch.Tensor:
    return torch.stack((torch.sin(angle_rad), torch.cos(angle_rad)), dim=1)


def build_team_actor_observation(
    inputs: TeamActorObservationInputs, scales: PushObservationScales, max_partner_age_s: float
) -> torch.Tensor:
    """TEAM_ACTOR_OBSERVATION_LAYOUT の順に正規化した Actor の観測 (num_envs, TEAM_ACTOR_OBSERVATION_DIM) を作る。

    向きは相対ヨー角の sin と cos で表す（角度を直接入れると ±π で値が飛ぶため）。相手の位置と向きは、届いた状態の
    時点の相手の位置を、自分の現在の位置と向きから見たもの。相手の状態の古さは max_partner_age_s で割って 0〜1 にする。
    相手の状態が一度も届いていない環境では、相手の成分を 0、古さを 1、partner_received を 0 にする。

    Args:
        inputs: 観測の材料。
        scales: 成分ごとの正規化係数。速度指令の vx・vy には lin_vel、yaw_rate には ang_vel を掛ける。
        max_partner_age_s: 相手の状態の古さの上限 [s]。TeamCommunicationConfig.max_age_s。正の値。
    """
    robot = inputs.robot
    box = inputs.box
    partner = inputs.partner
    message = partner.message
    dtype = inputs.last_commands.dtype
    received = partner.received.to(dtype).unsqueeze(1)
    command_scales = _command_scales(scales, inputs.last_commands)
    partner_rel_xy = relative_xy_in_robot_frame(message.position_xy_m, robot.position_xy_m, robot.yaw_rad)
    partner_rel_yaw = _sin_cos(wrap_angle_rad(message.yaw_rad - robot.yaw_rad))
    side = torch.full_like(robot.yaw_rad, inputs.side_sign).unsqueeze(1)
    return torch.cat(
        (
            rotate_world_to_robot_xy(robot.lin_vel_xy_mps, robot.yaw_rad) * scales.lin_vel,
            robot.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
            inputs.robot_projected_gravity,
            relative_xy_in_robot_frame(box.position_xy_m, robot.position_xy_m, robot.yaw_rad) * scales.position,
            _sin_cos(wrap_angle_rad(box.yaw_rad - robot.yaw_rad)),
            rotate_world_to_robot_xy(box.lin_vel_xy_mps, robot.yaw_rad) * scales.lin_vel,
            box.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
            relative_xy_in_robot_frame(inputs.goal_xy_m, robot.position_xy_m, robot.yaw_rad) * scales.position,
            inputs.last_commands * command_scales,
            side,
            # 届いていない環境の相手の成分は、自分の位置・向きから見た値にせず 0 にする
            partner_rel_xy * scales.position * received,
            partner_rel_yaw * received,
            message.command * command_scales * received,
            (partner.age_s / max_partner_age_s).unsqueeze(1),
            received,
        ),
        dim=-1,
    )


def build_team_state(inputs: TeamStateInputs, own_side_sign: float, scales: PushObservationScales) -> torch.Tensor:
    """Critic の状態 (num_envs, team_state_dim(ロボットの数)) を作る。

    ロボットの成分（TEAM_STATE_ROBOT_LAYOUT）をシーンの robots の順に並べ、そのあとに TEAM_STATE_SHARED_LAYOUT の
    成分を並べる。2 台の並び順は使う Actor によらず同じにし、最後の own_side_sign でどちらの Actor の価値かを区別する。
    パラメータを 2 台で共有しても、ロボットごとの報酬（転倒・衝突などの減点）の違いを価値に反映できるようにするため。

    Args:
        inputs: 全体の真の状態。
        own_side_sign: この状態を使う Actor の側（左 1.0、右 -1.0）。
        scales: 成分ごとの正規化係数。
    """
    box = inputs.box
    command_scales = _command_scales(scales, inputs.goal_xy_m)
    robot_parts = [
        torch.cat(
            (
                robot.body.position_xy_m * scales.position,
                _sin_cos(robot.body.yaw_rad),
                robot.body.lin_vel_xy_mps * scales.lin_vel,
                robot.body.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
                robot.projected_gravity,
                robot.last_commands * command_scales,
                robot.box_contact.to(inputs.goal_xy_m.dtype).unsqueeze(1),
            ),
            dim=-1,
        )
        for robot in inputs.robots
    ]
    shared = torch.cat(
        (
            box.position_xy_m * scales.position,
            _sin_cos(box.yaw_rad),
            box.lin_vel_xy_mps * scales.lin_vel,
            box.yaw_rate_radps.unsqueeze(1) * scales.ang_vel,
            inputs.goal_xy_m * scales.position,
            inputs.episode_progress.unsqueeze(1),
            torch.full_like(inputs.episode_progress, own_side_sign).unsqueeze(1),
        ),
        dim=-1,
    )
    return torch.cat((*robot_parts, shared), dim=-1)
