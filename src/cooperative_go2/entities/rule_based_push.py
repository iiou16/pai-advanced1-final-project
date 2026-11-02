"""規則ベースの箱押し（実験 E1 のベースライン）の速度指令の生成。

箱から目標への向きを「押す方向」とし、ロボットの位置から段階を選んで、その段階の目標点へ向かう速度指令を出す。
段階は毎ステップ位置関係だけから決め、内部状態を持たない。押している最中に箱がずれても、
次のステップで位置関係から段階を選び直せば自然に立て直せるため。

- DONE: 箱が目標領域に入った。停止する。
- PUSH: ロボットが箱の真後ろにいる。箱の中心へ向かって進み、箱を押す。
- APPROACH: ロボットが押し始める位置の線より後ろにいる。押し始める位置へ向かう。
- GO_AROUND: それ以外（箱の横や前）。箱に触れないよう横へ離れてから、押し始める位置の横へ回り込む。

位置はすべてワールド座標系の水平成分。速度指令は本体座標系の [vx, vy, yaw_rate] で、横移動（vy）は使わない。
"""

from dataclasses import dataclass
from enum import IntEnum

import torch
import torch.nn.functional as functional

from cooperative_go2.entities.planar_frame import wrap_angle_rad
from cooperative_go2.entities.push_config import RuleBasedPushConfig


class RuleBasedPushPhase(IntEnum):
    """規則ベースの箱押しの段階。"""

    DONE = 0
    PUSH = 1
    APPROACH = 2
    GO_AROUND = 3


@dataclass(frozen=True)
class RuleBasedPushInputs:
    """速度指令を決める材料。テンソルは並列環境のバッチ。

    Attributes:
        robot_xy_m: ロボットの胴体の位置 (num_envs, 2) [m]。
        robot_yaw_rad: ロボットのヨー角 (num_envs,) [rad]。
        box_xy_m: 箱の中心 (num_envs, 2) [m]。
        goal_xy_m: 目標領域の中心 (num_envs, 2) [m]。
    """

    robot_xy_m: torch.Tensor
    robot_yaw_rad: torch.Tensor
    box_xy_m: torch.Tensor
    goal_xy_m: torch.Tensor


@dataclass(frozen=True)
class RuleBasedPushOutput:
    """規則ベースの箱押しが出した指令と、そのときの段階。

    Attributes:
        commands: 速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]（常に 0）、yaw_rate [rad/s]。
        phases: 環境ごとの段階 (num_envs,)。RuleBasedPushPhase の値。
    """

    commands: torch.Tensor
    phases: torch.Tensor


def compute_rule_based_push_commands(
    inputs: RuleBasedPushInputs, config: RuleBasedPushConfig, goal_radius_m: float
) -> RuleBasedPushOutput:
    """環境ごとに段階を選び、その段階の目標点へ向かう速度指令を返す。

    Args:
        inputs: ロボット・箱・目標の位置。
        config: 速度・旋回のゲインと、回り込みの距離。
        goal_radius_m: 目標領域の半径 [m]。箱の中心がこの範囲（境界を含む）に入ったら停止する。
    """
    box_to_goal = inputs.goal_xy_m - inputs.box_xy_m
    goal_distance = torch.linalg.vector_norm(box_to_goal, dim=1)
    done = goal_distance <= goal_radius_m
    # 停止する環境では押す方向を使わない。箱が目標の中心と重なってもゼロ除算にならないよう 1 で割る
    push_dir = box_to_goal / torch.where(done, torch.ones_like(goal_distance), goal_distance).unsqueeze(1)
    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)

    box_to_robot = inputs.robot_xy_m - inputs.box_xy_m
    along = torch.sum(box_to_robot * push_dir, dim=1)
    lateral = torch.sum(box_to_robot * lateral_dir, dim=1)
    robot_to_box_dir = functional.normalize(-box_to_robot, dim=1)
    aligned = torch.sum(robot_to_box_dir * push_dir, dim=1) >= config.align_min_cos

    push = ~done & (along < 0) & aligned
    approach = ~done & ~push & (along <= -config.standoff_m)
    phases = torch.full_like(along, int(RuleBasedPushPhase.GO_AROUND), dtype=torch.int64)
    for mask, phase in (
        (done, RuleBasedPushPhase.DONE),
        (push, RuleBasedPushPhase.PUSH),
        (approach, RuleBasedPushPhase.APPROACH),
    ):
        phases = torch.where(mask, int(phase), phases)

    side = torch.where(lateral >= 0, 1.0, -1.0).unsqueeze(1) * lateral_dir * config.side_offset_m
    start_point = inputs.box_xy_m - push_dir * config.standoff_m
    # 押す方向の線から十分離れていれば押し始める位置の横へ、近ければ箱に触れないよう真横へ離れる
    go_around_target = torch.where(
        (torch.abs(lateral) >= config.side_offset_m).unsqueeze(1),
        start_point + side,
        inputs.box_xy_m + push_dir * along.unsqueeze(1) + side,
    )
    target = torch.where(
        push.unsqueeze(1), inputs.box_xy_m, torch.where(approach.unsqueeze(1), start_point, go_around_target)
    )
    speed = torch.where(push, config.push_speed_mps, config.approach_speed_mps)
    heading_error = _heading_error_rad(inputs.robot_xy_m, inputs.robot_yaw_rad, target)
    commands = _steer_toward(heading_error, speed, config)
    return RuleBasedPushOutput(
        commands=torch.where(done.unsqueeze(1), torch.zeros_like(commands), commands), phases=phases
    )


def _heading_error_rad(
    robot_xy_m: torch.Tensor, robot_yaw_rad: torch.Tensor, target_xy_m: torch.Tensor
) -> torch.Tensor:
    to_target = target_xy_m - robot_xy_m
    return wrap_angle_rad(torch.atan2(to_target[:, 1], to_target[:, 0]) - robot_yaw_rad)


def _steer_toward(heading_error: torch.Tensor, speed_mps: torch.Tensor, config: RuleBasedPushConfig) -> torch.Tensor:
    # 目標点の方向とのずれに比例して旋回し、向きが合うほど速く前進する。後ろ向きのときは前進せずその場で向きを変える
    yaw_rate = torch.clamp(config.yaw_gain_per_s * heading_error, -config.max_yaw_rate_radps, config.max_yaw_rate_radps)
    forward = speed_mps * torch.clamp(torch.cos(heading_error), min=0.0)
    return torch.stack((forward, torch.zeros_like(forward), yaw_rate), dim=1)
