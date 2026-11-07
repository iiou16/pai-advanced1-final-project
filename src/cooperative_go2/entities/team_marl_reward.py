"""2 台の協調学習の報酬の計算。

箱が退避領域に近づいた量・成功・箱の回転は 2 台で共有する（どちらの行動の結果かを分けられないため）。
押す位置に近づいた量・意図した面での接触・転倒・シーン外への離脱・ロボット同士の衝突・過大な速度指令はロボットごとに与える。
押す位置に近づいた量は、箱から離れた開始位置で箱に届く前に学習が止まらないようにするための誘導で、1 ステップの距離の
差なので 1 エピソードの合計は開始と終わりの距離の差に限られる（行き来しても増えない）。押す位置は規則ベースの協調と
同じく、箱の後ろに横に並ぶ位置（自分の側は agent_side_signs で前もって決める）。
転倒と離脱はどちらもエピソードを終えるので、同じ大きさで減点する。離脱を減点しないと、毎ステップの小さな減点
（指令の大きさなど）を避けるために、シーンの外へ歩いてエピソードを早く終える行動が得になるため。
すべて並列環境のバッチのまま計算し、各項は (ロボットの数, num_envs) のテンソルで表す。
"""

from dataclasses import dataclass, fields

import torch

from cooperative_go2.entities.planar_frame import wrap_angle_rad
from cooperative_go2.entities.push_reward import is_pushing_toward_goal
from cooperative_go2.entities.team_marl_config import TeamMarlRewardConfig, TeamMarlRewardScales


@dataclass(frozen=True)
class TeamMarlRewardInputs:
    """報酬の計算に使う、高レベル方策の 1 ステップ分の材料。位置はワールド座標系の水平成分 [m]。

    Attributes:
        previous_goal_distance_m: 前のステップでの箱と退避領域の中心の距離 (num_envs,) [m]。
        goal_distance_m: このステップでの箱と退避領域の中心の距離 (num_envs,) [m]。
        succeeded: 終了理由が成功になった環境のマスク (num_envs,)。同じステップで転倒などが優先された環境は含めない。
        previous_box_yaw_rad: 前のステップでの箱のヨー角 (num_envs,) [rad]。
        box_yaw_rad: このステップでの箱のヨー角 (num_envs,) [rad]。
        box_xy_m: 箱の中心 (num_envs, 2)。
        goal_xy_m: 退避領域の中心 (num_envs, 2)。
        robots_xy_m: ロボットごとの胴体の位置 (num_envs, 2)。
        previous_push_distances_m: ロボットごとの、前のステップでの自分の押す位置までの距離 (num_envs,) [m]。
        push_distances_m: ロボットごとの、このステップでの自分の押す位置までの距離 (num_envs,) [m]。
        box_contacts: ロボットごとの、箱に接触している環境のマスク (num_envs,)。
        fallen: ロボットごとの、このステップで転倒した環境のマスク (num_envs,)。
        out_of_bounds: ロボットごとの、そのロボットか箱がシーンの有効範囲の外に出た環境のマスク (num_envs,)。
        commands: ロボットごとの、このステップで出した速度指令 (num_envs, 3)。
    """

    previous_goal_distance_m: torch.Tensor
    goal_distance_m: torch.Tensor
    succeeded: torch.Tensor
    previous_box_yaw_rad: torch.Tensor
    box_yaw_rad: torch.Tensor
    box_xy_m: torch.Tensor
    goal_xy_m: torch.Tensor
    robots_xy_m: tuple[torch.Tensor, ...]
    previous_push_distances_m: tuple[torch.Tensor, ...]
    push_distances_m: tuple[torch.Tensor, ...]
    box_contacts: tuple[torch.Tensor, ...]
    fallen: tuple[torch.Tensor, ...]
    out_of_bounds: tuple[torch.Tensor, ...]
    commands: tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class TeamMarlRewardTerms:
    """報酬の項ごとの値。各フィールドは (ロボットの数, num_envs) のテンソル。フィールド名は TeamMarlRewardScales と一致する。

    共有の項（box_progress・success・box_rotation）は、全ロボットの行に同じ値が入る。
    """

    box_progress: torch.Tensor
    success: torch.Tensor
    box_rotation: torch.Tensor
    approach_progress: torch.Tensor
    intended_face_contact: torch.Tensor
    robot_fall: torch.Tensor
    out_of_bounds: torch.Tensor
    robot_collision: torch.Tensor
    control_effort: torch.Tensor

    def weighted(self, scales: TeamMarlRewardScales) -> "TeamMarlRewardTerms":
        """各項に係数を掛けた項を返す。係数は高レベル方策の 1 ステップあたりの重み。"""
        return TeamMarlRewardTerms(
            **{field.name: getattr(self, field.name) * getattr(scales, field.name) for field in fields(self)}
        )

    def per_robot_total(self) -> torch.Tensor:
        """ロボットごとの報酬の合計 (ロボットの数, num_envs) を返す。"""
        return torch.stack([getattr(self, field.name) for field in fields(self)]).sum(dim=0)

    def zeroed_where(self, mask: torch.Tensor) -> "TeamMarlRewardTerms":
        """mask (num_envs,) が True の環境の値を、すべての項・すべてのロボットで 0 にした項を返す。"""
        return TeamMarlRewardTerms(
            **{field.name: getattr(self, field.name).masked_fill(mask, 0.0) for field in fields(self)}
        )

    def robot_mean(self) -> dict[str, torch.Tensor]:
        """ログ出力用に、項の名前 → ロボットについて平均した値 (num_envs,) の対応を返す。"""
        return {field.name: getattr(self, field.name).mean(dim=0) for field in fields(self)}


TEAM_MARL_REWARD_TERM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(TeamMarlRewardTerms))


def push_position_distances_m(
    robots_xy_m: tuple[torch.Tensor, ...], box_xy_m: torch.Tensor, goal_xy_m: torch.Tensor,
    side_signs: tuple[float, ...], config: TeamMarlRewardConfig,
) -> tuple[torch.Tensor, ...]:  # fmt: skip
    """ロボットごとの、自分の押す位置までの距離 (num_envs,) [m] を返す。

    押す位置は、箱の中心から押す方向（箱 → 退避領域の中心）の逆へ push_standoff_m、押す方向の左を正とする横へ
    side_sign × push_lateral_offset_m ずらした位置。規則ベースの協調の押す位置（箱の回転に応じたずらしを除く）と同じ。
    箱が退避領域の中心にあり押す方向が定まらない環境では、押す方向を +x とする。
    """
    to_goal = goal_xy_m - box_xy_m
    norm = torch.linalg.vector_norm(to_goal, dim=1, keepdim=True)
    default_dir = torch.zeros_like(to_goal)
    default_dir[:, 0] = 1.0
    push_dir = torch.where(norm > 0, to_goal / torch.where(norm > 0, norm, torch.ones_like(norm)), default_dir)
    lateral_dir = torch.stack((-push_dir[:, 1], push_dir[:, 0]), dim=1)
    behind = box_xy_m - push_dir * config.push_standoff_m
    return tuple(
        torch.linalg.vector_norm(robot_xy - (behind + lateral_dir * (sign * config.push_lateral_offset_m)), dim=1)
        for robot_xy, sign in zip(robots_xy_m, side_signs, strict=True)
    )


def robots_too_close(robots_xy_m: tuple[torch.Tensor, ...], min_separation_m: float) -> tuple[torch.Tensor, ...]:
    """ロボットごとに、ほかのいずれかのロボットと胴体の中心が min_separation_m より近い環境のマスク (num_envs,) を返す。

    Genesis の接触判定はロボットと箱の組だけを取り出しているため、ロボット同士の衝突は胴体の中心の距離で近似する。
    """
    close = [torch.zeros_like(robots_xy_m[0][:, 0], dtype=torch.bool) for _ in robots_xy_m]
    for index, own in enumerate(robots_xy_m):
        for other_index, other in enumerate(robots_xy_m):
            if other_index != index:
                close[index] |= torch.linalg.vector_norm(own - other, dim=1) < min_separation_m
    return tuple(close)


def compute_team_marl_reward_terms(inputs: TeamMarlRewardInputs, config: TeamMarlRewardConfig) -> TeamMarlRewardTerms:
    """係数を掛ける前の、報酬の各項の値を計算する。

    Returns:
        各項の値。box_progress は箱が退避領域に近づいた距離 [m]（遠ざかれば負）。box_rotation はこのステップでの
        箱のヨー角の変化の大きさ [rad]。approach_progress は自分の押す位置に近づいた距離 [m]（遠ざかれば負）。success・intended_face_contact・robot_fall・out_of_bounds・robot_collision は該当すれば 1、
        しなければ 0。control_effort は速度指令の各成分の 2 乗和。
    """
    dtype = inputs.goal_distance_m.dtype
    num_robots = len(inputs.robots_xy_m)

    def shared(values: torch.Tensor) -> torch.Tensor:
        return values.unsqueeze(0).expand(num_robots, -1)

    intended_face = torch.stack(
        [
            contact & is_pushing_toward_goal(robot_xy, inputs.box_xy_m, inputs.goal_xy_m, config.intended_face_min_cos)
            for robot_xy, contact in zip(inputs.robots_xy_m, inputs.box_contacts, strict=True)
        ]
    )
    too_close = robots_too_close(inputs.robots_xy_m, config.min_robot_separation_m)
    return TeamMarlRewardTerms(
        box_progress=shared(inputs.previous_goal_distance_m - inputs.goal_distance_m),
        success=shared(inputs.succeeded.to(dtype)),
        box_rotation=shared(torch.abs(wrap_angle_rad(inputs.box_yaw_rad - inputs.previous_box_yaw_rad))),
        approach_progress=torch.stack(inputs.previous_push_distances_m) - torch.stack(inputs.push_distances_m),
        intended_face_contact=intended_face.to(dtype),
        robot_fall=torch.stack(inputs.fallen).to(dtype),
        out_of_bounds=torch.stack(inputs.out_of_bounds).to(dtype),
        robot_collision=torch.stack(too_close).to(dtype),
        control_effort=torch.stack([torch.sum(torch.square(command), dim=1) for command in inputs.commands]),
    )
