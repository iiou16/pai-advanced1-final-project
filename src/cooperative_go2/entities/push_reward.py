"""1 台の箱押しの高レベル方策の報酬の計算。

箱が目標に近づいた量を主な報酬にし、ロボット自身の移動量だけでは報酬が出ないようにする。
成功・意図した面での接触を加点し、箱の回転・転倒・危険な衝突・過大な速度指令を減点する。
すべて並列環境のバッチ (num_envs, ...) のテンソルのまま計算する。
"""

from dataclasses import dataclass, fields

import torch

from cooperative_go2.entities.planar_frame import wrap_angle_rad
from cooperative_go2.entities.push_config import PushRewardConfig, PushRewardScales
from cooperative_go2.entities.push_termination import PushTerminationReason


@dataclass(frozen=True)
class PushRewardInputs:
    """報酬の計算に使う、高レベル方策の 1 制御周期分の材料。位置はすべてワールド座標系の水平成分 [m]。

    Attributes:
        previous_goal_distance_m: 前のステップでの箱と目標の距離 (num_envs,) [m]。
        goal_distance_m: このステップでの箱と目標の距離 (num_envs,) [m]。
        termination_reasons: classify_push_terminations で決めた終了理由 (num_envs,)。PushTerminationReason の値。
            成功の報酬は理由が SUCCESS の環境だけに与える。成功の条件を満たしても、同じステップで転倒などが
            優先された環境には与えない（終了理由と報酬の内訳を食い違わせないため）。
        box_contact: ロボットが箱に接触している環境のマスク (num_envs,)。
        robot_xy_m: ロボットの胴体の位置 (num_envs, 2)。
        box_xy_m: 箱の中心 (num_envs, 2)。
        goal_xy_m: 目標位置 (num_envs, 2)。
        previous_box_yaw_rad: 前のステップでの箱のヨー角 (num_envs,) [rad]。
        box_yaw_rad: このステップでの箱のヨー角 (num_envs,) [rad]。
        fallen: ロボットが転倒した環境のマスク (num_envs,)。
        unsafe_collision: ロボットが箱以外（壁・他のロボットなど）と衝突した環境のマスク (num_envs,)。
        commands: このステップで出した速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
    """

    previous_goal_distance_m: torch.Tensor
    goal_distance_m: torch.Tensor
    termination_reasons: torch.Tensor
    box_contact: torch.Tensor
    robot_xy_m: torch.Tensor
    box_xy_m: torch.Tensor
    goal_xy_m: torch.Tensor
    previous_box_yaw_rad: torch.Tensor
    box_yaw_rad: torch.Tensor
    fallen: torch.Tensor
    unsafe_collision: torch.Tensor
    commands: torch.Tensor


@dataclass(frozen=True)
class PushRewardTerms:
    """報酬の項ごとの値。各フィールドは (num_envs,) のテンソル。フィールド名は PushRewardScales と一致する。"""

    box_progress: torch.Tensor
    success: torch.Tensor
    intended_face_contact: torch.Tensor
    box_rotation: torch.Tensor
    robot_fall: torch.Tensor
    unsafe_collision: torch.Tensor
    control_effort: torch.Tensor

    def weighted(self, scales: PushRewardScales) -> "PushRewardTerms":
        """各項に係数を掛けた項を返す。

        係数は高レベル方策の 1 ステップあたりの重みで、制御周期は掛けない。
        box_progress と box_rotation は 1 ステップの間の変化量なので、周期を変えても 1 エピソードの合計は変わらない。
        一方、intended_face_contact と control_effort は毎ステップ加算されるため、周期を変えたら係数を調整し直す。
        """
        return PushRewardTerms(
            **{field.name: getattr(self, field.name) * getattr(scales, field.name) for field in fields(self)}
        )

    def total(self) -> torch.Tensor:
        """全項の和 (num_envs,) を返す。"""
        return torch.stack([getattr(self, field.name) for field in fields(self)]).sum(dim=0)

    def named(self) -> dict[str, torch.Tensor]:
        """ログ出力用に、項の名前 → 値の対応を返す。"""
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def zeroed_where(self, mask: torch.Tensor) -> "PushRewardTerms":
        """mask (num_envs,) が True の環境の値を、すべての項で 0 にした項を返す。"""
        return PushRewardTerms(**{name: torch.where(mask, 0.0, value) for name, value in self.named().items()})


PUSH_REWARD_TERM_NAMES: tuple[str, ...] = tuple(field.name for field in fields(PushRewardTerms))


def is_pushing_toward_goal(
    robot_xy_m: torch.Tensor, box_xy_m: torch.Tensor, goal_xy_m: torch.Tensor, min_cos: float
) -> torch.Tensor:
    """ロボットが箱の、目標と反対側（押せば箱が目標へ進む側）にいる環境のマスク (num_envs,) を返す。

    「ロボット → 箱」の向きと「箱 → 目標」の向きのなす角の余弦が min_cos 以上なら True。
    箱が目標の中心にあるなどで向きが定まらない環境は、押す方向が無いので False とする。

    Args:
        robot_xy_m: ロボットの胴体の位置 (num_envs, 2) [m]。
        box_xy_m: 箱の中心 (num_envs, 2) [m]。
        goal_xy_m: 目標位置 (num_envs, 2) [m]。
        min_cos: 余弦の下限。-1 以上 1 以下。
    """
    robot_to_box = box_xy_m - robot_xy_m
    box_to_goal = goal_xy_m - box_xy_m
    norms = torch.linalg.vector_norm(robot_to_box, dim=1) * torch.linalg.vector_norm(box_to_goal, dim=1)
    defined = norms > 0
    cos_angle = torch.sum(robot_to_box * box_to_goal, dim=1) / torch.where(defined, norms, torch.ones_like(norms))
    return defined & (cos_angle >= min_cos)


def compute_push_reward_terms(inputs: PushRewardInputs, config: PushRewardConfig) -> PushRewardTerms:
    """係数を掛ける前の、報酬の各項の値を計算する。

    Args:
        inputs: 1 制御周期分の材料。
        config: 意図した面とみなす角度の条件。係数（scales）はここでは使わない。

    Returns:
        各項の値。box_progress は箱が目標に近づいた距離 [m]（遠ざかれば負）。
        success・intended_face_contact・robot_fall・unsafe_collision は該当すれば 1、しなければ 0。
        box_rotation はこのステップでの箱のヨー角の変化の大きさ [rad]。
        control_effort は速度指令の各成分の 2 乗和。
    """
    dtype = inputs.goal_distance_m.dtype
    aligned = is_pushing_toward_goal(inputs.robot_xy_m, inputs.box_xy_m, inputs.goal_xy_m, config.intended_face_min_cos)
    return PushRewardTerms(
        box_progress=inputs.previous_goal_distance_m - inputs.goal_distance_m,
        success=(inputs.termination_reasons == int(PushTerminationReason.SUCCESS)).to(dtype),
        intended_face_contact=(inputs.box_contact & aligned).to(dtype),
        box_rotation=torch.abs(wrap_angle_rad(inputs.box_yaw_rad - inputs.previous_box_yaw_rad)),
        robot_fall=inputs.fallen.to(dtype),
        unsafe_collision=inputs.unsafe_collision.to(dtype),
        control_effort=torch.sum(torch.square(inputs.commands), dim=1),
    )
