"""1 台の箱押しのエピソードの終了理由の判定。"""

from dataclasses import dataclass
from enum import IntEnum

import torch

from cooperative_go2.entities.push_config import PushArenaConfig


class PushTerminationReason(IntEnum):
    """箱押しのエピソードの終了理由。複数が同時に起きたときは値の大きい理由を採る。"""

    NONE = 0
    TIME_OUT = 1
    SUCCESS = 2
    FALL = 3
    OUT_OF_BOUNDS = 4
    SIM_DIVERGED = 5


@dataclass(frozen=True)
class PushTerminationFlags:
    """1 制御周期で起きた終了条件。各フィールドは (num_envs,) の bool マスク。

    Attributes:
        time_out: エピソードの長さに達した。
        succeeded: 成功の条件を満たした。
        fallen: ロボットが転倒した。
        out_of_bounds: 箱またはロボットがシーンの有効範囲の外に出た。
        diverged: 物理計算が発散した（NaN・異常な速度など）。
    """

    time_out: torch.Tensor
    succeeded: torch.Tensor
    fallen: torch.Tensor
    out_of_bounds: torch.Tensor
    diverged: torch.Tensor


def is_outside_arena(position_xy_m: torch.Tensor, arena: PushArenaConfig) -> torch.Tensor:
    """位置がシーンの有効範囲（境界を含む長方形）の外にある環境のマスク (num_envs,) を返す。

    Args:
        position_xy_m: 箱またはロボットの位置 (num_envs, 2)、各環境の原点から見た座標 [m]。
        arena: 有効範囲。
    """
    outside_x = torch.abs(position_xy_m[:, 0]) > arena.half_extent_x_m
    outside_y = torch.abs(position_xy_m[:, 1]) > arena.half_extent_y_m
    return outside_x | outside_y


def classify_push_terminations(flags: PushTerminationFlags) -> torch.Tensor:
    """環境ごとの終了理由 (num_envs,) を PushTerminationReason の値で返す。

    発散 > シーン外への離脱 > 転倒 > 成功 > 時間切れ の順に優先する。
    発散した環境の状態は信頼できず、ほかの判定ができないため最優先にする。
    離脱や転倒と同時に成功の条件を満たしても成功として数えないのは、成功率を甘く見積もらないため。
    """
    reasons = torch.full(
        flags.time_out.shape, int(PushTerminationReason.NONE), dtype=torch.int64, device=flags.time_out.device
    )
    for mask, reason in (
        (flags.time_out, PushTerminationReason.TIME_OUT),
        (flags.succeeded, PushTerminationReason.SUCCESS),
        (flags.fallen, PushTerminationReason.FALL),
        (flags.out_of_bounds, PushTerminationReason.OUT_OF_BOUNDS),
        (flags.diverged, PushTerminationReason.SIM_DIVERGED),
    ):
        reasons = torch.where(mask, int(reason), reasons)
    return reasons
