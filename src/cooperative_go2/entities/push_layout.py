"""エピソードの始めの箱と目標の配置を、シードを固定した乱数で引く。"""

import math
from dataclasses import dataclass

import torch

from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.entities.push_task_config import PushResetConfig


@dataclass(frozen=True)
class PushLayout:
    """並列環境ごとの箱と目標の配置。座標は各環境の原点から見た水平面の位置 [m]。

    Attributes:
        box_xy_m: 箱の中心 (num_envs, 2)。
        goal_xy_m: 目標領域の中心 (num_envs, 2)。
    """

    box_xy_m: torch.Tensor
    goal_xy_m: torch.Tensor


def _uniform(value_range: ValueRange, num_envs: int, generator: torch.Generator) -> torch.Tensor:
    unit = torch.rand(num_envs, generator=generator, device=generator.device)
    return value_range.low + (value_range.high - value_range.low) * unit


def sample_push_layout(
    base_box_xy_m: tuple[float, ...], reset: PushResetConfig, num_envs: int, generator: torch.Generator
) -> PushLayout:
    """箱を base_box_xy_m の周りにずらして置き、目標を箱から見た距離と方向で決める。

    Args:
        base_box_xy_m: 箱を置く基準の位置 (x, y) [m]（シーンの設定の箱の初期位置）。
        reset: ずらす量と、目標の距離・方向の範囲。
        num_envs: 並列環境の数。
        generator: 配置を引く乱数生成器。テンソルはこの生成器のデバイスに置く。
    """
    device = generator.device
    jitter = (torch.rand((num_envs, 2), generator=generator, device=device) * 2 - 1) * reset.box_xy_jitter_m
    box_xy = torch.tensor(base_box_xy_m, device=device) + jitter
    distance = _uniform(reset.goal_distance_m, num_envs, generator)
    direction = _uniform(reset.goal_direction_deg, num_envs, generator) * (math.pi / 180.0)
    goal_xy = box_xy + distance.unsqueeze(1) * torch.stack((torch.cos(direction), torch.sin(direction)), dim=1)
    return PushLayout(box_xy_m=box_xy, goal_xy_m=goal_xy)
