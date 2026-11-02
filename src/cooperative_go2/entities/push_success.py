"""箱押しの成功判定。箱の中心が目標領域に入り、その状態を所定のステップ数だけ続けたら成功とする。

判定に使う「目標領域に入ってから続いたステップ数」は、状態を持つオブジェクトにせず、
前のステップの値から次の値を計算する純粋な関数で扱う。リセットや並列環境の扱いを呼び出し元に任せるため。
"""

import torch


def is_in_goal_region(box_xy_m: torch.Tensor, goal_xy_m: torch.Tensor, goal_radius_m: float) -> torch.Tensor:
    """箱の中心が目標領域（目標位置を中心とする半径 goal_radius_m の円、境界を含む）にある環境のマスクを返す。

    Args:
        box_xy_m: 箱の中心 (num_envs, 2)、ワールド座標系 [m]。
        goal_xy_m: 目標領域の中心 (num_envs, 2)、ワールド座標系 [m]。
        goal_radius_m: 目標領域の半径 [m]。

    Returns:
        (num_envs,) の bool マスク。
    """
    return torch.linalg.vector_norm(box_xy_m - goal_xy_m, dim=1) <= goal_radius_m


def advance_goal_hold_steps(hold_steps: torch.Tensor, in_goal: torch.Tensor) -> torch.Tensor:
    """1 ステップ進めた後の「目標領域に入ってから続いたステップ数」を返す。領域を出た環境は 0 に戻る。

    Args:
        hold_steps: 前のステップまでに続いたステップ数 (num_envs,)。整数型。エピソードの最初は 0。
        in_goal: このステップで箱が目標領域にある環境のマスク (num_envs,)。
    """
    return torch.where(in_goal, hold_steps + 1, torch.zeros_like(hold_steps))


def has_held_goal(hold_steps: torch.Tensor, required_hold_steps: int) -> torch.Tensor:
    """目標領域に入った状態を required_hold_steps 以上続けた（成功した）環境のマスク (num_envs,) を返す。

    Args:
        hold_steps: 目標領域に入ってから続いたステップ数 (num_envs,)。
        required_hold_steps: 成功に必要なステップ数。PushSuccessConfig.required_hold_steps で求める。
    """
    return hold_steps >= required_hold_steps
