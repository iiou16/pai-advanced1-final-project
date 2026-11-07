"""2 台の協調学習の、高レベル方策の 1 ステップごとの成功と終了の判定。

エピソードの終わりは 2 台で共有する。どちらかが転倒・シーン外への離脱をしたら、2 台ともそのエピソードを終える。
規則ベースの協調でも、片方が失敗（FAILED）したら試行を終えて失敗に数えるため、比べる条件をそろえる。
終了理由は 1 台の箱押しと同じ列挙（PushTerminationReason）と優先順位を使う。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.push_config import PushArenaConfig
from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region
from cooperative_go2.entities.push_termination import PushTerminationFlags, classify_push_terminations, is_outside_arena


@dataclass(frozen=True)
class TeamJudgeInputs:
    """高レベル方策の 1 ステップを終えた時点の状態。テンソルは並列環境のバッチ。

    Attributes:
        robots_xy_m: ロボットごとの胴体の位置 (num_envs, 2) [m]。
        box_xy_m: 箱の中心 (num_envs, 2) [m]。
        fallen: ロボットごとの、このステップのどこかで転倒の傾きを超えた環境のマスク (num_envs,)。
        episode_steps: このステップを含めた、エピソードの高レベル方策のステップ数 (num_envs,)。
        hold_steps: 前のステップまでに箱が退避領域に入ったまま続いたステップ数 (num_envs,)。
        diverged: このステップの物理ステップのどこかで物理計算が発散した環境のマスク (num_envs,)。
    """

    robots_xy_m: tuple[torch.Tensor, ...]
    box_xy_m: torch.Tensor
    fallen: tuple[torch.Tensor, ...]
    episode_steps: torch.Tensor
    hold_steps: torch.Tensor
    diverged: torch.Tensor


@dataclass(frozen=True)
class TeamJudgeRules:
    """判定の条件。

    Attributes:
        retreat_xy_m: 退避領域の中心 (x, y) [m]。
        retreat_radius_m: 退避領域の半径 [m]。
        required_hold_steps: 箱の中心が退避領域に入ったまま続けたら成功とするステップ数。
        max_episode_steps: エピソードの長さ（高レベル方策のステップ数）。
        arena: シーンの有効範囲。
    """

    retreat_xy_m: tuple[float, float]
    retreat_radius_m: float
    required_hold_steps: int
    max_episode_steps: int
    arena: PushArenaConfig


@dataclass(frozen=True)
class TeamJudgement:
    """1 ステップの判定の結果。

    Attributes:
        hold_steps: このステップを含めて箱が退避領域に入ったまま続いたステップ数 (num_envs,)。
        flags: 起きた終了条件。fallen・out_of_bounds は、いずれかのロボットで起きたら True。
        reasons: 環境ごとの終了理由 (num_envs,)。PushTerminationReason の値で、終わっていなければ NONE。
    """

    hold_steps: torch.Tensor
    flags: PushTerminationFlags
    reasons: torch.Tensor


def judge_team_step(inputs: TeamJudgeInputs, rules: TeamJudgeRules) -> TeamJudgement:
    """箱が退避領域に入ったまま続いたステップ数を進め、2 台で共有する終了条件と終了理由を判定する。"""
    retreat = inputs.box_xy_m.new_tensor(rules.retreat_xy_m).expand_as(inputs.box_xy_m)
    in_retreat = is_in_goal_region(inputs.box_xy_m, retreat, rules.retreat_radius_m)
    hold_steps = advance_goal_hold_steps(inputs.hold_steps, in_retreat)
    out_of_bounds = is_outside_arena(inputs.box_xy_m, rules.arena)
    fallen = torch.zeros_like(out_of_bounds)
    for robot_xy, robot_fallen in zip(inputs.robots_xy_m, inputs.fallen, strict=True):
        out_of_bounds = out_of_bounds | is_outside_arena(robot_xy, rules.arena)
        fallen = fallen | robot_fallen
    flags = PushTerminationFlags(
        time_out=inputs.episode_steps >= rules.max_episode_steps,
        succeeded=has_held_goal(hold_steps, rules.required_hold_steps),
        fallen=fallen,
        out_of_bounds=out_of_bounds,
        diverged=inputs.diverged,
    )
    return TeamJudgement(hold_steps=hold_steps, flags=flags, reasons=classify_push_terminations(flags))
