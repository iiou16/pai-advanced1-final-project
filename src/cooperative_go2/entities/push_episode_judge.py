"""1 台の箱押しの、高レベル方策の 1 ステップごとの成功と終了の判定。

学習環境（Go2PushEnv）と ROS 2 のシミュレーションノードが同じ判定を使い、評価の条件をそろえる。
"""

from dataclasses import dataclass

import torch

from cooperative_go2.entities.gait_termination import exceeds_tilt_limit
from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.push_termination import (
    PushTerminationFlags,
    classify_push_terminations,
    is_outside_arena,
)


@dataclass(frozen=True)
class PushJudgeInputs:
    """高レベル方策の 1 ステップを終えた時点の状態。テンソルは並列環境のバッチ。

    Attributes:
        robot_xy_m: Go2 の胴体の位置 (num_envs, 2) [m]。
        box_xy_m: 箱の中心 (num_envs, 2) [m]。
        goal_xy_m: 目標領域の中心 (num_envs, 2) [m]。
        base_euler_deg: 初期姿勢から見た胴体のロール・ピッチ・ヨー (num_envs, 3) [deg]。
        episode_steps: このステップを含めた、エピソードの高レベル方策のステップ数 (num_envs,)。
        hold_steps: 前のステップまでに箱が目標領域に入ったまま続いたステップ数 (num_envs,)。
        diverged: このステップの物理ステップのどこかで物理計算が発散した環境のマスク (num_envs,)。
    """

    robot_xy_m: torch.Tensor
    box_xy_m: torch.Tensor
    goal_xy_m: torch.Tensor
    base_euler_deg: torch.Tensor
    episode_steps: torch.Tensor
    hold_steps: torch.Tensor
    diverged: torch.Tensor


@dataclass(frozen=True)
class PushJudgement:
    """1 ステップの判定の結果。

    Attributes:
        hold_steps: このステップを含めて箱が目標領域に入ったまま続いたステップ数 (num_envs,)。
        flags: 起きた終了条件。
        reasons: 環境ごとの終了理由 (num_envs,)。PushTerminationReason の値で、終わっていなければ NONE。
    """

    hold_steps: torch.Tensor
    flags: PushTerminationFlags
    reasons: torch.Tensor


def judge_push_step(inputs: PushJudgeInputs, task: PushTaskConfig) -> PushJudgement:
    """箱が目標領域に入ったまま続いたステップ数を進め、終了条件と終了理由を判定する。"""
    in_goal = is_in_goal_region(inputs.box_xy_m, inputs.goal_xy_m, task.success.goal_radius_m)
    hold_steps = advance_goal_hold_steps(inputs.hold_steps, in_goal)
    episode = task.episode
    out_of_bounds = is_outside_arena(inputs.robot_xy_m, task.arena) | is_outside_arena(inputs.box_xy_m, task.arena)
    flags = PushTerminationFlags(
        time_out=inputs.episode_steps >= episode.max_episode_length_steps(),
        succeeded=has_held_goal(hold_steps, task.required_hold_steps()),
        fallen=exceeds_tilt_limit(inputs.base_euler_deg, episode.fall_roll_deg, episode.fall_pitch_deg),
        out_of_bounds=out_of_bounds,
        diverged=inputs.diverged,
    )
    return PushJudgement(hold_steps=hold_steps, flags=flags, reasons=classify_push_terminations(flags))
