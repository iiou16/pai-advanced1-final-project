"""歩行のエピソードの終了理由の判定。"""

from enum import IntEnum

import torch

ROLL_INDEX = 0
PITCH_INDEX = 1


class TerminationReason(IntEnum):
    """エピソードの終了理由。複数が同時に起きたときは値の大きい理由を採る。"""

    NONE = 0
    TIME_OUT = 1
    FALL = 2
    SIM_DIVERGED = 3


def exceeds_tilt_limit(base_euler_deg: torch.Tensor, roll_limit_deg: float, pitch_limit_deg: float) -> torch.Tensor:
    """胴体の傾きが上限を超えた（転倒した）環境を True とするマスク (num_envs,) を返す。

    Args:
        base_euler_deg: 初期姿勢から見た胴体のロール・ピッチ・ヨー (num_envs, 3) [deg]。
        roll_limit_deg: ロールの絶対値の上限 [deg]。
        pitch_limit_deg: ピッチの絶対値の上限 [deg]。
    """
    roll_exceeded = torch.abs(base_euler_deg[:, ROLL_INDEX]) > roll_limit_deg
    pitch_exceeded = torch.abs(base_euler_deg[:, PITCH_INDEX]) > pitch_limit_deg
    return roll_exceeded | pitch_exceeded


def classify_terminations(time_out: torch.Tensor, fallen: torch.Tensor, diverged: torch.Tensor) -> torch.Tensor:
    """環境ごとの終了理由 (num_envs,) を TerminationReason の値で返す。

    発散 > 転倒 > 時間切れ の順に優先する。発散した環境の状態は信頼できず転倒を判定できないため発散を最優先にし、
    転倒は時間切れと重なっても失敗として記録する（時間切れは価値の推定で打ち切りとして扱われるため区別する）。

    Args:
        time_out: エピソードの長さ（ステップ数）を超えた環境のマスク (num_envs,)。
        fallen: 転倒した環境のマスク (num_envs,)。
        diverged: 物理計算が発散した環境のマスク (num_envs,)。
    """
    reasons = torch.full(time_out.shape, int(TerminationReason.NONE), dtype=torch.int64, device=time_out.device)
    for mask, reason in (
        (time_out, TerminationReason.TIME_OUT),
        (fallen, TerminationReason.FALL),
        (diverged, TerminationReason.SIM_DIVERGED),
    ):
        reasons = torch.where(mask, int(reason), reasons)
    return reasons
