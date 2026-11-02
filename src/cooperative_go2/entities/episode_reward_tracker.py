"""エピソードごとの報酬の内訳と終了理由の集計（学習ログ用）。"""

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from cooperative_go2.entities.gait_reward import GAIT_REWARD_TERM_NAMES
from cooperative_go2.entities.gait_termination import TerminationReason


@dataclass(frozen=True)
class EpisodeSummary:
    """1 制御ステップで終了したエピソード群の集計。値は 0 次元テンソル（GPU との同期を増やさないため）。

    Attributes:
        reward_per_second: 報酬の項名 → 終了したエピソードの「1 秒あたりの報酬」の平均。
        termination_fraction: 終了理由 → 終了したエピソードのうちその理由で終わった割合。
    """

    reward_per_second: Mapping[str, torch.Tensor]
    termination_fraction: Mapping[TerminationReason, torch.Tensor]

    def as_log(self) -> dict[str, torch.Tensor]:
        """ログに書くための、キー → 値の対応を返す。キーは rew_<項名> と termination_<理由>。"""
        rewards = {f"rew_{name}": value for name, value in self.reward_per_second.items()}
        terminations = {f"termination_{reason.name.lower()}": value for reason, value in self.termination_fraction.items()}
        return rewards | terminations


class EpisodeRewardTracker:
    """環境ごとに報酬の項を累積し、エピソードが終わった環境の分を集計して累積を戻す。"""

    def __init__(self, num_envs: int, control_dt_s: float, device: torch.device | str) -> None:
        """全環境の累積を 0 で初期化する。報酬の項は GAIT_REWARD_TERM_NAMES のすべてを扱う。

        Args:
            num_envs: 並列環境の数。
            control_dt_s: 1 制御ステップの時間 [s]。1 秒あたりの報酬の換算に使う。
            device: 累積を置くデバイス。
        """
        self._sums = {name: torch.zeros(num_envs, device=device) for name in GAIT_REWARD_TERM_NAMES}
        self._steps = torch.zeros(num_envs, dtype=torch.int64, device=device)
        self._control_dt_s = control_dt_s

    def add_step(self, terms: Mapping[str, torch.Tensor]) -> None:
        """1 制御ステップ分の報酬の項 (num_envs,) を累積する。

        Raises:
            KeyError: GAIT_REWARD_TERM_NAMES のいずれかの項が terms に無い。
        """
        for name, sums in self._sums.items():
            sums += terms[name]
        self._steps += 1

    def pop_completed_or_none(self, done: torch.Tensor, reasons: torch.Tensor) -> EpisodeSummary | None:
        """終了した環境 (done が True) の集計を返し、その環境の累積を 0 に戻す。終了した環境が無ければ None を返す。

        Args:
            done: このステップでエピソードが終わった環境のマスク (num_envs,)。
            reasons: 環境ごとの終了理由 (num_envs,)。TerminationReason の値。
        """
        if not bool(done.any()):
            return None
        episode_seconds = self._steps[done].to(torch.float32) * self._control_dt_s
        reward_per_second = {name: (sums[done] / episode_seconds).mean() for name, sums in self._sums.items()}
        finished_reasons = reasons[done]
        termination_fraction = {
            reason: (finished_reasons == int(reason)).to(torch.float32).mean()
            for reason in TerminationReason
            if reason != TerminationReason.NONE
        }
        for sums in self._sums.values():
            sums.masked_fill_(done, 0.0)
        self._steps.masked_fill_(done, 0)
        return EpisodeSummary(reward_per_second=reward_per_second, termination_fraction=termination_fraction)
