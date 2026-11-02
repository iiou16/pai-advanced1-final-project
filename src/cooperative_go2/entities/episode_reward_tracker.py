"""エピソードごとの報酬の内訳と終了理由の集計（学習ログ用）。歩行・箱押しなど、報酬の項と終了理由の組ごとに使う。"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum

import torch


@dataclass(frozen=True)
class EpisodeTrackingSpec:
    """集計する報酬の項と終了理由、1 ステップの時間。

    Attributes:
        term_names: 報酬の項の名前。add_step() にはこの名前をすべて渡す。
        termination_reasons: 終了理由の列挙。値 0 の要素は「終了していない」を表し、集計しない。
        step_seconds: 1 ステップの時間 [s]。1 秒あたりの報酬の換算に使う。
    """

    term_names: tuple[str, ...]
    termination_reasons: type[IntEnum]
    step_seconds: float


@dataclass(frozen=True)
class EpisodeSummary:
    """1 ステップで終了したエピソード群の集計。値は 0 次元テンソル（GPU との同期を増やさないため）。

    RSL-RL は 1 反復の間に得たこの集計を、終了したエピソードの数によらず単純に平均してログに書く。

    Attributes:
        reward_per_second: 報酬の項名 → 終了したエピソードの「1 秒あたりの報酬」の平均。
        termination_fraction: 終了理由 → 終了したエピソードのうちその理由で終わった割合。
    """

    reward_per_second: Mapping[str, torch.Tensor]
    termination_fraction: Mapping[IntEnum, torch.Tensor]

    def as_log(self) -> dict[str, torch.Tensor]:
        """ログに書くための、キー → 値の対応を返す。キーは rew_<項名> と termination_<理由>。"""
        rewards = {f"rew_{name}": value for name, value in self.reward_per_second.items()}
        terminations = {
            f"termination_{reason.name.lower()}": value for reason, value in self.termination_fraction.items()
        }
        return rewards | terminations


class EpisodeRewardTracker:
    """環境ごとに報酬の項を累積し、エピソードが終わった環境の分を集計する。"""

    def __init__(self, spec: EpisodeTrackingSpec, num_envs: int, device: torch.device | str) -> None:
        """全環境の累積を 0 で初期化する。

        Args:
            spec: 集計する報酬の項と終了理由、1 ステップの時間。
            num_envs: 並列環境の数。
            device: 累積を置くデバイス。
        """
        self._spec = spec
        self._sums = {name: torch.zeros(num_envs, device=device) for name in spec.term_names}
        self._steps = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def add_step(self, terms: Mapping[str, torch.Tensor]) -> None:
        """1 ステップ分の報酬の項 (num_envs,) を累積する。

        Raises:
            KeyError: spec.term_names のいずれかが terms に無い。
        """
        for name, sums in self._sums.items():
            sums += terms[name]
        self._steps += 1

    def summarize_completed_or_none(self, done: torch.Tensor, reasons: torch.Tensor) -> EpisodeSummary | None:
        """終了した環境 (done が True) のエピソードを集計する。終了した環境が無ければ None を返す。累積は変えない。

        Args:
            done: このステップでエピソードが終わった環境のマスク (num_envs,)。
            reasons: 環境ごとの終了理由 (num_envs,)。spec.termination_reasons の値。
        """
        if not bool(done.any()):
            return None
        episode_seconds = self._steps[done].to(torch.float32) * self._spec.step_seconds
        reward_per_second = {name: (sums[done] / episode_seconds).mean() for name, sums in self._sums.items()}
        finished_reasons = reasons[done]
        termination_fraction = {
            reason: (finished_reasons == int(reason)).to(torch.float32).mean()
            for reason in self._spec.termination_reasons
            if int(reason) != 0
        }
        return EpisodeSummary(reward_per_second=reward_per_second, termination_fraction=termination_fraction)

    def reset_where(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の累積を 0 に戻す。環境をリセットするたびに呼ぶ。"""
        for sums in self._sums.values():
            sums.masked_fill_(mask, 0.0)
        self._steps.masked_fill_(mask, 0)
