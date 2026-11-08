"""skrl の IPPO・MAPPO とネットワーク構成の設定（configs/push_multi.yaml の skrl 節）と、比べる学習方式。

学習アルゴリズム固有の語彙はこの層に閉じ込め、entities / use_cases には渡さない。
"""

import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError


class InvalidSkrlConfigError(InvalidConfigError):
    """skrl の設定値が、skrl の受け付ける値や値域を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidSkrlConfigError)


class MarlAlgorithm(Enum):
    """協調の学習方式。

    - IPPO: 各ロボットの Critic が、自分の Actor と同じ観測だけを使う（独立に学習する）。
    - MAPPO: 各ロボットの Critic が、学習時にだけ使える全体の真の状態を使う（集中 Critic・分散 Actor）。

    skrl の IPPO と MAPPO は、Critic に渡す入力（観測か状態か）を除いて同じ更新則なので、比べる条件の違いは
    Critic の入力だけになる。
    """

    IPPO = "ippo"
    MAPPO = "mappo"


@dataclass(frozen=True)
class MarlVariant:
    """比べる学習の組み合わせ。

    Attributes:
        algorithm: 学習方式。
        shared_parameters: 2 台で Actor と Critic のパラメータを共有するか。共有しなければロボットごとに別々に学習する。
    """

    algorithm: MarlAlgorithm
    shared_parameters: bool

    @property
    def name(self) -> str:
        """記録に使う名前（例: mappo-shared）。"""
        return f"{self.algorithm.value}-{'shared' if self.shared_parameters else 'separate'}"


def _require_hidden_dims(name: str, dims: tuple[int, ...]) -> None:
    if not dims or any(dim <= 0 for dim in dims):
        raise InvalidSkrlConfigError(f"{name} は正の整数を 1 つ以上並べる必要がある: {dims}")


@dataclass(frozen=True)
class SkrlMarlPpoConfig:
    """skrl の IPPO・MAPPO・ネットワーク・保存間隔の設定。

    PPO の値の意味は skrl の IPPO_CFG・MAPPO_CFG と同じ。学習率は KL ダイバージェンスが desired_kl に近づくよう
    skrl の KLAdaptiveLR で調整する（1 台の箱押しの RSL-RL の schedule: adaptive と同じ考え方）。
    1 回の更新（反復）は rollouts ステップ分を集めてから行い、checkpoint_interval_updates 回の更新ごとに保存する。

    Attributes:
        rollouts: 1 回の更新の前に、各環境で集める高レベル方策のステップ数。
        learning_epochs: 1 回の更新で、集めたデータを使う回数。
        mini_batches: 1 エポックのミニバッチの数。
        discount_factor: 割引率。[0, 1]。
        gae_lambda: GAE の λ。[0, 1]。
        learning_rate: 学習率の初期値。
        desired_kl: 学習率を調整する目標の KL ダイバージェンス。
        ratio_clip: 方策の比のクリップ幅。
        value_clip: 価値の予測のクリップ幅。
        entropy_loss_scale: エントロピーの損失の係数。0 以上。
        value_loss_scale: 価値の損失の係数。
        grad_norm_clip: 勾配のノルムの上限。
        actor_hidden_dims: Actor の中間層の幅。
        critic_hidden_dims: Critic の中間層の幅。
        init_noise_std: 行動の標準偏差の初期値。
        checkpoint_interval_updates: チェックポイントを保存する更新の間隔。
    """

    rollouts: int
    learning_epochs: int
    mini_batches: int
    discount_factor: float
    gae_lambda: float
    learning_rate: float
    desired_kl: float
    ratio_clip: float
    value_clip: float
    entropy_loss_scale: float
    value_loss_scale: float
    grad_norm_clip: float
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    init_noise_std: float
    checkpoint_interval_updates: int

    def __post_init__(self) -> None:
        positive_fields = (
            "rollouts", "learning_epochs", "mini_batches", "learning_rate", "desired_kl", "ratio_clip",
            "value_clip", "value_loss_scale", "grad_norm_clip", "init_noise_std", "checkpoint_interval_updates",
        )  # fmt: skip
        for name in positive_fields:
            _check.positive(f"skrl.{name}", getattr(self, name))
        _check.unit_interval("skrl.discount_factor", self.discount_factor)
        _check.unit_interval("skrl.gae_lambda", self.gae_lambda)
        _check.non_negative("skrl.entropy_loss_scale", self.entropy_loss_scale)
        _require_hidden_dims("skrl.actor_hidden_dims", self.actor_hidden_dims)
        _require_hidden_dims("skrl.critic_hidden_dims", self.critic_hidden_dims)

    @property
    def init_log_std(self) -> float:
        """行動の標準偏差の初期値の対数。"""
        return math.log(self.init_noise_std)

    def to_agent_cfg(self, run_dir: Path, agent_names: tuple[str, ...]) -> dict[str, object]:
        """IPPO・MAPPO の設定の辞書を作る。ログとチェックポイントは run_dir に書く。

        学習率の調整器（KLAdaptiveLR）のクラスは skrl 側で決まるので、呼び出し側で learning_rate_scheduler に入れる。
        辞書を値に持つ設定は、エージェントの名前をキーにして渡す。skrl 2.1 は辞書の値をエージェントごとの指定と
        みなし、キーがエージェントの名前でない辞書（既定の空の辞書を含む）を受け付けないため。
        """
        per_agent_dict_fields = {
            "learning_rate_scheduler_kwargs": {"kl_threshold": self.desired_kl},
            "observation_preprocessor_kwargs": {},
            "state_preprocessor_kwargs": {},
            "value_preprocessor_kwargs": {},
        }
        per_agent = {key: {name: dict(value) for name in agent_names} for key, value in per_agent_dict_fields.items()}
        return per_agent | {
            "rollouts": self.rollouts,
            "learning_epochs": self.learning_epochs,
            "mini_batches": self.mini_batches,
            "discount_factor": self.discount_factor,
            "gae_lambda": self.gae_lambda,
            "learning_rate": self.learning_rate,
            "ratio_clip": self.ratio_clip,
            "value_clip": self.value_clip,
            "entropy_loss_scale": self.entropy_loss_scale,
            "value_loss_scale": self.value_loss_scale,
            "grad_norm_clip": self.grad_norm_clip,
            # 時間切れで終えたエピソードは、続きの価値を足して学習する（時間切れは失敗ではないため）
            "time_limit_bootstrap": True,
            "experiment": {
                "directory": str(run_dir.parent),
                "experiment_name": run_dir.name,
                "write_interval": self.rollouts,
                "checkpoint_interval": self.rollouts * self.checkpoint_interval_updates,
            },
        }
