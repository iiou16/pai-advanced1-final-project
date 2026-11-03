"""RSL-RL の PPO とネットワーク構成の設定（configs/gait.yaml の rsl_rl 節）。

学習アルゴリズム固有の語彙はこの層に閉じ込め、entities / use_cases には渡さない。
"""

from dataclasses import dataclass

from rsl_rl.utils import resolve_nn_activation

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError

# RSL-RL の PPO が学習率を変える方式。adaptive は KL ダイバージェンスが desired_kl に近づくよう調整し、fixed は変えない
ALLOWED_SCHEDULES = ("adaptive", "fixed")
TENSORBOARD_LOGGER = "tensorboard"
# 環境が返す観測（TensorDict）のうち、方策の入力にするグループの名前
POLICY_OBSERVATION_GROUP = "policy"
# 方策と価値関数のネットワークは観測を共有する（非対称な観測を使わない）
OBSERVATION_GROUPS = {"actor": [POLICY_OBSERVATION_GROUP], "critic": [POLICY_OBSERVATION_GROUP]}


class InvalidRslRlConfigError(InvalidConfigError):
    """RSL-RL の設定値が、RSL-RL の受け付ける値や値域を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidRslRlConfigError)


def _require_hidden_dims(name: str, dims: tuple[int, ...]) -> None:
    if not dims or any(dim <= 0 for dim in dims):
        raise InvalidRslRlConfigError(f"{name} は正の整数を 1 つ以上並べる必要がある: {dims}")


@dataclass(frozen=True)
class RslRlPpoConfig:
    """RSL-RL の OnPolicyRunner に渡す PPO・ネットワーク・保存間隔の設定。

    各値の意味は RSL-RL の PPO・MLPModel の引数と同じ。save_interval はチェックポイントを保存する反復の間隔で、
    反復番号（0 から数える）がその倍数のときと、最後の反復で保存される。
    """

    clip_param: float
    desired_kl: float
    entropy_coef: float
    gamma: float
    lam: float
    learning_rate: float
    max_grad_norm: float
    num_learning_epochs: int
    num_mini_batches: int
    schedule: str
    use_clipped_value_loss: bool
    value_loss_coef: float
    num_steps_per_env: int
    actor_hidden_dims: tuple[int, ...]
    critic_hidden_dims: tuple[int, ...]
    activation: str
    init_noise_std: float
    save_interval: int

    def __post_init__(self) -> None:
        positive_fields = (
            "clip_param", "desired_kl", "learning_rate", "max_grad_norm", "num_learning_epochs",
            "num_mini_batches", "value_loss_coef", "num_steps_per_env", "init_noise_std", "save_interval",
        )  # fmt: skip
        for name in positive_fields:
            _check.positive(f"rsl_rl.{name}", getattr(self, name))
        _check.unit_interval("rsl_rl.gamma", self.gamma)
        _check.unit_interval("rsl_rl.lam", self.lam)
        _check.non_negative("rsl_rl.entropy_coef", self.entropy_coef)
        if self.schedule not in ALLOWED_SCHEDULES:
            raise InvalidRslRlConfigError(f"rsl_rl.schedule は {ALLOWED_SCHEDULES} のいずれか: {self.schedule!r}")
        _require_hidden_dims("rsl_rl.actor_hidden_dims", self.actor_hidden_dims)
        _require_hidden_dims("rsl_rl.critic_hidden_dims", self.critic_hidden_dims)
        try:
            resolve_nn_activation(self.activation)
        except ValueError as error:
            raise InvalidRslRlConfigError(f"rsl_rl.activation が RSL-RL に無い: {self.activation!r}") from error

    def to_runner_cfg(self, run_name: str) -> dict[str, object]:
        """OnPolicyRunner に渡す学習設定の辞書を作る。run_name はログに記録する実行の名前。"""
        return {
            "algorithm": {
                "class_name": "PPO",
                "clip_param": self.clip_param,
                "desired_kl": self.desired_kl,
                "entropy_coef": self.entropy_coef,
                "gamma": self.gamma,
                "lam": self.lam,
                "learning_rate": self.learning_rate,
                "max_grad_norm": self.max_grad_norm,
                "num_learning_epochs": self.num_learning_epochs,
                "num_mini_batches": self.num_mini_batches,
                "schedule": self.schedule,
                "use_clipped_value_loss": self.use_clipped_value_loss,
                "value_loss_coef": self.value_loss_coef,
            },
            "actor": {
                "class_name": "MLPModel",
                "hidden_dims": list(self.actor_hidden_dims),
                "activation": self.activation,
                "distribution_cfg": {
                    "class_name": "GaussianDistribution",
                    "init_std": self.init_noise_std,
                    "std_type": "scalar",
                },
            },
            "critic": {
                "class_name": "MLPModel",
                "hidden_dims": list(self.critic_hidden_dims),
                "activation": self.activation,
            },
            # RSL-RL は不足した観測グループを補うために渡した辞書を書き換えうるので、毎回新しく作る
            "obs_groups": {group: list(sources) for group, sources in OBSERVATION_GROUPS.items()},
            "num_steps_per_env": self.num_steps_per_env,
            "save_interval": self.save_interval,
            "run_name": run_name,
            "logger": TENSORBOARD_LOGGER,
        }
