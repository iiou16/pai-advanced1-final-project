"""RSL-RL で学習したチェックポイントを、推論用の方策として読み込む。

シミュレータ（Genesis）を読み込まない。ROS 2 の推論ノードがシミュレーションを持たずに方策を使えるようにするため。
"""

import pickle
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_DIM
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP, RslRlPpoConfig


class CheckpointNotFoundError(FileNotFoundError):
    """指定したチェックポイントのファイルが無いことを表す。"""


class CheckpointLoadError(RuntimeError):
    """チェックポイントを読み込めない（ファイルが壊れている、ネットワークの構成が設定と合わないなど）ことを表す。"""


class PolicySpecEnvironment:
    """シミュレータを使わずに OnPolicyRunner を組み立てるための、観測と行動の次元だけを持つ環境。

    推論専用で、step() は呼べない。OnPolicyRunner は構築時に観測の形を 1 回読むだけで、推論では環境を使わない。
    """

    def __init__(self, observation_dim: int, num_actions: int, device: torch.device | str) -> None:
        """観測 observation_dim 次元・行動 num_actions 次元の、観測を device に置く 1 環境分の仕様を作る。"""
        self.num_envs = 1
        self.num_actions = num_actions
        self._observation_dim = observation_dim
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.device = device
        self.cfg: dict[str, object] = {}

    def get_observations(self) -> TensorDict:
        """形だけを決めるための 0 の観測を返す。"""
        observation = torch.zeros((self.num_envs, self._observation_dim), device=self.device)
        return TensorDict({POLICY_OBSERVATION_GROUP: observation}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        """推論専用の環境は進められないので、常に例外を送出する。"""
        raise RuntimeError("PolicySpecEnvironment は推論用の方策を組み立てるためだけの環境で、step() できない")


class RslRlPolicy:
    """RSL-RL の方策ネットワークを、観測テンソル → 行動テンソルの関数として包む。

    出力は決定的（分布の平均）。パラメータは勾配を計算しない設定にして固定する。
    """

    def __init__(self, policy: torch.nn.Module) -> None:
        """OnPolicyRunner.get_inference_policy() が返した推論用の方策ネットワークを受け取り、パラメータを固定する。"""
        self._policy = policy.requires_grad_(False)

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """観測 (num_envs, 観測の次元) から行動 (num_envs, 行動の次元) を返す。"""
        observations = TensorDict({POLICY_OBSERVATION_GROUP: observation}, batch_size=[observation.shape[0]])
        return self._policy(observations)


def load_rsl_rl_gait_policy(
    checkpoint_path: Path, rsl_rl_config: RslRlPpoConfig, device: torch.device | str
) -> RslRlPolicy:
    """学習済みの歩行方策のチェックポイントを読み込み、推論用の方策を返す。シミュレータは使わない。

    Raises:
        CheckpointNotFoundError: checkpoint_path にファイルが無い。
        CheckpointLoadError: ファイルを読み込めない、またはネットワークの構成が設定と合わない。
    """
    spec = PolicySpecEnvironment(GAIT_OBSERVATION_DIM, NUM_GO2_JOINTS, device)
    return load_rsl_rl_policy(checkpoint_path, rsl_rl_config, spec)


def load_rsl_rl_policy(
    checkpoint_path: Path, rsl_rl_config: RslRlPpoConfig, spec: PolicySpecEnvironment
) -> RslRlPolicy:
    """学習済みのチェックポイントを読み込み、推論用の方策を返す。シミュレータは使わない。

    Args:
        checkpoint_path: model_<反復>.pt のパス。
        rsl_rl_config: チェックポイントを学習したときの RSL-RL の設定。ネットワークの構成を合わせるために使う。
        spec: 方策の観測と行動の次元、方策を置くデバイス。

    Raises:
        CheckpointNotFoundError: checkpoint_path にファイルが無い。
        CheckpointLoadError: ファイルを読み込めない、またはネットワークの構成が設定と合わない。
    """
    if not checkpoint_path.is_file():
        raise CheckpointNotFoundError(f"チェックポイントが無い: {checkpoint_path}")
    runner_cfg = rsl_rl_config.to_runner_cfg(checkpoint_path.parent.name)
    device = spec.device
    runner = OnPolicyRunner(spec, runner_cfg, None, device=str(device))
    try:
        runner.load(str(checkpoint_path), map_location=str(device))
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        # メモリ不足と GPU の実行時エラーは RuntimeError の一種だが、チェックポイントの不良ではないのでそのまま伝える
        raise
    except (pickle.UnpicklingError, EOFError, KeyError, IndexError, TypeError, RuntimeError) as error:
        # 中身が壊れている（UnpicklingError）、空や途中で切れている（EOFError）、学習のチェックポイントとは別の構造
        # （KeyError・IndexError・TypeError）、ネットワークの構成が合わない（RuntimeError）は、いずれもこの
        # チェックポイントを使えない点で同じ扱いにする
        raise CheckpointLoadError(f"チェックポイントを読み込めない: {checkpoint_path}: {error}") from error
    return RslRlPolicy(runner.get_inference_policy(device=str(device)))
