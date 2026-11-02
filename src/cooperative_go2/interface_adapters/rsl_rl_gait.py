"""RSL-RL の OnPolicyRunner で歩行方策を学習し、学習済みの方策を読み込む。"""

import pickle
from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner
from tensordict import TensorDict

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_DIM
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.interface_adapters.genesis_flat_scene import build_flat_scene
from cooperative_go2.interface_adapters.genesis_gait_env import GenesisGo2GaitEnv
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP, RslRlPpoConfig


class CheckpointNotFoundError(FileNotFoundError):
    """指定したチェックポイントのファイルが無いことを表す。"""


class CheckpointLoadError(RuntimeError):
    """チェックポイントを読み込めない（ファイルが壊れている、ネットワークの構成が設定と合わないなど）ことを表す。"""


class RslRlGaitTrainer:
    """OnPolicyRunner で歩行方策を学習し、チェックポイントを model_<反復>.pt として出力先に保存する。"""

    def __init__(self, env: GenesisGo2GaitEnv, rsl_rl_config: RslRlPpoConfig, run_dir: Path) -> None:
        runner_cfg = rsl_rl_config.to_runner_cfg(run_dir.name)
        self._runner = OnPolicyRunner(env, runner_cfg, str(run_dir), device=str(env.device))
        # 学習時の本リポジトリのコミットと差分をログに残し、チェックポイントの作成条件を後から辿れるようにする
        self._runner.add_git_repo_to_log(__file__)

    def learn(self, num_iterations: int) -> None:
        """PPO を指定した反復回数だけ進める。エピソードの進みは環境ごとにばらして始める。"""
        self._runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)


class RslRlGaitTrainerFactory:
    """シーン S0_flat の歩行環境と RSL-RL の学習器を組み立てる。前提: gs.init() 済みであること。"""

    def __init__(self, run_dir: Path, rsl_rl_config: RslRlPpoConfig, command_generator: torch.Generator) -> None:
        """学習器の組み立てに使う出力先・設定・乱数生成器を受け取る。

        Args:
            run_dir: 学習の出力先。
            rsl_rl_config: PPO とネットワークの設定。
            command_generator: 学習中の速度指令を引く乱数生成器。学習のシードで固定して渡す。
        """
        self._run_dir = run_dir
        self._rsl_rl_config = rsl_rl_config
        self._command_generator = command_generator

    def create(self, config: GaitConfig) -> RslRlGaitTrainer:
        """設定に従ってシーン・環境・学習器を作る。"""
        env = GenesisGo2GaitEnv(config, build_flat_scene(config, camera_spec=None), self._command_generator)
        return RslRlGaitTrainer(env, self._rsl_rl_config, self._run_dir)


class GaitPolicySpecEnvironment:
    """シミュレータを使わずに OnPolicyRunner を組み立てるための、観測と行動の次元だけを持つ環境。

    推論専用で、step() は呼べない。OnPolicyRunner は構築時に観測の形を 1 回読むだけで、推論では環境を使わない。
    """

    def __init__(self, device: torch.device | str) -> None:
        """観測を device に置く 1 環境分の仕様を作る。"""
        self.num_envs = 1
        self.num_actions = NUM_GO2_JOINTS
        self.max_episode_length = 1
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=device)
        self.device = device
        self.cfg: dict[str, object] = {}

    def get_observations(self) -> TensorDict:
        """形だけを決めるための 0 の観測を返す。"""
        observation = torch.zeros((self.num_envs, GAIT_OBSERVATION_DIM), device=self.device)
        return TensorDict({POLICY_OBSERVATION_GROUP: observation}, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        """推論専用の環境は進められないので、常に例外を送出する。"""
        raise RuntimeError("GaitPolicySpecEnvironment は推論用の方策を組み立てるためだけの環境で、step() できない")


class RslRlGaitPolicy:
    """RSL-RL の方策ネットワークを、観測テンソル → 行動テンソルの関数として包む。出力は決定的（分布の平均）。"""

    def __init__(self, policy: torch.nn.Module) -> None:
        self._policy = policy

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        """観測 (num_envs, GAIT_OBSERVATION_DIM) から行動 (num_envs, 12) を返す。"""
        observations = TensorDict({POLICY_OBSERVATION_GROUP: observation}, batch_size=[observation.shape[0]])
        return self._policy(observations)


def load_rsl_rl_gait_policy(
    checkpoint_path: Path, rsl_rl_config: RslRlPpoConfig, device: torch.device | str
) -> RslRlGaitPolicy:
    """学習済みのチェックポイントを読み込み、推論用の方策を返す。シミュレータは使わない。

    Args:
        checkpoint_path: model_<反復>.pt のパス。
        rsl_rl_config: チェックポイントを学習したときの RSL-RL の設定。ネットワークの構成を合わせるために使う。
        device: 方策を置くデバイス。

    Raises:
        CheckpointNotFoundError: checkpoint_path にファイルが無い。
        CheckpointLoadError: ファイルを読み込めない、またはネットワークの構成が設定と合わない。
    """
    if not checkpoint_path.is_file():
        raise CheckpointNotFoundError(f"チェックポイントが無い: {checkpoint_path}")
    runner_cfg = rsl_rl_config.to_runner_cfg(checkpoint_path.parent.name)
    runner = OnPolicyRunner(GaitPolicySpecEnvironment(device), runner_cfg, None, device=str(device))
    try:
        runner.load(str(checkpoint_path), map_location=str(device))
    except torch.OutOfMemoryError:
        # メモリ不足は RuntimeError の一種だが、チェックポイントの不良ではないので変換せずにそのまま伝える
        raise
    except (pickle.UnpicklingError, EOFError, RuntimeError) as error:
        # 中身が壊れている（UnpicklingError）、空や途中で切れている（EOFError）、ネットワークの構成が合わない
        # （RuntimeError）は、いずれもこのチェックポイントを使えない点で同じ扱いにする
        raise CheckpointLoadError(f"チェックポイントを読み込めない: {checkpoint_path}: {error}") from error
    return RslRlGaitPolicy(runner.get_inference_policy(device=str(device)))
