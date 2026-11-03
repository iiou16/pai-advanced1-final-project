"""RSL-RL の OnPolicyRunner で歩行方策を学習する。"""

from pathlib import Path

import torch
from rsl_rl.runners import OnPolicyRunner

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.interface_adapters.gait_env import Go2GaitEnv
from cooperative_go2.interface_adapters.genesis_gait_simulation import GenesisGaitSimulation
from cooperative_go2.interface_adapters.rsl_rl_config import RslRlPpoConfig


class RslRlGaitTrainer:
    """OnPolicyRunner で歩行方策を学習し、チェックポイントを model_<反復>.pt として出力先に保存する。"""

    def __init__(self, env: Go2GaitEnv, rsl_rl_config: RslRlPpoConfig, run_dir: Path) -> None:
        """学習器を組み立てる。ログと設定の写しは run_dir に、実行の名前は run_dir の名前にする。

        Args:
            env: 学習に使う環境。
            rsl_rl_config: PPO とネットワークの設定。
            run_dir: 出力先。学習中にチェックポイント・TensorBoard のログ・リポジトリの差分（git/）を書き込む。
        """
        runner_cfg = rsl_rl_config.to_runner_cfg(run_dir.name)
        self._runner = OnPolicyRunner(env, runner_cfg, str(run_dir), device=str(env.device))
        # 学習時の本リポジトリのコミットと差分をログに残し、チェックポイントの作成条件を後から辿れるようにする
        self._runner.add_git_repo_to_log(__file__)

    def learn(self, num_iterations: int) -> None:
        """PPO を指定した反復回数だけ進める。エピソードの進みは環境ごとにばらして始める。"""
        self._runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)


class RslRlGaitTrainerFactory:
    """シーン S0_flat の歩行環境と RSL-RL の学習器を組み立てる。前提: gs.init() 済みであること。"""

    def __init__(self, run_dir: Path, rsl_rl_config: RslRlPpoConfig, generator: torch.Generator) -> None:
        """学習器の組み立てに使う出力先・設定・乱数生成器を受け取る。

        Args:
            run_dir: 学習の出力先。
            rsl_rl_config: PPO とネットワークの設定。
            generator: 学習中の速度指令とリセット時の関節角のノイズを引く乱数生成器。学習のシードで固定して渡す。
        """
        self._run_dir = run_dir
        self._rsl_rl_config = rsl_rl_config
        self._generator = generator

    def create(self, config: GaitConfig) -> RslRlGaitTrainer:
        """設定に従ってシーン・環境・学習器を作る。

        Raises:
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
        """
        env = Go2GaitEnv(config, GenesisGaitSimulation.build(config, camera_spec=None), self._generator)
        return RslRlGaitTrainer(env, self._rsl_rl_config, self._run_dir)
