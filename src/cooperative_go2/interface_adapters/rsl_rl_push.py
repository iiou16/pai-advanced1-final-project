"""RSL-RL の OnPolicyRunner で箱押しの高レベル方策を学習する。学習の出力先には設定ファイルの写しを残す。"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch
from rsl_rl.runners import OnPolicyRunner

from cooperative_go2.interface_adapters.config_loader import PushSceneSettings, PushTrainingSettings, load_gait_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_push_scene import PushSceneRequest
from cooperative_go2.interface_adapters.genesis_push_simulation import GenesisPushSimulation
from cooperative_go2.interface_adapters.push_env import Go2PushEnv, PushEnvSetup
from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError

# 出力先に複製する設定ファイルの名前。評価時は push_single.yaml の写しから方策のネットワーク構成を読み、
# 評価の設定と学習時の条件が食い違っていないかを確かめる。scene.yaml は学習時のシーンの記録として残す
PUSH_RUN_CONFIG_FILENAME = "push_single.yaml"
PUSH_RUN_SCENE_CONFIG_FILENAME = "scene.yaml"


class FilePushRunStore:
    """出力先のディレクトリを作り、学習に使った箱押しの設定とシーンの設定を複製する。"""

    def __init__(self, run_dir: Path, config_source: Path, scene_config_source: Path) -> None:
        """出力先と、複製する設定ファイルを受け取る。ディレクトリは initialize() まで作らない。

        Args:
            run_dir: 出力先のディレクトリ。まだ存在してはならない。
            config_source: 学習に使う箱押しの設定ファイル。読み込み・検証済みであること。
            scene_config_source: 学習に使うシーンの設定ファイル。読み込み・検証済みであること。
        """
        self._run_dir = run_dir
        self._config_source = config_source
        self._scene_config_source = scene_config_source

    def initialize(self) -> None:
        """出力先を作り、設定ファイルを複製する。

        Raises:
            RunAlreadyExistsError: 出力先が既にある。過去の結果を上書きしないため止める。
        """
        try:
            self._run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunAlreadyExistsError(f"出力先が既にある（上書きしない）: {self._run_dir}") from error
        shutil.copyfile(self._config_source, self._run_dir / PUSH_RUN_CONFIG_FILENAME)
        shutil.copyfile(self._scene_config_source, self._run_dir / PUSH_RUN_SCENE_CONFIG_FILENAME)


class RslRlPushTrainer:
    """OnPolicyRunner で高レベル方策を学習し、チェックポイントを model_<反復>.pt として出力先に保存する。"""

    def __init__(self, env: Go2PushEnv, settings: PushTrainingSettings, run_dir: Path) -> None:
        """学習器を組み立てる。ログは run_dir に、実行の名前は run_dir の名前にする。"""
        runner_cfg = settings.rsl_rl.to_runner_cfg(run_dir.name)
        self._runner = OnPolicyRunner(env, runner_cfg, str(run_dir), device=str(env.device))
        # 学習時の本リポジトリのコミットと差分をログに残し、チェックポイントの作成条件を後から辿れるようにする
        self._runner.add_git_repo_to_log(__file__)

    def learn(self, num_iterations: int) -> None:
        """PPO を指定した反復回数だけ進める。エピソードの進みは環境ごとにばらして始める。"""
        self._runner.learn(num_learning_iterations=num_iterations, init_at_random_ep_len=True)


@dataclass(frozen=True)
class RslRlPushTrainerFactory:
    """シーン S1_push・固定した歩行方策・箱押しの環境・RSL-RL の学習器を組み立てる。前提: gs.init() 済み。

    Attributes:
        run_dir: 学習の出力先。
        settings: 箱押しの設定。
        scene: シーンの設定。
        generator: 配置とリセット時の関節角のばらつきを引く乱数生成器。学習のシードで固定して渡す。
    """

    run_dir: Path
    settings: PushTrainingSettings
    scene: PushSceneSettings
    generator: torch.Generator

    def create(self) -> RslRlPushTrainer:
        """環境と学習器を作る。

        Raises:
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
            InvalidPushConfigError: 箱押しの設定が歩行方策の学習時の設定と矛盾する。
            CheckpointNotFoundError, CheckpointLoadError: 歩行方策を読み込めない。
        """
        settings = self.settings
        gait_run_dir = Path(settings.gait_policy.run_dir)
        gait = load_gait_settings(gait_run_dir / RUN_CONFIG_FILENAME).gait
        num_envs = settings.task.episode.num_envs
        simulation = GenesisPushSimulation.build(gait, self.scene, PushSceneRequest(num_envs, video=None))
        controller = load_gait_controller(gait_run_dir, settings.gait_policy.checkpoint, num_envs, gs.device)
        setup = PushEnvSetup(settings.task, gait, self.scene.box.initial_xy_m, self.generator)
        return RslRlPushTrainer(Go2PushEnv(setup, simulation, controller), settings, self.run_dir)
