"""skrl の IPPO・MAPPO で 2 台の協調を学習する。学習の出力先には設定ファイルの写しと学習方式の記録を残す。"""

import shutil
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch
from skrl.multi_agents.torch import MultiAgent
from skrl.trainers.torch import SequentialTrainer

from cooperative_go2.interface_adapters.config_loader import TeamMarlRunSettings, load_gait_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.skrl_marl_config import MarlVariant
from cooperative_go2.interface_adapters.skrl_team import SkrlTeamEnv, build_team_agent
from cooperative_go2.interface_adapters.skrl_team_policy import (
    TEAM_RUN_COMMIT_FILENAME,
    TEAM_RUN_CONFIG_FILENAME,
    TEAM_RUN_SCENE_CONFIG_FILENAME,
    write_variant,
)
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv, TeamMarlEnvSetup
from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError


@dataclass(frozen=True)
class TeamMarlRunRecord:
    """学習の出力先に残す記録。

    Attributes:
        config_source: 学習に使う協調学習の設定ファイル。読み込み・検証済みであること。
        scene_config_source: 学習に使うシーンの設定ファイル。読み込み・検証済みであること。
        variant: 学習方式。
        commit_sha: 学習に使ったリポジトリのコミット（未コミットの変更があれば -dirty 付き）。
    """

    config_source: Path
    scene_config_source: Path
    variant: MarlVariant
    commit_sha: str


class FileTeamMarlRunStore:
    """出力先のディレクトリを作り、設定ファイルの写し・学習方式・コミットを残す。"""

    def __init__(self, run_dir: Path, record: TeamMarlRunRecord) -> None:
        """出力先と残す記録を受け取る。ディレクトリは initialize() まで作らない。"""
        self._run_dir = run_dir
        self._record = record

    def initialize(self) -> None:
        """出力先を作り、記録を残す。

        Raises:
            RunAlreadyExistsError: 出力先が既にある。過去の結果を上書きしないため止める（skrl は既存の出力先に書き足すため、
                ここで止める）。
        """
        try:
            self._run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunAlreadyExistsError(f"出力先が既にある（上書きしない）: {self._run_dir}") from error
        record = self._record
        shutil.copyfile(record.config_source, self._run_dir / TEAM_RUN_CONFIG_FILENAME)
        shutil.copyfile(record.scene_config_source, self._run_dir / TEAM_RUN_SCENE_CONFIG_FILENAME)
        write_variant(self._run_dir, record.variant)
        (self._run_dir / TEAM_RUN_COMMIT_FILENAME).write_text(record.commit_sha + "\n", encoding="utf-8")


class SkrlTeamTrainer:
    """skrl の SequentialTrainer で協調を学習する。チェックポイントは出力先の checkpoints/ に保存される。"""

    def __init__(self, env: SkrlTeamEnv, agent: MultiAgent, rollouts: int) -> None:
        """skrl の環境・エージェントと、1 回の更新で集めるステップ数を受け取る。"""
        self._env = env
        self._agent = agent
        self._rollouts = rollouts

    def learn(self, num_iterations: int) -> None:
        """num_iterations 回の更新（各 rollouts ステップを集める）だけ学習する。"""
        trainer = SequentialTrainer(
            env=self._env,
            agents=self._agent,
            cfg={"timesteps": num_iterations * self._rollouts, "headless": True, "close_environment_at_exit": False},
        )
        trainer.train()


@dataclass(frozen=True)
class SkrlTeamTrainerFactory:
    """シーン S3_team・固定した歩行方策・協調学習の環境・skrl の学習器を組み立てる。前提: gs.init() 済み。

    Attributes:
        run_dir: 学習の出力先。
        run: 協調学習の設定とシーンの設定。
        variant: 学習方式。
        generator: 開始位置・関節角のばらつき・通信の欠落を引く乱数生成器。学習のシードで固定して渡す。
    """

    run_dir: Path
    run: TeamMarlRunSettings
    variant: MarlVariant
    generator: torch.Generator

    def create(self) -> SkrlTeamTrainer:
        """環境と学習器を作る。

        Raises:
            FileNotFoundError: 歩行方策の設定の写し（gait.yaml）が無い。
            ConfigValidationError: 歩行方策の設定の写しが現在の設定の形式に合わない。
            SimulationSetupError: シーンを構築できない、または Go2 のモデルが想定と異なる。
            InvalidPushConfigError: 協調学習の設定が歩行方策の学習時の設定と矛盾する。
            CheckpointNotFoundError, CheckpointLoadError: 歩行方策を読み込めない。
        """
        settings = self.run.settings
        gait_run_dir = Path(settings.gait_policy.run_dir)
        gait = load_gait_settings(gait_run_dir / RUN_CONFIG_FILENAME).gait
        num_envs = settings.task.episode.num_envs
        simulation = GenesisCorridorSimulation.build(gait, self.run.scene, CorridorSceneRequest(num_envs, camera=None))
        controllers = tuple(
            load_gait_controller(gait_run_dir, settings.gait_policy.checkpoint, num_envs, gs.device)
            for _ in simulation.robot_names
        )
        setup = TeamMarlEnvSetup(settings.task, gait, self.run.scene.scene, self.generator)
        env = SkrlTeamEnv(Go2TeamMarlEnv(setup, simulation, controllers), simulation.robot_names)
        agent = build_team_agent(env, settings.skrl, self.variant, self.run_dir)
        return SkrlTeamTrainer(env, agent, settings.skrl.rollouts)
