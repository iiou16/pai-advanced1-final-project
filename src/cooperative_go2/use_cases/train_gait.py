"""歩行方策を学習する流れ。"""

from typing import Protocol

from cooperative_go2.entities.gait_config import GaitConfig


class RunAlreadyExistsError(RuntimeError):
    """同じ名前の学習の出力先が既にあり、上書きすると過去の結果を失うことを表す。"""


class GaitRunStore(Protocol):
    """学習 1 回分の出力先（チェックポイント・ログ・設定の写し）。"""

    def initialize(self) -> None:
        """出力先を新しく作り、学習に使う設定を保存する。

        Raises:
            RunAlreadyExistsError: 出力先が既にある。
        """
        ...


class GaitTrainer(Protocol):
    """歩行方策を PPO で学習し、チェックポイントを出力先に保存する。"""

    def learn(self, num_iterations: int) -> None:
        """PPO を指定した反復回数だけ進める。

        Raises:
            SimulationSetupError: 学習中に衝突候補などの数が設定の上限を超えた。
        """
        ...


class GaitTrainerFactory(Protocol):
    """シミュレーション環境と学習器を組み立てる。環境の構築は重いため、出力先の準備が済んでから呼ぶ。"""

    def create(self, config: GaitConfig) -> GaitTrainer:
        """設定に従って学習器を作る。

        Raises:
            SimulationSetupError: シミュレーションを用意できない。
        """
        ...


def train_gait(config: GaitConfig, run_store: GaitRunStore, trainer_factory: GaitTrainerFactory) -> None:
    """設定に従って歩行方策を学習する。

    出力先を先に用意するため、名前が重複していれば環境を構築する前に止まる。

    Raises:
        RunAlreadyExistsError: 同じ名前の出力先が既にある。
        SimulationSetupError: シミュレーションを用意できない、または学習中に設定の上限を超えた。
            出力先は作成済みのまま残る。
    """
    run_store.initialize()
    trainer = trainer_factory.create(config)
    trainer.learn(config.training.max_iterations)
