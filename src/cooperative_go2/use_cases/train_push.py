"""箱押しの高レベル方策を学習する流れ。"""

from typing import Protocol

from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError


class PushRunStore(Protocol):
    """学習 1 回分の出力先（チェックポイント・ログ・設定の写し）。"""

    def initialize(self) -> None:
        """出力先を新しく作り、学習に使う設定を保存する。

        Raises:
            RunAlreadyExistsError: 出力先が既にある。
        """
        ...


class PushTrainer(Protocol):
    """高レベル方策を PPO で学習し、チェックポイントを出力先に保存する。"""

    def learn(self, num_iterations: int) -> None:
        """PPO を指定した反復回数だけ進める。

        Raises:
            SimulationSetupError: 学習中に衝突候補などの数が設定の上限を超えた。
        """
        ...


class PushTrainerFactory(Protocol):
    """シミュレーション環境と学習器を組み立てる。環境の構築は重いため、出力先の準備が済んでから呼ぶ。"""

    def create(self) -> PushTrainer:
        """学習器を作る。

        Raises:
            SimulationSetupError: シミュレーションを用意できない。
        """
        ...


def train_push(num_iterations: int, run_store: PushRunStore, trainer_factory: PushTrainerFactory) -> None:
    """高レベル方策を num_iterations 反復学習する。

    出力先を先に用意するため、名前が重複していれば環境を構築する前に止まる。

    Raises:
        RunAlreadyExistsError: 同じ名前の出力先が既にある。
        SimulationSetupError: シミュレーションを用意できない、または学習中に設定の上限を超えた。
            出力先は作成済みのまま残る。
    """
    run_store.initialize()
    trainer = trainer_factory.create()
    trainer.learn(num_iterations)


__all__ = ["PushRunStore", "PushTrainer", "PushTrainerFactory", "RunAlreadyExistsError", "train_push"]
