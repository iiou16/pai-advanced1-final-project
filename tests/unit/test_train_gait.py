from pathlib import Path

import pytest

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.interface_adapters.config_loader import load_gait_settings
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME, FileGaitRunStore
from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError, train_gait


class FakeTrainer:
    def __init__(self) -> None:
        self.learned_iterations: list[int] = []

    def learn(self, num_iterations: int) -> None:
        self.learned_iterations.append(num_iterations)


class FakeTrainerFactory:
    def __init__(self) -> None:
        self.trainer = FakeTrainer()
        self.created = False

    def create(self, config: GaitConfig) -> FakeTrainer:
        self.created = True
        return self.trainer


@pytest.fixture
def config(gait_config_path: Path) -> GaitConfig:
    return load_gait_settings(gait_config_path).gait


def test_training_runs_configured_iterations_and_keeps_config_copy(
    tmp_path: Path, gait_config_path: Path, config: GaitConfig
) -> None:
    run_dir = tmp_path / "gait-run"
    factory = FakeTrainerFactory()

    train_gait(config, FileGaitRunStore(run_dir, gait_config_path), factory)

    assert factory.trainer.learned_iterations == [config.training.max_iterations]
    assert (run_dir / RUN_CONFIG_FILENAME).read_bytes() == gait_config_path.read_bytes()


def test_existing_run_dir_stops_before_building_trainer(
    tmp_path: Path, gait_config_path: Path, config: GaitConfig
) -> None:
    run_dir = tmp_path / "gait-run"
    run_dir.mkdir()
    previous_checkpoint = run_dir / "model_100.pt"
    previous_checkpoint.write_bytes(b"previous")
    factory = FakeTrainerFactory()

    with pytest.raises(RunAlreadyExistsError) as exc_info:
        train_gait(config, FileGaitRunStore(run_dir, gait_config_path), factory)

    assert isinstance(exc_info.value.__cause__, FileExistsError)
    assert not factory.created
    assert previous_checkpoint.read_bytes() == b"previous"
