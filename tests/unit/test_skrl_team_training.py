from pathlib import Path

import pytest
import torch
import yaml
from conftest import REPO_ROOT, TeamMarlEnvFactory

from cooperative_go2.entities.team_marl_observation import TEAM_ACTOR_OBSERVATION_DIM
from cooperative_go2.interface_adapters.config_loader import load_team_marl_run_settings
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.skrl_marl_config import MarlAlgorithm, MarlVariant
from cooperative_go2.interface_adapters.skrl_team import SkrlTeamEnv, build_team_agent
from cooperative_go2.interface_adapters.skrl_team_policy import InvalidRunRecordError, load_team_policy, load_variant
from cooperative_go2.interface_adapters.skrl_team_training import (
    FileTeamMarlRunStore,
    SkrlTeamTrainer,
    TeamMarlRunRecord,
)
from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError

NAMES = ("go2_a", "go2_b")
ROLLOUTS = 4
VARIANT = MarlVariant(MarlAlgorithm.MAPPO, shared_parameters=False)


@pytest.fixture
def small_config(tmp_path: Path) -> Path:
    # 1 回の更新ごとにチェックポイントを残す、小さな学習の設定
    data = yaml.safe_load((REPO_ROOT / "configs" / "push_multi.yaml").read_text(encoding="utf-8"))
    data["skrl"] |= {"rollouts": ROLLOUTS, "checkpoint_interval_updates": 1, "mini_batches": 1}
    path = tmp_path / "push_multi_small.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _record(config: Path) -> TeamMarlRunRecord:
    return TeamMarlRunRecord(config, REPO_ROOT / "configs" / "scenes" / "s3_team.yaml", VARIANT, "abc123")


def _train(make_team_marl_env: TeamMarlEnvFactory, config: Path, run_dir: Path, iterations: int) -> None:
    FileTeamMarlRunStore(run_dir, _record(config)).initialize()
    ppo = load_team_marl_run_settings(config).settings.skrl
    env = SkrlTeamEnv(make_team_marl_env()[0], NAMES)
    agent = build_team_agent(env, ppo, VARIANT, run_dir)
    SkrlTeamTrainer(env, agent, ppo.rollouts).learn(iterations)


def test_training_saves_a_checkpoint_that_loads_as_an_actor(
    make_team_marl_env: TeamMarlEnvFactory, small_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"

    _train(make_team_marl_env, small_config, run_dir, iterations=2)
    policy = load_team_policy(run_dir, iteration=2, agent_name="go2_b", device="cpu")

    commands = policy.mean_commands(torch.zeros(3, TEAM_ACTOR_OBSERVATION_DIM))
    assert commands.shape == (3, 3)
    assert load_variant(run_dir) == VARIANT
    assert (run_dir / "git_commit.txt").read_text(encoding="utf-8").strip() == "abc123"


def test_missing_checkpoint_is_reported(
    make_team_marl_env: TeamMarlEnvFactory, small_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _train(make_team_marl_env, small_config, run_dir, iterations=1)

    with pytest.raises(CheckpointNotFoundError):
        load_team_policy(run_dir, iteration=5, agent_name="go2_a", device="cpu")


def test_unknown_robot_name_is_reported(
    make_team_marl_env: TeamMarlEnvFactory, small_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _train(make_team_marl_env, small_config, run_dir, iterations=1)

    with pytest.raises(CheckpointLoadError):
        load_team_policy(run_dir, iteration=1, agent_name="go2_c", device="cpu")


def test_existing_run_dir_is_not_overwritten(small_config: Path, tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(RunAlreadyExistsError):
        FileTeamMarlRunStore(run_dir, _record(small_config)).initialize()


@pytest.mark.parametrize(
    "content",
    ["{not json", '{"algorithm": "mappo"}', '{"algorithm": "qmix", "shared_parameters": true}',
     '{"algorithm": "mappo", "shared_parameters": "false"}'],
    ids=["broken-json", "missing-key", "unknown-algorithm", "non-boolean-sharing"],
)  # fmt: skip
def test_invalid_variant_record_is_reported(tmp_path: Path, content: str) -> None:
    (tmp_path / "variant.json").write_text(content, encoding="utf-8")

    with pytest.raises(InvalidRunRecordError):
        load_variant(tmp_path)


def test_truncated_checkpoint_is_reported_as_unloadable(
    make_team_marl_env: TeamMarlEnvFactory, small_config: Path, tmp_path: Path
) -> None:
    run_dir = tmp_path / "run"
    _train(make_team_marl_env, small_config, run_dir, iterations=1)
    checkpoint = run_dir / "checkpoints" / f"agent_{ROLLOUTS}.pt"
    checkpoint.write_bytes(checkpoint.read_bytes()[:100])

    with pytest.raises(CheckpointLoadError):
        load_team_policy(run_dir, iteration=1, agent_name="go2_a", device="cpu")
