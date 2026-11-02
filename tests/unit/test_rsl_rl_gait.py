"""RSL-RL との契約テスト。シミュレータを使わず、CPU で OnPolicyRunner を組み立てる。"""

import dataclasses
from pathlib import Path

import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from cooperative_go2.entities.gait_observation import GAIT_OBSERVATION_DIM
from cooperative_go2.interface_adapters.config_loader import load_gait_settings
from cooperative_go2.interface_adapters.rsl_rl_config import RslRlPpoConfig
from cooperative_go2.interface_adapters.rsl_rl_gait import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    GaitPolicySpecEnvironment,
    load_rsl_rl_gait_policy,
)

DEVICE = "cpu"


@pytest.fixture
def rsl_rl_config(gait_config_path: Path) -> RslRlPpoConfig:
    return load_gait_settings(gait_config_path).rsl_rl


@pytest.fixture
def checkpoint_path(tmp_path: Path, rsl_rl_config: RslRlPpoConfig) -> Path:
    """学習直後と同じ形式のチェックポイント（初期化したネットワーク）を保存する。"""
    torch.manual_seed(0)
    runner_cfg = rsl_rl_config.to_runner_cfg("contract-test")
    runner = OnPolicyRunner(GaitPolicySpecEnvironment(DEVICE), runner_cfg, None, device=DEVICE)
    path = tmp_path / "model_0.pt"
    runner.save(str(path))
    return path


def test_loaded_policy_maps_observation_to_twelve_joint_actions(checkpoint_path, rsl_rl_config) -> None:
    policy = load_rsl_rl_gait_policy(checkpoint_path, rsl_rl_config, DEVICE)

    actions = policy(torch.zeros(5, GAIT_OBSERVATION_DIM))

    assert actions.shape == (5, 12)


def test_same_checkpoint_and_input_give_same_output(checkpoint_path, rsl_rl_config) -> None:
    observation = torch.linspace(-1.0, 1.0, GAIT_OBSERVATION_DIM).unsqueeze(0)

    first = load_rsl_rl_gait_policy(checkpoint_path, rsl_rl_config, DEVICE)(observation)
    second = load_rsl_rl_gait_policy(checkpoint_path, rsl_rl_config, DEVICE)(observation)

    assert torch.equal(first, second)


def test_missing_checkpoint_is_reported(tmp_path, rsl_rl_config) -> None:
    with pytest.raises(CheckpointNotFoundError):
        load_rsl_rl_gait_policy(tmp_path / "model_999.pt", rsl_rl_config, DEVICE)


def test_checkpoint_with_different_network_is_rejected(checkpoint_path, rsl_rl_config) -> None:
    wider = dataclasses.replace(rsl_rl_config, actor_hidden_dims=(64, 64))

    with pytest.raises(CheckpointLoadError) as exc_info:
        load_rsl_rl_gait_policy(checkpoint_path, wider, DEVICE)
    assert isinstance(exc_info.value.__cause__, RuntimeError)


def test_empty_checkpoint_is_rejected(tmp_path, rsl_rl_config) -> None:
    empty_checkpoint = tmp_path / "model_0.pt"
    empty_checkpoint.write_bytes(b"")

    with pytest.raises(CheckpointLoadError) as exc_info:
        load_rsl_rl_gait_policy(empty_checkpoint, rsl_rl_config, DEVICE)
    assert isinstance(exc_info.value.__cause__, EOFError)


def test_spec_environment_cannot_be_stepped() -> None:
    with pytest.raises(RuntimeError):
        GaitPolicySpecEnvironment(DEVICE).step(torch.zeros(1, 12))
