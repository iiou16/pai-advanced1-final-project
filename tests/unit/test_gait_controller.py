"""歩行方策の推論器の契約: 観測の組み立て、出力の目標関節角、前回の行動の保持、チェックポイントからの再現性。"""

import shutil
from pathlib import Path

import pytest
import torch
from rsl_rl.runners import OnPolicyRunner

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import (
    GAIT_OBSERVATION_DIM,
    GaitObservationInputs,
    build_gait_observation,
)
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2Proprioception
from cooperative_go2.interface_adapters.config_loader import load_gait_settings
from cooperative_go2.interface_adapters.gait_checkpoint import checkpoint_path, load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.rsl_rl_gait import PolicySpecEnvironment, load_rsl_rl_gait_policy
from cooperative_go2.use_cases.gait_controller import GaitController, InvalidVelocityCommandShapeError

DEVICE = "cpu"
NUM_ROBOTS = 2
CHECKPOINT_ITERATION = 0


@pytest.fixture
def gait_config(gait_config_path: Path) -> GaitConfig:
    return load_gait_settings(gait_config_path).gait


def _proprioception(scale: float = 1.0) -> Go2Proprioception:
    return Go2Proprioception(
        base_ang_vel=torch.tensor([[0.1, -0.2, 0.3]] * NUM_ROBOTS) * scale,
        projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * NUM_ROBOTS),
        dof_pos=torch.linspace(-1.0, 1.0, 12).expand(NUM_ROBOTS, -1) * scale,
        dof_vel=torch.linspace(2.0, -2.0, 12).expand(NUM_ROBOTS, -1) * scale,
    )


def _commands() -> torch.Tensor:
    return torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 1.0]])


class RecordingPolicy:
    """受け取った観測を記録し、決まった行動を返す方策。"""

    def __init__(self, action_value: float) -> None:
        self.action_value = action_value
        self.observations: list[torch.Tensor] = []

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.observations.append(observation)
        return torch.full((observation.shape[0], 12), self.action_value)


def _default_dof_pos(config: GaitConfig) -> torch.Tensor:
    return torch.tensor(config.robot.default_joint_angles_rad.in_order())


def test_output_is_absolute_joint_target_from_scaled_residual(gait_config: GaitConfig) -> None:
    controller = GaitController(RecordingPolicy(action_value=1.0), gait_config, NUM_ROBOTS, DEVICE)

    joint_targets = controller.compute_joint_targets(_proprioception(), _commands())

    expected = _default_dof_pos(gait_config) + gait_config.env.action_scale
    assert torch.allclose(joint_targets, expected.expand(NUM_ROBOTS, -1))


def test_residual_is_clipped_like_training(gait_config: GaitConfig) -> None:
    clip = gait_config.env.clip_actions
    controller = GaitController(RecordingPolicy(action_value=clip * 10), gait_config, NUM_ROBOTS, DEVICE)

    joint_targets = controller.compute_joint_targets(_proprioception(), _commands())

    expected = _default_dof_pos(gait_config) + clip * gait_config.env.action_scale
    assert torch.allclose(joint_targets, expected.expand(NUM_ROBOTS, -1))


def test_policy_sees_training_observation_with_previous_action(gait_config: GaitConfig) -> None:
    policy = RecordingPolicy(action_value=0.5)
    controller = GaitController(policy, gait_config, NUM_ROBOTS, DEVICE)

    controller.compute_joint_targets(_proprioception(), _commands())
    controller.compute_joint_targets(_proprioception(), _commands())

    expected = build_gait_observation(
        GaitObservationInputs(
            proprioception=_proprioception(),
            commands=_commands(),
            last_actions=torch.full((NUM_ROBOTS, 12), 0.5),
            default_dof_pos=_default_dof_pos(gait_config),
        ),
        gait_config.observation,
    )
    assert torch.allclose(policy.observations[1], expected)


def test_reset_clears_previous_action_only_for_masked_robots(gait_config: GaitConfig) -> None:
    policy = RecordingPolicy(action_value=0.5)
    controller = GaitController(policy, gait_config, NUM_ROBOTS, DEVICE)
    controller.compute_joint_targets(_proprioception(), _commands())

    controller.reset_where(torch.tensor([True, False]))
    controller.compute_joint_targets(_proprioception(), _commands())

    last_actions = policy.observations[1][:, -12:]
    assert torch.equal(last_actions[0], torch.zeros(12))
    assert torch.equal(last_actions[1], torch.full((12,), 0.5))


@pytest.mark.parametrize(
    "shape",
    [(NUM_ROBOTS, 2), (NUM_ROBOTS + 1, 3), (3,)],
    ids=["wrong-command-dim", "wrong-robot-count", "one-dimensional"],
)
def test_command_with_wrong_shape_is_rejected(gait_config: GaitConfig, shape: tuple[int, ...]) -> None:
    controller = GaitController(RecordingPolicy(action_value=0.0), gait_config, NUM_ROBOTS, DEVICE)

    with pytest.raises(InvalidVelocityCommandShapeError):
        controller.compute_joint_targets(_proprioception(), torch.zeros(shape))


@pytest.fixture
def run_dir(tmp_path: Path, gait_config_path: Path) -> Path:
    """学習の出力先と同じ構成（設定の写しと初期化したネットワークのチェックポイント）のディレクトリ。"""
    shutil.copyfile(gait_config_path, tmp_path / RUN_CONFIG_FILENAME)
    torch.manual_seed(0)
    runner_cfg = load_gait_settings(gait_config_path).rsl_rl.to_runner_cfg("controller-test")
    spec = PolicySpecEnvironment(GAIT_OBSERVATION_DIM, NUM_GO2_JOINTS, DEVICE)
    runner = OnPolicyRunner(spec, runner_cfg, None, device=DEVICE)
    runner.save(str(checkpoint_path(tmp_path, CHECKPOINT_ITERATION)))
    return tmp_path


def _rollout_joint_targets(controller: GaitController) -> list[torch.Tensor]:
    return [controller.compute_joint_targets(_proprioception(scale), _commands()) for scale in (1.0, 0.5, -0.5)]


def test_same_checkpoint_and_inputs_give_identical_joint_targets(run_dir: Path) -> None:
    first = load_gait_controller(run_dir, CHECKPOINT_ITERATION, NUM_ROBOTS, DEVICE)
    torch.manual_seed(1)  # 2 つ目の読み込みでネットワークの初期値を変え、出力が重みだけで決まることを確かめる
    second = load_gait_controller(run_dir, CHECKPOINT_ITERATION, NUM_ROBOTS, DEVICE)

    first_targets = _rollout_joint_targets(first)
    second_targets = _rollout_joint_targets(second)

    for first_step, second_step in zip(first_targets, second_targets, strict=True):
        assert torch.equal(first_step, second_step)


def test_loaded_policy_does_not_track_gradients_even_outside_no_grad(run_dir: Path, gait_config_path: Path) -> None:
    rsl_rl_config = load_gait_settings(gait_config_path).rsl_rl
    policy = load_rsl_rl_gait_policy(checkpoint_path(run_dir, CHECKPOINT_ITERATION), rsl_rl_config, DEVICE)

    actions = policy(torch.zeros(NUM_ROBOTS, GAIT_OBSERVATION_DIM))

    assert not actions.requires_grad
