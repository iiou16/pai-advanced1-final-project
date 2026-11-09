import dataclasses
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2State, initial_go2_state
from cooperative_go2.entities.push_layout import PushLayout
from cooperative_go2.entities.push_observation import PUSH_OBSERVATION_DIM, PlanarBodyState
from cooperative_go2.entities.push_task_config import PushTaskConfig
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_training_settings
from cooperative_go2.interface_adapters.push_env import Go2PushEnv, PushEnvSetup
from cooperative_go2.use_cases.gait_controller import GaitController

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_ENVS = 2
BASE_BOX_XY = (1.0, 0.0)


class CountingPolicy:
    """呼ばれるたびに、全関節で 0.01 ずつ大きくなる行動を返す歩行方策。目標関節角の遅れを見分けるために使う。"""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return torch.full((observation.shape[0], NUM_GO2_JOINTS), 0.01 * self.calls)


class FakePushSimulation:
    """物理を持たず、テストが箱とロボットの状態を直接決めるシミュレーション。適用された目標関節角を記録する。"""

    def __init__(self, gait: GaitConfig) -> None:
        self._default_dof_pos = torch.tensor(gait.robot.default_joint_angles_rad.in_order())
        self._initial = initial_go2_state(
            torch.tensor(gait.robot.base_init_pos_m), torch.tensor([0.0, 0.0, -1.0]), self._default_dof_pos
        )
        self.robot = _planar(0.0, 0.0)
        self.box = _planar(*BASE_BOX_XY)
        self.contact = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.diverged = torch.zeros(NUM_ENVS, dtype=torch.bool)
        self.applied_targets: list[torch.Tensor] = []
        self.reset_masks: list[torch.Tensor] = []

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self._default_dof_pos

    @property
    def initial_go2_state(self) -> Go2State:
        return self._initial

    def read_go2_state(self) -> Go2State:
        # 全環境に共通の 1 行の初期状態を、環境の数の行に広げて返す（Go2 は既定姿勢で立ったまま）
        all_envs = torch.ones(NUM_ENVS, dtype=torch.bool)
        return self._initial.reset_where(all_envs, self._initial)

    def read_robot_planar(self) -> PlanarBodyState:
        return self.robot

    def read_box_planar(self) -> PlanarBodyState:
        return self.box

    def robot_box_contact_mask(self) -> torch.Tensor:
        return self.contact

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        self.applied_targets.append(joint_targets.clone())

    def step(self) -> None:
        pass

    def diverged_mask(self) -> torch.Tensor:
        return self.diverged

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor, layout: PushLayout) -> None:
        self.reset_masks.append(mask.clone())
        rows = mask.unsqueeze(1)
        box_xy = torch.where(rows, layout.box_xy_m, self.box.position_xy_m)
        self.box = dataclasses.replace(self.box, position_xy_m=box_xy)


def _planar(x: float, y: float) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x, y]] * NUM_ENVS),
        yaw_rad=torch.zeros(NUM_ENVS),
        lin_vel_xy_mps=torch.zeros(NUM_ENVS, 2),
        yaw_rate_radps=torch.zeros(NUM_ENVS),
    )


@pytest.fixture
def gait() -> GaitConfig:
    return load_gait_settings(REPO_ROOT / "configs" / "gait.yaml").gait


@pytest.fixture
def task() -> PushTaskConfig:
    loaded = load_push_training_settings(REPO_ROOT / "configs" / "push_single.yaml").task
    # 配置を決定的にし、箱がずれないようにする（報酬と終了判定の検証のため）
    episode = dataclasses.replace(loaded.episode, num_envs=NUM_ENVS)
    reset = dataclasses.replace(loaded.reset, box_xy_jitter_m=0.0)
    return dataclasses.replace(loaded, episode=episode, reset=reset)


class CommandRecordingController:
    """歩行方策に渡された速度指令を記録し、既定姿勢の目標関節角を返す推論器。"""

    def __init__(self, default_dof_pos: torch.Tensor) -> None:
        self._default_dof_pos = default_dof_pos
        self.received_commands: list[torch.Tensor] = []

    def reset_where(self, mask: torch.Tensor) -> None:
        pass

    def compute_joint_targets(self, proprioception: object, commands: torch.Tensor) -> torch.Tensor:
        self.received_commands.append(commands.clone())
        return self._default_dof_pos.expand(NUM_ENVS, -1).clone()


def _setup(task: PushTaskConfig, gait: GaitConfig, command_delay_gait_steps: int) -> PushEnvSetup:
    return PushEnvSetup(
        task=task,
        gait=gait,
        base_box_xy_m=BASE_BOX_XY,
        generator=torch.Generator().manual_seed(0),
        command_delay_gait_steps=command_delay_gait_steps,
    )


def _make_env(task: PushTaskConfig, gait: GaitConfig) -> tuple[Go2PushEnv, FakePushSimulation]:
    simulation = FakePushSimulation(gait)
    controller = GaitController(CountingPolicy(), gait, NUM_ENVS, "cpu")
    return Go2PushEnv(_setup(task, gait, command_delay_gait_steps=0), simulation, controller), simulation


def _make_env_recording_commands(
    task: PushTaskConfig, gait: GaitConfig, command_delay_gait_steps: int
) -> tuple[Go2PushEnv, FakePushSimulation, CommandRecordingController]:
    simulation = FakePushSimulation(gait)
    controller = CommandRecordingController(simulation.default_dof_pos)
    env = Go2PushEnv(_setup(task, gait, command_delay_gait_steps), simulation, controller)  # type: ignore[arg-type]
    return env, simulation, controller


def test_observation_and_action_shapes(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, _ = _make_env(task, gait)

    observations = env.get_observations()["policy"]
    next_observations, reward, done, _ = env.step(torch.zeros(NUM_ENVS, 3))

    assert env.num_actions == 3
    assert observations.shape == (NUM_ENVS, PUSH_OBSERVATION_DIM)
    assert next_observations["policy"].shape == (NUM_ENVS, PUSH_OBSERVATION_DIM)
    assert reward.shape == (NUM_ENVS,)
    assert done.shape == (NUM_ENVS,)


def test_gait_policy_runs_at_gait_rate_within_one_high_level_step(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)

    env.step(torch.zeros(NUM_ENVS, 3))

    assert len(simulation.applied_targets) == task.gait_steps_per_high_level_step(gait.env.control_dt_s)


def test_joint_targets_are_applied_one_gait_step_late_when_latency_is_simulated(
    task: PushTaskConfig, gait: GaitConfig
) -> None:
    assert gait.env.simulate_action_latency
    env, simulation = _make_env(task, gait)
    default = simulation.default_dof_pos
    scale = gait.env.action_scale

    env.step(torch.zeros(NUM_ENVS, 3))

    # 1 回目は既定姿勢（前回の行動が無い）、2 回目は 1 回目に方策が出した行動（0.01）から作った目標を適用する
    assert torch.allclose(simulation.applied_targets[0], default.expand(NUM_ENVS, -1))
    assert torch.allclose(simulation.applied_targets[1], (default + 0.01 * scale).expand(NUM_ENVS, -1))


def test_commands_are_clipped_to_configured_limits(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, _ = _make_env(task, gait)

    outcome = env.step_with_outcome(torch.tensor([[5.0, -5.0, 5.0], [0.1, 0.0, -0.2]]))

    # 速度指令の大きさの報酬は、切り詰めた後の指令から計算される
    limits = task.commands
    clipped = torch.tensor([[limits.vx_mps.high, limits.vy_mps.low, limits.yaw_rate_radps.high], [0.1, 0.0, -0.2]])
    expected_effort = torch.sum(clipped**2, dim=1) * task.reward.scales.control_effort
    assert torch.allclose(outcome.reward_terms["control_effort"], expected_effort)


def test_progress_reward_is_box_approach_to_goal(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    # 目標は箱から +x に 1 m。箱が 0.1 m 目標へ動いたことにする
    simulation.box = dataclasses.replace(
        simulation.box, position_xy_m=simulation.box.position_xy_m + torch.tensor([0.1, 0.0])
    )

    outcome = env.step_with_outcome(torch.zeros(NUM_ENVS, 3))

    expected = torch.full((NUM_ENVS,), 0.1 * task.reward.scales.box_progress)
    assert torch.allclose(outcome.reward_terms["box_progress"], expected, atol=1e-5)


def test_first_step_after_reset_has_no_progress_from_previous_episode(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    # 環境 0 だけ箱を場外へ出してエピソードを終わらせる。リセットで箱は基準の位置に戻る
    outside = torch.tensor([[100.0, 0.0], list(BASE_BOX_XY)])
    simulation.box = dataclasses.replace(simulation.box, position_xy_m=outside)
    first = env.step_with_outcome(torch.zeros(NUM_ENVS, 3))
    assert first.termination_reasons.tolist()[0] == PushTerminationReason.OUT_OF_BOUNDS

    second = env.step_with_outcome(torch.zeros(NUM_ENVS, 3))

    # リセット後の初期距離を前回の距離にしているので、箱が動いていなければ進捗は 0
    assert second.reward_terms["box_progress"].tolist() == pytest.approx([0.0, 0.0], abs=1e-6)


def test_reset_restores_delayed_joint_targets_to_default(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    simulation.box = dataclasses.replace(simulation.box, position_xy_m=torch.tensor([[100.0, 0.0], list(BASE_BOX_XY)]))
    env.step(torch.zeros(NUM_ENVS, 3))
    simulation.applied_targets.clear()

    env.step(torch.zeros(NUM_ENVS, 3))

    # リセットした環境 0 は既定姿勢から、続いている環境 1 は前の行動から作った目標から始まる
    first_applied = simulation.applied_targets[0]
    assert torch.allclose(first_applied[0], simulation.default_dof_pos)
    assert not torch.allclose(first_applied[1], simulation.default_dof_pos)


def test_divergence_zeroes_reward_and_is_recorded_as_reason(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    simulation.diverged = torch.tensor([True, False])
    simulation.box = dataclasses.replace(
        simulation.box, position_xy_m=simulation.box.position_xy_m + torch.tensor([0.1, 0.0])
    )

    outcome = env.step_with_outcome(torch.zeros(NUM_ENVS, 3))

    assert outcome.termination_reasons.tolist()[0] == PushTerminationReason.SIM_DIVERGED
    assert outcome.reward.tolist()[0] == 0.0
    assert outcome.reward.tolist()[1] != 0.0


def test_success_after_holding_box_in_goal(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    goal_x = BASE_BOX_XY[0] + task.reset.goal_distance_m.low
    simulation.box = dataclasses.replace(simulation.box, position_xy_m=torch.tensor([[goal_x, 0.0]] * NUM_ENVS))

    reasons = [
        env.step_with_outcome(torch.zeros(NUM_ENVS, 3)).termination_reasons for _ in range(task.required_hold_steps())
    ]

    assert all(int(reason) == PushTerminationReason.NONE for reason in reasons[-2])
    assert reasons[-1].tolist() == [PushTerminationReason.SUCCESS] * NUM_ENVS


def test_episode_summary_logs_push_reward_terms_and_reasons(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, simulation = _make_env(task, gait)
    simulation.box = dataclasses.replace(simulation.box, position_xy_m=torch.tensor([[100.0, 0.0]] * NUM_ENVS))

    _, _, _, extras = env.step(torch.zeros(NUM_ENVS, 3))

    log = extras["episode"]
    assert "rew_box_progress" in log
    assert log["termination_out_of_bounds"].item() == 1.0


def test_commands_reach_gait_policy_immediately_without_delay(task: PushTaskConfig, gait: GaitConfig) -> None:
    env, _, controller = _make_env_recording_commands(task, gait, command_delay_gait_steps=0)
    commands = torch.tensor([[0.3, 0.0, 0.5]] * NUM_ENVS)

    env.step(commands)

    assert all(torch.equal(received, commands) for received in controller.received_commands)


def test_delayed_commands_reach_gait_policy_after_delay_steps_and_stop_before(
    task: PushTaskConfig, gait: GaitConfig
) -> None:
    delay_steps = 2
    env, _, controller = _make_env_recording_commands(task, gait, command_delay_gait_steps=delay_steps)
    commands = torch.tensor([[0.3, 0.0, 0.5]] * NUM_ENVS)

    env.step(commands)

    received = controller.received_commands
    assert all(torch.equal(command, torch.zeros_like(commands)) for command in received[:delay_steps])
    assert all(torch.equal(command, commands) for command in received[delay_steps:])


def test_reset_discards_delayed_commands_of_previous_episode(task: PushTaskConfig, gait: GaitConfig) -> None:
    delay_steps = 2
    env, simulation, controller = _make_env_recording_commands(task, gait, command_delay_gait_steps=delay_steps)
    commands = torch.tensor([[0.3, 0.0, 0.5]] * NUM_ENVS)
    simulation.diverged = torch.tensor([True, False])
    env.step(commands)
    simulation.diverged = torch.zeros(NUM_ENVS, dtype=torch.bool)
    controller.received_commands.clear()

    env.step(torch.zeros(NUM_ENVS, 3))

    first = controller.received_commands[0]
    assert torch.equal(first[0], torch.zeros(3))
    assert torch.equal(first[1], commands[1])


def test_last_commands_in_observation_are_commands_before_delay(task: PushTaskConfig, gait: GaitConfig) -> None:
    delayed_env, _, _ = _make_env_recording_commands(task, gait, command_delay_gait_steps=3)
    undelayed_env, _, _ = _make_env_recording_commands(task, gait, command_delay_gait_steps=0)
    commands = torch.tensor([[0.3, 0.0, 0.5]] * NUM_ENVS)

    delayed_observation, _, _, _ = delayed_env.step(commands)
    undelayed_observation, _, _, _ = undelayed_env.step(commands)

    assert torch.equal(delayed_observation["policy"], undelayed_observation["policy"])
