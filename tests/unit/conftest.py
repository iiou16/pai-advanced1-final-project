import dataclasses
from collections.abc import Callable
from pathlib import Path

import pytest
import torch
import yaml

from cooperative_go2.entities.corridor import BoxFootprint
from cooperative_go2.entities.go2_state import Go2Proprioception, Go2State, initial_go2_state
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.interface_adapters.config_loader import (
    GaitSettings,
    load_team_marl_run_settings,
    parse_gait_settings,
)
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv, TeamMarlEnvSetup
from cooperative_go2.use_cases.gait_controller import GaitController

REPO_ROOT = Path(__file__).resolve().parents[2]
GAIT_CONFIG_PATH = REPO_ROOT / "configs" / "gait.yaml"
GAIT_EVAL_CONFIG_PATH = REPO_ROOT / "configs" / "gait_eval.yaml"

Go2StateFactory = Callable[..., Go2State]


@pytest.fixture
def gait_config_path() -> Path:
    """リポジトリの configs/gait.yaml。値の検査には使わず、読み込めることの確認だけに使う。"""
    return GAIT_CONFIG_PATH


@pytest.fixture
def gait_eval_config_path() -> Path:
    """リポジトリの configs/gait_eval.yaml（評価シナリオ）。"""
    return GAIT_EVAL_CONFIG_PATH


@pytest.fixture
def gait_config_data() -> dict:
    """configs/gait.yaml を読み込んだ dict。テストごとに新しく読み込むので書き換えてよい。"""
    return yaml.safe_load(GAIT_CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def make_go2_state() -> Go2StateFactory:
    """すべて 0 の状態を基準に、指定したフィールドだけ差し替えた Go2State を作る関数。"""

    def factory(num_envs: int = 1, **overrides: torch.Tensor) -> Go2State:
        known_fields = {
            "base_ang_vel", "projected_gravity", "dof_pos", "dof_vel", "base_pos_m", "base_euler_deg", "base_lin_vel",
        }  # fmt: skip
        unknown = set(overrides) - known_fields
        if unknown:
            raise TypeError(f"Go2State に無いフィールド: {sorted(unknown)}")
        proprioception = Go2Proprioception(
            base_ang_vel=overrides.get("base_ang_vel", torch.zeros(num_envs, 3)),
            projected_gravity=overrides.get("projected_gravity", torch.zeros(num_envs, 3)),
            dof_pos=overrides.get("dof_pos", torch.zeros(num_envs, 12)),
            dof_vel=overrides.get("dof_vel", torch.zeros(num_envs, 12)),
        )
        return Go2State(
            proprioception=proprioception,
            base_pos_m=overrides.get("base_pos_m", torch.zeros(num_envs, 3)),
            base_euler_deg=overrides.get("base_euler_deg", torch.zeros(num_envs, 3)),
            base_lin_vel=overrides.get("base_lin_vel", torch.zeros(num_envs, 3)),
        )

    return factory


SMALL_NUM_ENVS = 2


@pytest.fixture
def small_gait_settings(gait_config_data: dict) -> GaitSettings:
    """2 環境・エピソード 5 ステップ（0.1 s）・前進 0.5 m/s 固定の指令・リセット時のノイズなしにした歩行の設定。"""
    gait_config_data["env"]["num_envs"] = SMALL_NUM_ENVS
    gait_config_data["env"]["control_dt_s"] = 0.02
    gait_config_data["env"]["episode_length_s"] = 0.1
    gait_config_data["command"]["resampling_time_s"] = 0.1
    gait_config_data["command"]["lin_vel_x_mps"] = {"low": 0.5, "high": 0.5}
    gait_config_data["command"]["lin_vel_y_mps"] = {"low": 0.0, "high": 0.0}
    gait_config_data["command"]["yaw_rate_radps"] = {"low": 0.0, "high": 0.0}
    gait_config_data["command"]["zero_lin_vel_probability"] = 0.0
    gait_config_data["command"]["zero_yaw_rate_probability"] = 0.0
    gait_config_data["env"]["reset_joint_pos_noise_rad"] = 0.0
    return parse_gait_settings(gait_config_data)


class FakeGaitSimulation:
    """GaitSimulation の偽物。read_state() は next_state、diverged_mask() は next_diverged を返す。"""

    def __init__(self, num_envs: int) -> None:
        self.default_dof_pos = torch.zeros(12)
        self.initial_state = initial_go2_state(
            base_pos_m=torch.tensor([0.0, 0.0, 0.3]),
            projected_gravity=torch.tensor([0.0, 0.0, -1.0]),
            default_dof_pos=self.default_dof_pos,
        )
        self.next_state = self.initial_state.reset_where(torch.ones(num_envs, dtype=torch.bool), self.initial_state)
        self.next_diverged = torch.zeros(num_envs, dtype=torch.bool)
        self.applied_targets: list[torch.Tensor] = []
        self.reset_masks: list[torch.Tensor] = []
        self.reset_dof_pos: list[torch.Tensor] = []
        self.num_steps = 0

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        self.applied_targets.append(joint_targets)

    def step(self) -> None:
        self.num_steps += 1

    def read_state(self) -> Go2State:
        return self.next_state

    def diverged_mask(self) -> torch.Tensor:
        return self.next_diverged

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor) -> None:
        self.reset_masks.append(mask.clone())
        self.reset_dof_pos.append(dof_pos.clone())


@pytest.fixture
def fake_simulation() -> FakeGaitSimulation:
    return FakeGaitSimulation(SMALL_NUM_ENVS)


TEAM_NUM_ENVS = 2


class FakeTeamSimulation:
    """物理を持たない通路のシーン。テストがロボットと箱の状態を直接決め、物理ステップの中で動かせる。

    step_hook を設定すると、物理ステップのたびに呼ぶ（エピソードの途中で相手の真の状態を動かすために使う）。
    """

    def __init__(self, default_dof_pos: torch.Tensor, num_envs: int = TEAM_NUM_ENVS) -> None:
        self._default_dof_pos = default_dof_pos
        self.num_envs = num_envs
        self.robots = [_team_planar(num_envs, 0.7, 0.25), _team_planar(num_envs, 0.7, -0.25)]
        self.box = _team_planar(num_envs, 1.4, 0.0)
        self.contacts = (torch.zeros(num_envs, dtype=torch.bool), torch.zeros(num_envs, dtype=torch.bool))
        self.euler_deg = [torch.zeros(num_envs, 3), torch.zeros(num_envs, 3)]
        self.step_hook: Callable[[], None] | None = None
        self.reset_calls: list[tuple[torch.Tensor, tuple[torch.Tensor, ...]]] = []
        self.steps = 0

    @property
    def default_dof_pos(self) -> torch.Tensor:
        return self._default_dof_pos

    def read_go2_states(self) -> tuple[Go2State, ...]:
        return tuple(self._go2_state(index) for index in range(2))

    def _go2_state(self, index: int) -> Go2State:
        num_envs = self.num_envs
        proprioception = Go2Proprioception(
            base_ang_vel=torch.zeros(num_envs, 3),
            projected_gravity=torch.tensor([[0.0, 0.0, -1.0]] * num_envs),
            dof_pos=self._default_dof_pos.expand(num_envs, -1).clone(),
            dof_vel=torch.zeros(num_envs, 12),
        )
        position = self.robots[index].position_xy_m
        return Go2State(
            proprioception=proprioception,
            base_pos_m=torch.cat((position, torch.full((num_envs, 1), 0.42)), dim=1),
            base_euler_deg=self.euler_deg[index],
            base_lin_vel=torch.zeros(num_envs, 3),
        )

    def read_robot_planars(self) -> tuple[PlanarBodyState, ...]:
        return tuple(self.robots)

    def read_box_planar(self) -> PlanarBodyState:
        return self.box

    def read_box_footprint(self) -> BoxFootprint:
        return BoxFootprint(center_xy_m=self.box.position_xy_m, yaw_rad=self.box.yaw_rad, half_size_xy_m=(0.2, 0.5))

    def robot_box_contact_masks(self) -> tuple[torch.Tensor, ...]:
        return self.contacts

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        if len(joint_targets) != 2:
            raise ValueError(joint_targets)

    def step(self) -> None:
        self.steps += 1
        if self.step_hook is not None:
            self.step_hook()

    def diverged_mask(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.bool)

    def reset_at(
        self, mask: torch.Tensor, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]
    ) -> None:
        self.reset_calls.append((mask.clone(), tuple(xy.clone() for xy in robot_xy_m)))
        rows = mask.unsqueeze(1)
        for index, xy in enumerate(robot_xy_m):
            robot = self.robots[index]
            self.robots[index] = PlanarBodyState(
                position_xy_m=torch.where(rows, xy, robot.position_xy_m),
                yaw_rad=robot.yaw_rad.masked_fill(mask, 0.0),
                lin_vel_xy_mps=robot.lin_vel_xy_mps.masked_fill(rows, 0.0),
                yaw_rate_radps=robot.yaw_rate_radps.masked_fill(mask, 0.0),
            )
            self.euler_deg[index] = self.euler_deg[index].masked_fill(rows, 0.0)
        self.box = PlanarBodyState(
            position_xy_m=torch.where(rows, torch.tensor([1.4, 0.0]), self.box.position_xy_m),
            yaw_rad=self.box.yaw_rad.masked_fill(mask, 0.0),
            lin_vel_xy_mps=self.box.lin_vel_xy_mps.masked_fill(rows, 0.0),
            yaw_rate_radps=self.box.yaw_rate_radps.masked_fill(mask, 0.0),
        )


def _team_planar(num_envs: int, x_m: float, y_m: float) -> PlanarBodyState:
    return PlanarBodyState(
        position_xy_m=torch.tensor([[x_m, y_m]] * num_envs),
        yaw_rad=torch.zeros(num_envs),
        lin_vel_xy_mps=torch.zeros(num_envs, 2),
        yaw_rate_radps=torch.zeros(num_envs),
    )


class _ZeroGaitPolicy:
    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        return torch.zeros(observation.shape[0], 12)


TeamMarlEnvFactory = Callable[..., tuple["Go2TeamMarlEnv", FakeTeamSimulation]]


@pytest.fixture
def make_team_marl_env() -> TeamMarlEnvFactory:
    """configs/push_multi.yaml の設定で、FakeTeamSimulation を使う協調学習の環境を作る関数。

    引数 seed で乱数のシードを、task_overrides で設定の task の値（dataclasses.replace に渡す値）を変えられる。
    """

    def factory(seed: int = 0, **task_overrides: object) -> tuple[Go2TeamMarlEnv, FakeTeamSimulation]:
        run = load_team_marl_run_settings(REPO_ROOT / "configs" / "push_multi.yaml")
        task = run.settings.task
        task = dataclasses.replace(
            task, episode=dataclasses.replace(task.episode, num_envs=TEAM_NUM_ENVS), **task_overrides
        )
        gait = parse_gait_settings(yaml.safe_load(GAIT_CONFIG_PATH.read_text(encoding="utf-8"))).gait
        default_dof_pos = torch.tensor(gait.robot.default_joint_angles_rad.in_order())
        simulation = FakeTeamSimulation(default_dof_pos)
        controllers = tuple(GaitController(_ZeroGaitPolicy(), gait, TEAM_NUM_ENVS, "cpu") for _ in range(2))
        setup = TeamMarlEnvSetup(
            task=task, gait=gait, scene=run.scene.scene, generator=torch.Generator().manual_seed(seed)
        )
        return Go2TeamMarlEnv(setup, simulation, controllers), simulation

    return factory
