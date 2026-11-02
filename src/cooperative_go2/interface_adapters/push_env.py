"""1 台の箱押しの高レベル方策の学習環境。RSL-RL の VecEnv と同じ属性・メソッドを持ち、OnPolicyRunner にそのまま渡せる。

行動は速度指令 [vx, vy, yaw_rate] で、高レベル方策の 1 ステップの間に、固定した歩行方策を歩行の制御周期で複数回呼ぶ。
観測・報酬・終了判定・配置は entities の部品で計算し、物理はシミュレーションの操作（PushSimulation）に任せる。
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDict

from cooperative_go2.entities.episode_reward_tracker import EpisodeRewardTracker, EpisodeSummary, EpisodeTrackingSpec
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_termination import exceeds_tilt_limit
from cooperative_go2.entities.go2_state import Go2State, sample_reset_dof_pos
from cooperative_go2.entities.push_evaluation import goal_distance_m
from cooperative_go2.entities.push_layout import sample_push_layout
from cooperative_go2.entities.push_observation import PlanarBodyState, PushObservationInputs, build_push_observation
from cooperative_go2.entities.push_reward import PUSH_REWARD_TERM_NAMES, PushRewardInputs, compute_push_reward_terms
from cooperative_go2.entities.push_success import advance_goal_hold_steps, has_held_goal, is_in_goal_region
from cooperative_go2.entities.push_task_config import PushTaskConfig, require_compatible_with_gait
from cooperative_go2.entities.push_termination import (
    PushTerminationFlags,
    PushTerminationReason,
    classify_push_terminations,
    detect_divergence,
    is_outside_arena,
)
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP
from cooperative_go2.use_cases.gait_controller import GaitController


class PushSimulation(Protocol):
    """箱押しの環境が使うシミュレーションの操作。物理ステップを進めるのはこの操作だけ。"""

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (12,) [rad]。"""
        ...

    @property
    def initial_go2_state(self) -> Go2State:
        """リセット直後の Go2 の状態（全環境に共通の 1 行）。"""
        ...

    def read_go2_state(self) -> Go2State:
        """Go2 の現在の状態を返す。"""
        ...

    def read_robot_planar(self) -> PlanarBodyState:
        """Go2 の胴体の水平面の状態（ワールド座標系）を返す。"""
        ...

    def read_box_planar(self) -> PlanarBodyState:
        """箱の水平面の状態（ワールド座標系）を返す。"""
        ...

    def robot_box_contact_mask(self) -> torch.Tensor:
        """Go2 が箱に触れている環境のマスク (num_envs,) を返す。"""
        ...

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。"""
        ...

    def step(self) -> None:
        """物理を歩行の 1 制御周期進める。

        Raises:
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        ...

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで物理計算が発散した環境のマスク (num_envs,) を返す。"""
        ...

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor, box_xy_m: torch.Tensor) -> None:
        """mask が True の環境の Go2 を初期位置・関節角 dof_pos で、箱を box_xy_m (num_envs, 2) に水平に置く。"""
        ...


@dataclass(frozen=True)
class PushEnvSetup:
    """箱押しの環境の設定一式。

    Attributes:
        task: 箱押しの学習環境の設定。
        gait: 固定して使う歩行方策を学習したときの設定（学習の出力先に複製されたもの）。
        base_box_xy_m: 箱を置く基準の位置 (x, y) [m]（シーンの設定の箱の初期位置）。
        generator: 配置とリセット時の関節角のばらつきを引く乱数生成器。シードを固定して渡す。
    """

    task: PushTaskConfig
    gait: GaitConfig
    base_box_xy_m: tuple[float, ...]
    generator: torch.Generator


@dataclass(frozen=True)
class PushStepOutcome:
    """高レベル方策の 1 ステップの結果。

    Attributes:
        observation: 次の行動を決めるための観測。終了した環境はリセット後の観測。
        reward: 報酬 (num_envs,)。物理計算が発散した環境は 0。
        done: エピソードが終わった環境のマスク (num_envs,)。
        termination_reasons: 環境ごとの終了理由 (num_envs,)。PushTerminationReason の値。
        reward_terms: 係数を掛けた報酬の項の名前 → 値 (num_envs,)。
        episode_summary: このステップで終わったエピソードの集計。終わった環境が無ければ None。
    """

    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    termination_reasons: torch.Tensor
    reward_terms: dict[str, torch.Tensor]
    episode_summary: EpisodeSummary | None


class Go2PushEnv:
    """1 台の Go2 が箱を目標領域へ押す、高レベル方策の並列環境。

    リセットでは、Go2 を歩行方策の学習時と同じ初期位置・関節角のばらつきで置き、箱と目標を sample_push_layout で置く。
    歩行方策の学習時に行動を 1 制御周期遅らせていた場合（env.simulate_action_latency）は、ここでも目標関節角を
    1 制御周期遅らせて適用する。歩行方策が学習したときと同じ遅れで動かすため。
    S1 には壁も他のロボットも無いので、報酬の unsafe_collision は常に 0 にする。

    num_envs・num_actions・max_episode_length・episode_length_buf・device・cfg は RSL-RL の VecEnv の契約。
    """

    def __init__(self, setup: PushEnvSetup, simulation: PushSimulation, controller: GaitController) -> None:
        """設定を歩行方策の条件と突き合わせ、全環境をリセットして最初の観測を用意する。

        Args:
            setup: 箱押しの環境の設定一式。
            simulation: setup.task.episode.num_envs 個の並列環境で構築したシミュレーションの操作。
            controller: setup.gait で学習した歩行方策の推論器。ロボットの数は num_envs。

        Raises:
            InvalidPushConfigError: 箱押しの設定が歩行方策の学習時の設定と矛盾する。
        """
        require_compatible_with_gait(setup.task, setup.gait)
        task = setup.task
        self.cfg = task
        self.num_envs = task.episode.num_envs
        self.num_actions = VELOCITY_COMMAND_SIZE
        self.device = setup.generator.device
        self.max_episode_length = task.episode.max_episode_length_steps()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._setup = setup
        self._simulation = simulation
        self._controller = controller
        self._gait_steps = task.gait_steps_per_high_level_step(setup.gait.env.control_dt_s)
        self._required_hold_steps = task.required_hold_steps()
        limits = task.commands.as_ranges()
        self._command_low = torch.tensor([limit.low for limit in limits], device=self.device)
        self._command_high = torch.tensor([limit.high for limit in limits], device=self.device)
        tracking_spec = EpisodeTrackingSpec(
            PUSH_REWARD_TERM_NAMES, PushTerminationReason, task.episode.high_level_period_s
        )
        self._rewards = EpisodeRewardTracker(tracking_spec, self.num_envs, self.device)
        self._go2_state = simulation.initial_go2_state
        self._robot = simulation.read_robot_planar()
        self._box = simulation.read_box_planar()
        self._box_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._goal_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._previous_box_yaw = torch.zeros(self.num_envs, device=self.device)
        self._last_commands = torch.zeros((self.num_envs, VELOCITY_COMMAND_SIZE), device=self.device)
        self._pending_joint_targets = simulation.default_dof_pos.expand(self.num_envs, -1).clone()
        self.reset()

    def get_observations(self) -> TensorDict:
        """現在の観測を RSL-RL の観測グループの形で返す。"""
        return TensorDict({POLICY_OBSERVATION_GROUP: self._observation}, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:
        """全環境をリセットし、観測を返す。"""
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._observation = self._observe()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        """RSL-RL の VecEnv.step。step_with_outcome() の結果を RSL-RL の形に変換する。"""
        outcome = self.step_with_outcome(actions)
        time_out = outcome.termination_reasons == int(PushTerminationReason.TIME_OUT)
        extras: dict[str, object] = {"time_outs": time_out.to(torch.float32)}
        if outcome.episode_summary is not None:
            extras["episode"] = outcome.episode_summary.as_log()
        return self.get_observations(), outcome.reward, outcome.done, extras

    def step_with_outcome(self, actions: torch.Tensor) -> PushStepOutcome:
        """速度指令を高レベル方策の 1 周期のあいだ歩行方策に与えて物理を進め、終了した環境をリセットする。

        Args:
            actions: 速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
                設定の範囲（commands）に切り詰めてから使う。

        Raises:
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        commands = torch.clamp(actions, self._command_low, self._command_high)
        diverged = self._walk(commands)
        self._robot = self._simulation.read_robot_planar()
        self._box = self._simulation.read_box_planar()
        self._box_contact = self._simulation.robot_box_contact_mask()
        diverged = diverged | detect_divergence(
            (self._robot.position_xy_m, self._box.position_xy_m, self._go2_state.base_pos_m),
            (self._robot.lin_vel_xy_mps, self._box.lin_vel_xy_mps),
            self.cfg.divergence,
        )
        self.episode_length_buf += 1
        reasons = self._classify_terminations(diverged)
        reward_terms = self._weighted_reward_terms(commands, reasons, diverged)
        self._rewards.add_step(reward_terms)
        done = reasons != int(PushTerminationReason.NONE)
        episode_summary = self._rewards.summarize_completed_or_none(done, reasons)
        self._previous_goal_distance = goal_distance_m(self._box.position_xy_m, self._goal_xy)
        self._previous_box_yaw = self._box.yaw_rad
        self._last_commands = commands
        self._reset_envs(done)
        self._observation = self._observe()
        return PushStepOutcome(
            observation=self._observation,
            reward=torch.stack(list(reward_terms.values())).sum(dim=0),
            done=done,
            termination_reasons=reasons,
            reward_terms=reward_terms,
            episode_summary=episode_summary,
        )

    def _walk(self, commands: torch.Tensor) -> torch.Tensor:
        # 歩行方策を歩行の制御周期で呼び、発散した環境のマスクを返す
        diverged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        simulate_latency = self._setup.gait.env.simulate_action_latency
        for _ in range(self._gait_steps):
            joint_targets = self._controller.compute_joint_targets(self._go2_state.proprioception, commands)
            if simulate_latency:
                self._simulation.apply_joint_targets(self._pending_joint_targets)
                self._pending_joint_targets = joint_targets
            else:
                self._simulation.apply_joint_targets(joint_targets)
            self._simulation.step()
            diverged = diverged | self._simulation.diverged_mask()
            self._go2_state = self._simulation.read_go2_state()
        return diverged

    def _classify_terminations(self, diverged: torch.Tensor) -> torch.Tensor:
        in_goal = is_in_goal_region(self._box.position_xy_m, self._goal_xy, self.cfg.success.goal_radius_m)
        self._hold_steps = advance_goal_hold_steps(self._hold_steps, in_goal)
        episode = self.cfg.episode
        arena = self.cfg.arena
        self._fallen = exceeds_tilt_limit(self._go2_state.base_euler_deg, episode.fall_roll_deg, episode.fall_pitch_deg)
        out_of_bounds = is_outside_arena(self._robot.position_xy_m, arena) | is_outside_arena(
            self._box.position_xy_m, arena
        )
        flags = PushTerminationFlags(
            time_out=self.episode_length_buf >= self.max_episode_length,
            succeeded=has_held_goal(self._hold_steps, self._required_hold_steps),
            fallen=self._fallen,
            out_of_bounds=out_of_bounds,
            diverged=diverged,
        )
        return classify_push_terminations(flags)

    def _weighted_reward_terms(
        self, commands: torch.Tensor, reasons: torch.Tensor, diverged: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        inputs = PushRewardInputs(
            previous_goal_distance_m=self._previous_goal_distance,
            goal_distance_m=goal_distance_m(self._box.position_xy_m, self._goal_xy),
            termination_reasons=reasons,
            box_contact=self._box_contact,
            robot_xy_m=self._robot.position_xy_m,
            box_xy_m=self._box.position_xy_m,
            goal_xy_m=self._goal_xy,
            previous_box_yaw_rad=self._previous_box_yaw,
            box_yaw_rad=self._box.yaw_rad,
            # 転倒の減点は、同じステップでシーン外への離脱などが優先されても与える（転倒したこと自体は変わらないため）
            fallen=self._fallen,
            unsafe_collision=torch.zeros_like(self._box_contact),
            commands=commands,
        )
        terms = compute_push_reward_terms(inputs, self.cfg.reward)
        # 発散した環境の状態は NaN になりうる。学習を止めずにそのエピソードだけを失敗として終えるため、報酬を 0 にする
        return terms.weighted(self.cfg.reward.scales).zeroed_where(diverged).named()

    def _reset_envs(self, mask: torch.Tensor) -> None:
        setup = self._setup
        layout = sample_push_layout(setup.base_box_xy_m, self.cfg.reset, self.num_envs, setup.generator)
        reset_dof_pos = sample_reset_dof_pos(
            self._simulation.default_dof_pos, self.num_envs, setup.gait.env.reset_joint_pos_noise_rad, setup.generator
        )
        self._simulation.reset(mask, reset_dof_pos, layout.box_xy_m)
        self._controller.reset_where(mask)
        self._go2_state = self._go2_state.reset_where(
            mask, self._simulation.initial_go2_state.with_dof_pos(reset_dof_pos)
        )
        # 前のステップの値（箱と目標の距離・箱の向き・指令・遅らせて適用する目標関節角）も新しいエピソードの値にする。
        # 残すと、リセット直後のステップで前のエピソードとの差が報酬になる
        rows = mask.unsqueeze(1)
        self._goal_xy = torch.where(rows, layout.goal_xy_m, self._goal_xy)
        self._previous_goal_distance = torch.where(
            mask, goal_distance_m(layout.box_xy_m, layout.goal_xy_m), self._previous_goal_distance
        )
        self._previous_box_yaw = self._previous_box_yaw.masked_fill(mask, 0.0)
        self._last_commands = self._last_commands.masked_fill(rows, 0.0)
        self._pending_joint_targets = torch.where(rows, self._simulation.default_dof_pos, self._pending_joint_targets)
        self._hold_steps = self._hold_steps.masked_fill(mask, 0)
        self._box_contact = self._box_contact & ~mask
        self._robot = self._robot_at_reset(mask)
        self._box = self._box_at_reset(mask, layout.box_xy_m)
        self._rewards.reset_where(mask)
        self.episode_length_buf = self.episode_length_buf.masked_fill(mask, 0)

    def _robot_at_reset(self, mask: torch.Tensor) -> PlanarBodyState:
        # リセットした環境の Go2 は、初期位置で初期の向き（ヨー 0）のまま静止している
        initial_xy = self._simulation.initial_go2_state.base_pos_m[:, :2]
        return _replace_planar_rows(self._robot, mask, initial_xy)

    def _box_at_reset(self, mask: torch.Tensor, box_xy_m: torch.Tensor) -> PlanarBodyState:
        return _replace_planar_rows(self._box, mask, box_xy_m)

    def _observe(self) -> torch.Tensor:
        inputs = PushObservationInputs(
            robot=self._robot,
            robot_projected_gravity=self._go2_state.proprioception.projected_gravity,
            box=self._box,
            goal_xy_m=self._goal_xy,
            box_contact=self._box_contact,
            last_commands=self._last_commands,
        )
        return build_push_observation(inputs, self.cfg.observation)


def _replace_planar_rows(state: PlanarBodyState, mask: torch.Tensor, position_xy_m: torch.Tensor) -> PlanarBodyState:
    # mask の行を、position_xy_m の位置にヨー 0 で静止した状態に置き換える
    rows = mask.unsqueeze(1)
    return PlanarBodyState(
        position_xy_m=torch.where(rows, position_xy_m.expand_as(state.position_xy_m), state.position_xy_m),
        yaw_rad=state.yaw_rad.masked_fill(mask, 0.0),
        lin_vel_xy_mps=state.lin_vel_xy_mps.masked_fill(rows, 0.0),
        yaw_rate_radps=state.yaw_rate_radps.masked_fill(mask, 0.0),
    )
