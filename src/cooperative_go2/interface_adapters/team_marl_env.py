"""2 台の協調学習（IPPO・MAPPO）の並列環境。通路のシーン S3_team で、2 つの Actor を同時に動かす。

行動はロボットごとの速度指令 [vx, vy, yaw_rate] で、高レベル方策の 1 ステップの間に、固定した歩行方策を
ロボットごとに歩行の制御周期で複数回呼ぶ。観測・状態・報酬・終了判定・通信の模擬は entities の部品で計算し、
物理はシミュレーションの操作（TeamSimulation）に任せる。学習ライブラリ（skrl）の形への変換は skrl_team_env が担う。
"""

from dataclasses import dataclass
from typing import Protocol

import torch

from cooperative_go2.entities.corridor import BoxFootprint, is_corridor_open
from cooperative_go2.entities.corridor_scene_config import CorridorSceneConfig
from cooperative_go2.entities.episode_reward_tracker import EpisodeRewardTracker, EpisodeSummary, EpisodeTrackingSpec
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_termination import exceeds_tilt_limit
from cooperative_go2.entities.go2_state import Go2State, sample_reset_dof_pos
from cooperative_go2.entities.partner_channel import DelayedPartnerChannel, PartnerMessage
from cooperative_go2.entities.push_evaluation import goal_distance_m
from cooperative_go2.entities.push_observation import PlanarBodyState
from cooperative_go2.entities.push_task_config import require_high_level_compatible_with_gait
from cooperative_go2.entities.push_termination import PushTerminationReason, detect_divergence, is_outside_arena
from cooperative_go2.entities.team_marl_config import TeamMarlTaskConfig
from cooperative_go2.entities.team_marl_judge import TeamJudgeInputs, TeamJudgement, TeamJudgeRules, judge_team_step
from cooperative_go2.entities.team_marl_observation import (
    TeamActorObservationInputs,
    TeamRobotTruth,
    TeamStateInputs,
    build_team_actor_observation,
    build_team_state,
)
from cooperative_go2.entities.team_marl_reward import (
    TEAM_MARL_REWARD_TERM_NAMES,
    TeamMarlRewardInputs,
    TeamMarlRewardTerms,
    compute_team_marl_reward_terms,
    push_position_distances_m,
)
from cooperative_go2.entities.team_task_config import sample_team_start_positions
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.use_cases.gait_controller import GaitController

ROLL_AXIS, PITCH_AXIS = 0, 1
TEAM_SIZE = 2


class TeamSimulation(Protocol):
    """協調学習の環境が使う、通路のシーンの操作。物理ステップを進めるのはこの操作だけ。ロボットの値はシーンの robots の順。"""

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """Go2 の既定姿勢の関節角 (12,) [rad]。"""
        ...

    def read_go2_states(self) -> tuple[Go2State, ...]:
        """各 Go2 の現在の状態を返す。"""
        ...

    def read_robot_planars(self) -> tuple[PlanarBodyState, ...]:
        """各 Go2 の胴体の水平面の状態（ワールド座標系）を返す。"""
        ...

    def read_box_planar(self) -> PlanarBodyState:
        """箱の水平面の状態（ワールド座標系）を返す。"""
        ...

    def read_box_footprint(self) -> BoxFootprint:
        """箱を真上から見た長方形を返す。"""
        ...

    def robot_box_contact_masks(self) -> tuple[torch.Tensor, ...]:
        """各 Go2 が箱に触れている環境のマスク (num_envs,) を返す。"""
        ...

    def apply_joint_targets(self, joint_targets: tuple[torch.Tensor, ...]) -> None:
        """各 Go2 の PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。"""
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

    def reset_at(
        self, mask: torch.Tensor, dof_pos: tuple[torch.Tensor, ...], robot_xy_m: tuple[torch.Tensor, ...]
    ) -> None:
        """mask の環境の各 Go2 を位置 robot_xy_m (num_envs, 2)・関節角 dof_pos で、箱を初期位置に置く。"""
        ...


class TeamSizeError(ValueError):
    """シーンのロボットの数が、協調学習の前提（2 台）や設定の agent_side_signs の数と合わないことを表す。"""


@dataclass(frozen=True)
class TeamMarlEnvSetup:
    """協調学習の環境の設定一式。

    Attributes:
        task: 協調学習の環境の設定。
        gait: 固定して使う歩行方策を学習したときの設定（学習の出力先に複製されたもの）。
        scene: 通路のシーンの配置（ロボットの初期位置・退避領域・通路）。
        generator: 開始位置・リセット時の関節角のばらつき・通信の欠落を引く乱数生成器。シードを固定して渡す。
    """

    task: TeamMarlTaskConfig
    gait: GaitConfig
    scene: CorridorSceneConfig
    generator: torch.Generator


@dataclass(frozen=True)
class TeamStepSnapshot:
    """物理を進めた直後、リセットする前の状態。終わった試行の最終状態や、評価の指標の記録に使う。

    Attributes:
        robots: ロボットごとの胴体の水平面の状態。
        box_footprint: 箱を真上から見た長方形。
        box_contacts: ロボットごとの、箱に接触している環境のマスク (num_envs,)。
        fallen: ロボットごとの、このステップで転倒の傾きを超えた環境のマスク (num_envs,)。
        corridor_open: 通路が開いている（箱が通行領域と重ならない）環境のマスク (num_envs,)。
    """

    robots: tuple[PlanarBodyState, ...]
    box_footprint: BoxFootprint
    box_contacts: tuple[torch.Tensor, ...]
    fallen: tuple[torch.Tensor, ...]
    corridor_open: torch.Tensor


@dataclass(frozen=True)
class TeamStepOutcome:
    """高レベル方策の 1 ステップの結果。

    Attributes:
        observations: ロボットごとの、次の行動を決めるための Actor の観測。終了した環境はリセット後の観測。
        states: ロボットごとの、Critic の状態。終了した環境はリセット後の状態。
        rewards: ロボットごとの報酬 (ロボットの数, num_envs)。物理計算が発散した環境は 0。
        done: エピソードが終わった環境のマスク (num_envs,)。2 台で共有する。
        termination_reasons: 環境ごとの終了理由 (num_envs,)。PushTerminationReason の値。
        reward_terms: 係数を掛けた報酬の項。
        episode_summary: このステップで終わったエピソードの集計。終わった環境が無ければ None。
        pre_reset: 物理を進めた直後、リセットする前の状態。
    """

    observations: tuple[torch.Tensor, ...]
    states: tuple[torch.Tensor, ...]
    rewards: torch.Tensor
    done: torch.Tensor
    termination_reasons: torch.Tensor
    reward_terms: TeamMarlRewardTerms
    episode_summary: EpisodeSummary | None
    pre_reset: TeamStepSnapshot


class Go2TeamMarlEnv:
    """2 台の Go2 が重い箱を通路の出口の先の退避領域へ押す、協調学習の並列環境。

    リセットでは、規則ベースの協調と同じ引き方でロボットの開始位置（押す位置の後ろへ散らす）と関節角の
    ばらつきを引く。同じシード・デバイスで最初のリセットを行えば、P4 の並列環境の確認と同じ開始位置になる。
    歩行方策の学習時に行動を 1 制御周期遅らせていた場合は、ここでも目標関節角を 1 制御周期遅らせて適用する。
    相手の状態は DelayedPartnerChannel を通して、設定の遅れと欠落つきで Actor の観測に入れる。
    """

    def __init__(
        self, setup: TeamMarlEnvSetup, simulation: TeamSimulation, controllers: tuple[GaitController, ...]
    ) -> None:
        """設定を歩行方策の条件と突き合わせ、全環境をリセットして最初の観測を用意する。

        Args:
            setup: 協調学習の環境の設定一式。
            simulation: setup.task.episode.num_envs 個の並列環境で構築した通路のシーンの操作。
            controllers: ロボットごとの、setup.gait で学習した歩行方策の推論器（ロボットの数は num_envs）。

        Raises:
            InvalidPushConfigError: 周期・指令の範囲・転倒の判定が歩行方策の学習時の設定と矛盾する。
            TeamSizeError: シーンのロボット・推論器・agent_side_signs の数が 2 でない。
        """
        task = setup.task
        require_high_level_compatible_with_gait(task.episode, task.commands, setup.gait)
        counts = {"scene.robots": len(setup.scene.robots), "controllers": len(controllers),
                  "agent_side_signs": task.num_agents}  # fmt: skip
        if set(counts.values()) != {TEAM_SIZE}:
            raise TeamSizeError(f"協調学習はロボット {TEAM_SIZE} 台で動かす: {counts}")
        self.cfg = task
        self.num_envs = task.episode.num_envs
        self.num_agents = TEAM_SIZE
        self.device = setup.generator.device
        self.max_episode_length = task.episode.max_episode_length_steps()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._setup = setup
        self._simulation = simulation
        self._controllers = controllers
        self._period_s = task.episode.high_level_period_s
        self._gait_steps = round(self._period_s / setup.gait.env.control_dt_s)
        limits = task.commands.as_ranges()
        self.command_low = torch.tensor([limit.low for limit in limits], device=self.device)
        self.command_high = torch.tensor([limit.high for limit in limits], device=self.device)
        retreat = setup.scene.retreat_area
        self._goal_xy = torch.tensor(retreat.center_xy_m, device=self.device).expand(self.num_envs, -1)
        self._judge_rules = TeamJudgeRules(
            retreat_xy_m=(retreat.center_xy_m[0], retreat.center_xy_m[1]),
            retreat_radius_m=retreat.radius_m,
            required_hold_steps=task.required_hold_steps(),
            max_episode_steps=self.max_episode_length,
            arena=task.arena,
        )
        tracking_spec = EpisodeTrackingSpec(TEAM_MARL_REWARD_TERM_NAMES, PushTerminationReason, self._period_s)
        self._rewards = EpisodeRewardTracker(tracking_spec, self.num_envs, self.device)
        delay_steps = task.communication_delay_steps()
        # _channels[i] は、ロボット i が相手から受け取る通信
        self._channels = tuple(
            DelayedPartnerChannel(task.communication, delay_steps, self.num_envs, self.device) for _ in range(TEAM_SIZE)
        )
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._hold_steps = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._previous_goal_distance = zeros.clone()
        self._previous_box_yaw = zeros.clone()
        self._previous_push_distances = (zeros.clone(), zeros.clone())
        no_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._box_contacts = (no_envs, no_envs)
        self._last_commands = tuple(
            torch.zeros((self.num_envs, VELOCITY_COMMAND_SIZE), device=self.device) for _ in range(TEAM_SIZE)
        )
        self._pending_joint_targets = tuple(
            simulation.default_dof_pos.expand(self.num_envs, -1).clone() for _ in range(TEAM_SIZE)
        )
        self.reset()

    @property
    def gait_steps_per_high_level_step(self) -> int:
        """高レベル方策の 1 ステップで、歩行方策を呼んで物理を進める回数。"""
        return self._gait_steps

    @property
    def goal_xy_m(self) -> torch.Tensor:
        """退避領域の中心 (num_envs, 2)、ワールド座標系 [m]。"""
        return self._goal_xy

    @property
    def box_planar(self) -> PlanarBodyState:
        """現在の箱の水平面の状態（ワールド座標系）。"""
        return self._box

    @property
    def observations(self) -> tuple[torch.Tensor, ...]:
        """ロボットごとの現在の Actor の観測 (num_envs, TEAM_ACTOR_OBSERVATION_DIM)。"""
        return self._observations

    @property
    def states(self) -> tuple[torch.Tensor, ...]:
        """ロボットごとの現在の Critic の状態 (num_envs, team_state_dim(2))。"""
        return self._states

    def reset(self) -> tuple[torch.Tensor, ...]:
        """全環境をリセットし、ロボットごとの Actor の観測を返す。"""
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._observe()
        return self._observations

    def step(self, actions: tuple[torch.Tensor, ...]) -> TeamStepOutcome:
        """ロボットごとの速度指令を高レベル方策の 1 周期のあいだ歩行方策に与えて物理を進め、終了した環境をリセットする。

        Args:
            actions: ロボットごとの速度指令 (num_envs, 3)。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
                設定の範囲（commands）に切り詰めてから使う。

        Raises:
            TeamSizeError: actions の数がロボットの数と違う。
            SimulationCapacityExceededError: 衝突候補などの数が設定の上限を超えた。
        """
        if len(actions) != TEAM_SIZE:
            raise TeamSizeError(f"ロボット {TEAM_SIZE} 台分の行動が必要: {len(actions)} 個")
        commands = tuple(torch.clamp(action, self.command_low, self.command_high) for action in actions)
        stamp_s = self.episode_length_buf.to(torch.float32) * self._period_s
        sent = tuple(
            PartnerMessage(position_xy_m=robot.position_xy_m, yaw_rad=robot.yaw_rad, command=command, stamp_s=stamp_s)
            for robot, command in zip(self._robots, commands, strict=True)
        )
        diverged, fallen = self._walk(commands)
        simulation = self._simulation
        self._robots = simulation.read_robot_planars()
        self._box = simulation.read_box_planar()
        self._box_contacts = simulation.robot_box_contact_masks()
        go2_states = simulation.read_go2_states()
        self._projected_gravity = tuple(state.proprioception.projected_gravity for state in go2_states)
        diverged = diverged | detect_divergence(
            (self._box.position_xy_m, *(robot.position_xy_m for robot in self._robots),
             *(state.base_pos_m for state in go2_states)),
            (self._box.lin_vel_xy_mps, *(robot.lin_vel_xy_mps for robot in self._robots)),
            self.cfg.divergence,
        )  # fmt: skip
        self.episode_length_buf += 1
        robots_xy = tuple(robot.position_xy_m for robot in self._robots)
        judgement = judge_team_step(
            TeamJudgeInputs(
                robots_xy_m=robots_xy,
                box_xy_m=self._box.position_xy_m,
                fallen=fallen,
                episode_steps=self.episode_length_buf,
                hold_steps=self._hold_steps,
                diverged=diverged,
            ),
            self._judge_rules,
        )
        self._hold_steps = judgement.hold_steps
        reasons = judgement.reasons
        goal_distance = goal_distance_m(self._box.position_xy_m, self._goal_xy)
        push_distances = self._push_distances()
        reward_terms = self._weighted_reward_terms(commands, goal_distance, push_distances, fallen, judgement)
        self._rewards.add_step(reward_terms.robot_mean())
        done = reasons != int(PushTerminationReason.NONE)
        episode_summary = self._rewards.summarize_completed_or_none(done, reasons)
        # ロボット i は相手（1 - i）が配信した状態を受け取る
        for receiver, channel in enumerate(self._channels):
            channel.transmit(sent[1 - receiver], self._setup.generator)
        self._previous_goal_distance = goal_distance
        self._previous_box_yaw = self._box.yaw_rad
        self._previous_push_distances = push_distances
        self._last_commands = commands
        footprint = simulation.read_box_footprint()
        pre_reset = TeamStepSnapshot(
            robots=self._robots,
            box_footprint=footprint,
            box_contacts=self._box_contacts,
            fallen=fallen,
            corridor_open=is_corridor_open(footprint, self._setup.scene.corridor),
        )
        self._reset_envs(done)
        self._observe()
        return TeamStepOutcome(
            observations=self._observations,
            states=self._states,
            rewards=reward_terms.per_robot_total(),
            done=done,
            termination_reasons=reasons,
            reward_terms=reward_terms,
            episode_summary=episode_summary,
            pre_reset=pre_reset,
        )

    def _walk(self, commands: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        # 歩行方策を歩行の制御周期で呼び、発散した環境のマスクと、ロボットごとの転倒の傾きを超えた環境のマスクを返す。
        # 転倒は物理ステップごとに調べる。規則ベースの協調の並列環境での確認と同じく、一時的に大きく傾いた場合も数えるため
        simulation = self._simulation
        diverged = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        fallen = [torch.zeros_like(diverged) for _ in range(TEAM_SIZE)]
        simulate_latency = self._setup.gait.env.simulate_action_latency
        tilt = self.cfg.episode
        for _ in range(self._gait_steps):
            states = simulation.read_go2_states()
            targets = tuple(
                controller.compute_joint_targets(state.proprioception, command)
                for controller, state, command in zip(self._controllers, states, commands, strict=True)
            )
            for index, state in enumerate(states):
                fallen[index] |= exceeds_tilt_limit(state.base_euler_deg, tilt.fall_roll_deg, tilt.fall_pitch_deg)
            if simulate_latency:
                simulation.apply_joint_targets(self._pending_joint_targets)
                self._pending_joint_targets = targets
            else:
                simulation.apply_joint_targets(targets)
            simulation.step()
            diverged = diverged | simulation.diverged_mask()
        for index, state in enumerate(simulation.read_go2_states()):
            fallen[index] |= exceeds_tilt_limit(state.base_euler_deg, tilt.fall_roll_deg, tilt.fall_pitch_deg)
        return diverged, tuple(fallen)

    def _weighted_reward_terms(
        self, commands: tuple[torch.Tensor, ...], goal_distance: torch.Tensor, push_distances: tuple[torch.Tensor, ...],
        fallen: tuple[torch.Tensor, ...], judgement: TeamJudgement,
    ) -> TeamMarlRewardTerms:  # fmt: skip
        reasons = judgement.reasons
        box_outside = is_outside_arena(self._box.position_xy_m, self.cfg.arena)
        inputs = TeamMarlRewardInputs(
            previous_goal_distance_m=self._previous_goal_distance,
            goal_distance_m=goal_distance,
            succeeded=reasons == int(PushTerminationReason.SUCCESS),
            previous_box_yaw_rad=self._previous_box_yaw,
            box_yaw_rad=self._box.yaw_rad,
            box_xy_m=self._box.position_xy_m,
            goal_xy_m=self._goal_xy,
            robots_xy_m=tuple(robot.position_xy_m for robot in self._robots),
            previous_push_distances_m=self._previous_push_distances,
            push_distances_m=push_distances,
            box_contacts=self._box_contacts,
            fallen=fallen,
            out_of_bounds=tuple(
                is_outside_arena(robot.position_xy_m, self.cfg.arena) | box_outside for robot in self._robots
            ),
            commands=commands,
        )
        terms = compute_team_marl_reward_terms(inputs, self.cfg.reward)
        # 発散した環境の状態は NaN になりうる。学習を止めずにそのエピソードだけを失敗として終えるため、報酬を 0 にする
        return terms.weighted(self.cfg.reward.scales).zeroed_where(judgement.flags.diverged)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        setup = self._setup
        simulation = self._simulation
        # 規則ベースの協調の並列環境での確認と同じ順（開始位置 → ロボットごとの関節角）で引く。同じシードの最初のリセットで、
        # 同じ開始位置・関節角になるようにするため
        initial_xy = tuple(tuple(robot.initial_xy_m) for robot in setup.scene.robots)
        start_xy = sample_team_start_positions(initial_xy, self.cfg.reset, self.num_envs, setup.generator)
        noise_rad = setup.gait.env.reset_joint_pos_noise_rad
        dof_pos = tuple(
            sample_reset_dof_pos(simulation.default_dof_pos, self.num_envs, noise_rad, setup.generator)
            for _ in range(TEAM_SIZE)
        )
        simulation.reset_at(mask, dof_pos, start_xy)
        for controller in self._controllers:
            controller.reset_where(mask)
        for channel in self._channels:
            channel.reset_where(mask)
        rows = mask.unsqueeze(1)
        self._last_commands = tuple(command.masked_fill(rows, 0.0) for command in self._last_commands)
        self._pending_joint_targets = tuple(
            torch.where(rows, simulation.default_dof_pos, targets) for targets in self._pending_joint_targets
        )
        self._hold_steps = self._hold_steps.masked_fill(mask, 0)
        self._box_contacts = tuple(contact & ~mask for contact in self._box_contacts)
        self._rewards.reset_where(mask)
        self.episode_length_buf = self.episode_length_buf.masked_fill(mask, 0)
        # リセットした環境は置き直した位置から読み直す。前のエピソードとの差が報酬にならないよう、前の値も揃える
        self._robots = simulation.read_robot_planars()
        self._box = simulation.read_box_planar()
        self._projected_gravity = tuple(
            state.proprioception.projected_gravity for state in simulation.read_go2_states()
        )
        self._previous_goal_distance = torch.where(
            mask, goal_distance_m(self._box.position_xy_m, self._goal_xy), self._previous_goal_distance
        )
        self._previous_box_yaw = torch.where(mask, self._box.yaw_rad, self._previous_box_yaw)
        self._previous_push_distances = tuple(
            torch.where(mask, current, previous)
            for current, previous in zip(self._push_distances(), self._previous_push_distances, strict=True)
        )

    def _push_distances(self) -> tuple[torch.Tensor, ...]:
        return push_position_distances_m(
            tuple(robot.position_xy_m for robot in self._robots), self._box.position_xy_m, self._goal_xy,
            self.cfg.agent_side_signs, self.cfg.reward,
        )  # fmt: skip

    def _observe(self) -> None:
        task = self.cfg
        now_s = self.episode_length_buf.to(torch.float32) * self._period_s
        self._observations = tuple(
            build_team_actor_observation(
                TeamActorObservationInputs(
                    robot=self._robots[index],
                    robot_projected_gravity=self._projected_gravity[index],
                    box=self._box,
                    goal_xy_m=self._goal_xy,
                    last_commands=self._last_commands[index],
                    side_sign=task.agent_side_signs[index],
                    partner=self._channels[index].reception(now_s),
                ),
                task.observation,
                task.communication.max_age_s,
            )
            for index in range(TEAM_SIZE)
        )
        truth = TeamStateInputs(
            robots=tuple(
                TeamRobotTruth(
                    body=self._robots[index],
                    projected_gravity=self._projected_gravity[index],
                    last_commands=self._last_commands[index],
                    box_contact=self._box_contacts[index],
                )
                for index in range(TEAM_SIZE)
            ),
            box=self._box,
            goal_xy_m=self._goal_xy,
            episode_progress=self.episode_length_buf.to(torch.float32) / self.max_episode_length,
        )
        self._states = tuple(build_team_state(truth, sign, task.observation) for sign in task.agent_side_signs)
