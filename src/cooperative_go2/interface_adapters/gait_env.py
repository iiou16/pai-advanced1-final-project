"""Go2 の歩行学習環境。RSL-RL の VecEnv と同じ属性・メソッドを持ち、OnPolicyRunner にそのまま渡せる。

報酬・観測・終了判定・指令・行動の履歴は entities の部品で計算し、物理はシミュレーションの操作（GaitSimulation）に
任せる。このクラスは部品を順に呼ぶことだけを担う。
"""

from dataclasses import dataclass
from typing import Protocol

import torch
from tensordict import TensorDict

from cooperative_go2.entities.episode_reward_tracker import EpisodeRewardTracker, EpisodeSummary
from cooperative_go2.entities.gait_action import GaitActionHistory, to_joint_targets
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_observation import GaitObservationInputs, build_gait_observation
from cooperative_go2.entities.gait_reward import GaitRewardInputs, GaitRewardTerms, compute_gait_reward_terms
from cooperative_go2.entities.gait_termination import TerminationReason, classify_terminations, exceeds_tilt_limit
from cooperative_go2.entities.go2 import NUM_GO2_JOINTS
from cooperative_go2.entities.go2_state import Go2State, sample_reset_dof_pos
from cooperative_go2.entities.velocity_command import VelocityCommand, VelocityCommandSampler
from cooperative_go2.interface_adapters.rsl_rl_config import POLICY_OBSERVATION_GROUP


class GaitSimulation(Protocol):
    """歩行環境が使うシミュレーションの操作。物理ステップを進めるのはこの操作だけ。"""

    @property
    def default_dof_pos(self) -> torch.Tensor:
        """既定姿勢の関節角 (12,) [rad]。"""
        ...

    @property
    def initial_state(self) -> Go2State:
        """リセット直後の状態（全環境に共通の 1 行）。"""
        ...

    def apply_joint_targets(self, joint_targets: torch.Tensor) -> None:
        """PD 制御の目標関節角 (num_envs, 12) [rad] を設定する。"""
        ...

    def step(self) -> None:
        """物理を 1 制御周期進める。

        Raises:
            SimulationSetupError: 衝突候補などの数が設定の上限を超えた。
        """
        ...

    def read_state(self) -> Go2State:
        """現在の状態を返す。"""
        ...

    def diverged_mask(self) -> torch.Tensor:
        """直前の物理ステップで物理計算が発散した環境のマスク (num_envs,) を返す。"""
        ...

    def reset(self, mask: torch.Tensor, dof_pos: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境を初期位置で、関節角を dof_pos (num_envs, 12) [rad] にして静止させる。"""
        ...


@dataclass(frozen=True)
class GaitStepOutcome:
    """1 制御ステップの結果。

    Attributes:
        observation: 次の行動を決めるための観測 (num_envs, GAIT_OBSERVATION_DIM)。終了した環境はリセット後の観測。
        reward: 報酬 (num_envs,)。物理計算が発散した環境は 0。
        done: エピソードが終わった環境のマスク (num_envs,)。
        time_out: 時間切れだけで終わった環境のマスク (num_envs,)。
        pre_reset_state: 物理計算の直後、リセットする前の状態。
        termination_reasons: 環境ごとの終了理由 (num_envs,)。TerminationReason の値。
        episode_summary: このステップで終わったエピソードの集計。終わった環境が無ければ None。
    """

    observation: torch.Tensor
    reward: torch.Tensor
    done: torch.Tensor
    time_out: torch.Tensor
    pre_reset_state: Go2State
    termination_reasons: torch.Tensor
    episode_summary: EpisodeSummary | None


class Go2GaitEnv:
    """Go2 の歩行を学習・評価する並列環境。

    num_envs・num_actions・max_episode_length・episode_length_buf・device・cfg は RSL-RL の VecEnv の契約で、
    OnPolicyRunner が episode_length_buf を外から代入して各環境のエピソードの進みをばらす。
    """

    def __init__(self, config: GaitConfig, simulation: GaitSimulation, generator: torch.Generator) -> None:
        """全環境を初期姿勢にして、最初の観測を用意する。

        Args:
            config: 歩行の設定。simulation はこの設定（env.num_envs を含む）で構築したものであること。
            simulation: 物理を進めるシミュレーションの操作。
            generator: 速度指令とリセット時の関節角のノイズを引く乱数生成器。シードを固定して渡す。
                テンソルはこの生成器のデバイスに置く。
        """
        self.cfg = config
        self.num_envs = config.env.num_envs
        self.num_actions = NUM_GO2_JOINTS
        self.device = generator.device
        self.max_episode_length = config.max_episode_length_steps()
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.int64, device=self.device)
        self._simulation = simulation
        self._generator = generator
        self._commands = VelocityCommandSampler(config.command, self.num_envs, generator)
        self._rewards = EpisodeRewardTracker(self.num_envs, config.env.control_dt_s, self.device)
        self._actions = GaitActionHistory.zeros(self.num_envs, NUM_GO2_JOINTS, self.device)
        self._state = simulation.initial_state
        self.reset()

    def fix_command(self, command: VelocityCommand) -> None:
        """全環境の速度指令を直ちに command に切り替え、以後も固定する（評価用）。観測も新しい指令で作り直す。"""
        self._commands.fix(command)
        self._observation = self._observe()

    def get_observations(self) -> TensorDict:
        """現在の観測を RSL-RL の観測グループの形で返す。"""
        return TensorDict({POLICY_OBSERVATION_GROUP: self._observation}, batch_size=[self.num_envs])

    def reset(self) -> TensorDict:
        """全環境を初期姿勢に戻し、観測を返す。"""
        self._reset_envs(torch.ones(self.num_envs, dtype=torch.bool, device=self.device))
        self._observation = self._observe()
        return self.get_observations()

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, object]]:
        """RSL-RL の VecEnv.step。step_with_outcome() の結果を RSL-RL の形に変換する。"""
        outcome = self.step_with_outcome(actions)
        extras: dict[str, object] = {"time_outs": outcome.time_out.to(torch.float32)}
        if outcome.episode_summary is not None:
            extras["episode"] = outcome.episode_summary.as_log()
        return self.get_observations(), outcome.reward, outcome.done, extras

    def step_with_outcome(self, actions: torch.Tensor) -> GaitStepOutcome:
        """行動を適用して 1 制御周期進める。終了した環境はその場で初期姿勢に戻す。

        Raises:
            SimulationSetupError: 衝突候補などの数が設定の上限を超えた。
        """
        env_config = self.cfg.env
        self._actions = self._actions.advance(actions, env_config.clip_actions)
        applied = self._actions.applied(env_config.simulate_action_latency)
        default_dof_pos = self._simulation.default_dof_pos
        self._simulation.apply_joint_targets(to_joint_targets(applied, default_dof_pos, env_config.action_scale))
        self._simulation.step()
        self.episode_length_buf += 1
        state = self._simulation.read_state()
        reasons = self._classify_terminations(state)
        reward_terms = self._weighted_reward_terms(state, diverged=reasons == int(TerminationReason.SIM_DIVERGED))
        self._rewards.add_step(reward_terms)
        self._commands.resample(self.episode_length_buf % self.cfg.command_resampling_interval_steps() == 0)
        done = reasons != int(TerminationReason.NONE)
        episode_summary = self._rewards.summarize_completed_or_none(done, reasons)
        self._state = state
        self._reset_envs(done)
        self._observation = self._observe()
        return GaitStepOutcome(
            observation=self._observation,
            reward=reward_terms.total(),
            done=done,
            time_out=reasons == int(TerminationReason.TIME_OUT),
            pre_reset_state=state,
            termination_reasons=reasons,
            episode_summary=episode_summary,
        )

    def _classify_terminations(self, state: Go2State) -> torch.Tensor:
        env_config = self.cfg.env
        fallen = exceeds_tilt_limit(
            state.base_euler_deg, env_config.termination_roll_deg, env_config.termination_pitch_deg
        )
        # 公式の例と同じく、エピソードの長さを 1 ステップ超えたところで時間切れにする
        time_out = self.episode_length_buf > self.max_episode_length
        return classify_terminations(time_out, fallen, self._simulation.diverged_mask())

    def _weighted_reward_terms(self, state: Go2State, diverged: torch.Tensor) -> GaitRewardTerms:
        inputs = GaitRewardInputs(state, self._commands.commands, self._actions, self._simulation.default_dof_pos)
        terms = compute_gait_reward_terms(inputs, self.cfg.reward)
        # 発散した環境の状態は NaN になりうる。学習を止めずにそのエピソードだけを失敗として終えるため、報酬を 0 にする
        # （RSL-RL は報酬に NaN があると学習を止める）
        return terms.weighted(self.cfg.reward.scales, self.cfg.env.control_dt_s).zeroed_where(diverged)

    def _reset_envs(self, mask: torch.Tensor) -> None:
        reset_dof_pos = sample_reset_dof_pos(
            self._simulation.default_dof_pos, self.num_envs, self.cfg.env.reset_joint_pos_noise_rad, self._generator
        )
        self._simulation.reset(mask, reset_dof_pos)
        self._state = self._state.reset_where(mask, self._simulation.initial_state.with_dof_pos(reset_dof_pos))
        self._actions = self._actions.reset_where(mask)
        self._rewards.reset_where(mask)
        self.episode_length_buf = self.episode_length_buf.masked_fill(mask, 0)
        self._commands.resample(mask)

    def _observe(self) -> torch.Tensor:
        inputs = GaitObservationInputs(
            proprioception=self._state.proprioception,
            commands=self._commands.commands,
            last_actions=self._actions.current,
            default_dof_pos=self._simulation.default_dof_pos,
        )
        return build_gait_observation(inputs, self.cfg.observation)
