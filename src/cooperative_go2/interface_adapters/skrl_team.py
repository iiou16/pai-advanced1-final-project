"""2 台の協調学習の環境・方策・価値関数を、skrl の IPPO・MAPPO につなぐ。

skrl の IPPO と MAPPO は、どちらも Actor と Critic に観測（observations）と状態（states）の両方を渡す。
集中 Critic かどうかは、Critic のネットワークがどちらを読むかで決まるため、ここで MarlAlgorithm に応じて選ぶ。
Actor は常に観測だけを読み、全体の真の状態（states）を読まない。
"""

from collections.abc import Mapping
from pathlib import Path

import gymnasium
import torch
from skrl.envs.wrappers.torch.base import MultiAgentEnvWrapper
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.multi_agents.torch import MultiAgent
from skrl.multi_agents.torch.ippo import IPPO
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.resources.schedulers.torch import KLAdaptiveLR
from torch import nn

from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.team_marl_observation import TEAM_ACTOR_OBSERVATION_DIM, team_state_dim
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.interface_adapters.skrl_marl_config import MarlAlgorithm, MarlVariant, SkrlMarlPpoConfig
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv

OBSERVATIONS_KEY = "observations"
STATES_KEY = "states"
POLICY_ROLE = "policy"
VALUE_ROLE = "value"
# skrl が 1 回の書き出しで記録する、終わったエピソードの集計の置き場所（SequentialTrainer の environment_info）
EPISODE_INFO_KEY = "episode"
_ALGORITHMS: Mapping[MarlAlgorithm, type[MultiAgent]] = {MarlAlgorithm.IPPO: IPPO, MarlAlgorithm.MAPPO: MAPPO}
# Critic が読む入力。IPPO は自分の観測、MAPPO は全体の真の状態
CRITIC_INPUT_KEYS: Mapping[MarlAlgorithm, str] = {MarlAlgorithm.IPPO: OBSERVATIONS_KEY, MarlAlgorithm.MAPPO: STATES_KEY}


def _unbounded_box(dim: int) -> gymnasium.spaces.Box:
    return gymnasium.spaces.Box(low=-float("inf"), high=float("inf"), shape=(dim,))


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    # 活性化関数は 1 台の箱押しの高レベル方策（RSL-RL、elu）と同じにする
    layers: list[nn.Module] = []
    width = input_dim
    for hidden in hidden_dims:
        layers += [nn.Linear(width, hidden), nn.ELU()]
        width = hidden
    layers.append(nn.Linear(width, output_dim))
    return nn.Sequential(*layers)


class SkrlTeamEnv(MultiAgentEnvWrapper):
    """Go2TeamMarlEnv を skrl の多エージェント環境の形（エージェント名をキーとする辞書）で見せる。

    報酬・終了は (num_envs, 1) にする。時間切れは truncated、それ以外の終了（成功・転倒・離脱・発散）は terminated。
    エピソードが終わったステップでは、報酬の内訳と終了理由の割合を infos["episode"] に入れて学習ログに残す。
    """

    def __init__(self, env: Go2TeamMarlEnv, agent_names: tuple[str, ...]) -> None:
        """環境と、シーンの robots と同じ順のロボットの名前を受け取る。

        Raises:
            ValueError: agent_names の数が環境のロボットの数と違う。
        """
        super().__init__(env)
        if len(agent_names) != env.num_agents:
            raise ValueError(f"ロボット {env.num_agents} 台分の名前が必要: {agent_names}")
        self._team_env = env
        self._names = list(agent_names)
        action_space = gymnasium.spaces.Box(
            low=env.command_low.cpu().numpy(), high=env.command_high.cpu().numpy(), shape=(VELOCITY_COMMAND_SIZE,)
        )
        self._observation_spaces = {name: _unbounded_box(TEAM_ACTOR_OBSERVATION_DIM) for name in self._names}
        self._state_spaces = {name: _unbounded_box(team_state_dim(env.num_agents)) for name in self._names}
        self._action_spaces = {name: action_space for name in self._names}

    @property
    def agents(self) -> list[str]:
        """エージェント（ロボット）の名前。エピソードの間に増減しない。"""
        return self._names

    @property
    def possible_agents(self) -> list[str]:
        """エージェント（ロボット）の名前。"""
        return self._names

    @property
    def num_agents(self) -> int:
        """エージェントの数。"""
        return len(self._names)

    @property
    def max_num_agents(self) -> int:
        """エージェントの数の上限。"""
        return len(self._names)

    @property
    def observation_spaces(self) -> dict[str, gymnasium.Space]:
        """エージェントごとの Actor の観測の空間。"""
        return self._observation_spaces

    @property
    def state_spaces(self) -> dict[str, gymnasium.Space | None]:
        """エージェントごとの Critic の状態の空間。"""
        return self._state_spaces

    @property
    def action_spaces(self) -> dict[str, gymnasium.Space]:
        """エージェントごとの行動（速度指令）の空間。範囲は環境が切り詰める範囲と同じ。"""
        return self._action_spaces

    def reset(self) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
        """全環境をリセットし、エージェントごとの観測を返す。"""
        return self._by_name(self._team_env.reset()), {}

    def step(self, actions: dict[str, torch.Tensor]) -> tuple[
        dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, torch.Tensor],
        dict[str, object],
    ]:  # fmt: skip
        """エージェントごとの行動で 1 ステップ進め、観測・報酬・terminated・truncated・infos を返す。

        Raises:
            KeyError: actions にエージェントの名前が欠けている。
        """
        outcome = self._team_env.step(tuple(actions[name] for name in self._names))
        time_out = outcome.termination_reasons == int(PushTerminationReason.TIME_OUT)
        terminated = (outcome.done & ~time_out).unsqueeze(1)
        truncated = time_out.unsqueeze(1)
        infos: dict[str, object] = {}
        if outcome.episode_summary is not None:
            infos[EPISODE_INFO_KEY] = outcome.episode_summary.as_log()
        return (
            self._by_name(outcome.observations),
            {name: outcome.rewards[index].unsqueeze(1) for index, name in enumerate(self._names)},
            {name: terminated for name in self._names},
            {name: truncated for name in self._names},
            infos,
        )

    def state(self) -> dict[str, torch.Tensor | None]:
        """エージェントごとの Critic の状態を返す。"""
        return self._by_name(self._team_env.states)

    def render(self, *args: object, **kwargs: object) -> None:
        """描画しない（評価動画は Genesis のカメラで別に撮る）。"""

    def close(self) -> None:
        """閉じるものは無い（シミュレーションは呼び出し元が持つ）。"""

    def _by_name(self, values: tuple[torch.Tensor, ...]) -> dict[str, torch.Tensor]:
        return {name: value for name, value in zip(self._names, values, strict=True)}


class TeamPolicy(GaussianMixin, Model):
    """Actor。Actor の観測だけから、速度指令の平均を出す。標準偏差は観測によらない学習パラメータ。"""

    def __init__(self, ppo: SkrlMarlPpoConfig, device: torch.device | str) -> None:
        """設定のネットワークの幅と標準偏差の初期値で作る。"""
        Model.__init__(
            self,
            observation_space=_unbounded_box(TEAM_ACTOR_OBSERVATION_DIM),
            action_space=_unbounded_box(VELOCITY_COMMAND_SIZE),
            device=device,
        )
        # 行動は環境が範囲に切り詰めるので、ここでは切り詰めない（切り詰めると対数尤度が実際の行動と食い違う）
        GaussianMixin.__init__(self, clip_actions=False, clip_log_std=True, reduction="sum", role=POLICY_ROLE)
        self.net = _mlp(TEAM_ACTOR_OBSERVATION_DIM, ppo.actor_hidden_dims, VELOCITY_COMMAND_SIZE)
        self.log_std_parameter = nn.Parameter(torch.full((VELOCITY_COMMAND_SIZE,), ppo.init_log_std))

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, object]]:
        """観測 inputs["observations"] から速度指令の平均と標準偏差の対数を出す。状態（states）は読まない。"""
        return self.net(inputs[OBSERVATIONS_KEY]), {"log_std": self.log_std_parameter}

    def mean_commands(self, observations: torch.Tensor) -> torch.Tensor:
        """評価と ROS 2 の推論で使う、観測 (n, TEAM_ACTOR_OBSERVATION_DIM) に対する速度指令の平均 (n, 3)。"""
        return self.net(observations)


class TeamValue(DeterministicMixin, Model):
    """Critic。IPPO では Actor の観測、MAPPO では全体の真の状態から、価値を出す。"""

    def __init__(
        self, ppo: SkrlMarlPpoConfig, algorithm: MarlAlgorithm, num_agents: int, device: torch.device | str
    ) -> None:
        """学習方式に応じた入力の次元で作る。"""
        input_key = CRITIC_INPUT_KEYS[algorithm]
        input_dim = TEAM_ACTOR_OBSERVATION_DIM if input_key == OBSERVATIONS_KEY else team_state_dim(num_agents)
        Model.__init__(
            self,
            observation_space=_unbounded_box(TEAM_ACTOR_OBSERVATION_DIM),
            state_space=_unbounded_box(team_state_dim(num_agents)),
            action_space=_unbounded_box(1),
            device=device,
        )
        DeterministicMixin.__init__(self, clip_actions=False, role=VALUE_ROLE)
        self._input_key = input_key
        self.net = _mlp(input_dim, ppo.critic_hidden_dims, 1)

    def compute(self, inputs: dict[str, torch.Tensor], role: str = "") -> tuple[torch.Tensor, dict[str, object]]:
        """学習方式で決まる入力（観測か状態）から価値を出す。"""
        return self.net(inputs[self._input_key]), {}


def _build_models(env: SkrlTeamEnv, ppo: SkrlMarlPpoConfig, variant: MarlVariant) -> dict[str, dict[str, Model]]:
    device = env.device
    if variant.shared_parameters:
        policy = TeamPolicy(ppo, device)
        value = TeamValue(ppo, variant.algorithm, env.num_agents, device)
        return {name: {POLICY_ROLE: policy, VALUE_ROLE: value} for name in env.possible_agents}
    return {
        name: {
            POLICY_ROLE: TeamPolicy(ppo, device),
            VALUE_ROLE: TeamValue(ppo, variant.algorithm, env.num_agents, device),
        }
        for name in env.possible_agents
    }


def build_team_agent(env: SkrlTeamEnv, ppo: SkrlMarlPpoConfig, variant: MarlVariant, run_dir: Path) -> MultiAgent:
    """IPPO か MAPPO の skrl のエージェントを組み立てる。ログとチェックポイントは run_dir に書く。

    パラメータを共有する方式では、2 台に同じ Actor・Critic を渡し、最適化器と学習率の調整器も 1 つにする。
    skrl はロボットごとに更新を行うため、共有したパラメータは 1 回の更新で 2 台のデータで順に更新される。
    最適化器を 1 つにするのは、同じパラメータを別々の最適化器の状態（Adam の moment）で動かさないため。
    """
    names = env.possible_agents
    cfg = ppo.to_agent_cfg(run_dir, tuple(names)) | {"learning_rate_scheduler": KLAdaptiveLR}
    agent = _ALGORITHMS[variant.algorithm](
        possible_agents=names,
        models=_build_models(env, ppo, variant),
        memories={
            name: RandomMemory(memory_size=ppo.rollouts, num_envs=env.num_envs, device=env.device) for name in names
        },
        observation_spaces=env.observation_spaces,
        state_spaces=env.state_spaces,
        action_spaces=env.action_spaces,
        device=env.device,
        cfg=cfg,
    )
    if variant.shared_parameters:
        first = names[0]
        for name in names[1:]:
            agent.optimizers[name] = agent.optimizers[first]
            agent.schedulers[name] = agent.schedulers[first]
            agent.checkpoint_modules[name]["optimizer"] = agent.optimizers[first]
    return agent
