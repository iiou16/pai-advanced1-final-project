from pathlib import Path

import pytest
import torch
from conftest import REPO_ROOT, TEAM_NUM_ENVS, TeamMarlEnvFactory
from skrl.trainers.torch.sequential import SequentialTrainerCfg

from cooperative_go2.entities.team_marl_observation import (
    TEAM_ACTOR_OBSERVATION_DIM,
    team_state_dim,
)
from cooperative_go2.interface_adapters.config_loader import load_team_marl_run_settings
from cooperative_go2.interface_adapters.skrl_marl_config import (
    MarlAlgorithm,
    MarlVariant,
    SkrlMarlPpoConfig,
)
from cooperative_go2.interface_adapters.skrl_team import (
    TERMINAL_STATES_INFO_KEY,
    SkrlTeamEnv,
    TeamPolicy,
    TeamValue,
    build_team_agent,
)
from cooperative_go2.interface_adapters.team_marl_env import TeamSizeError

NAMES = ("go2_a", "go2_b")
NUM_SAMPLES = 4


@pytest.fixture
def ppo() -> SkrlMarlPpoConfig:
    return load_team_marl_run_settings(REPO_ROOT / "configs" / "push_multi.yaml").settings.skrl


def _inputs(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "observations": torch.randn(NUM_SAMPLES, TEAM_ACTOR_OBSERVATION_DIM, generator=generator),
        "states": torch.randn(NUM_SAMPLES, team_state_dim(2), generator=generator),
    }


def _with(inputs: dict[str, torch.Tensor], key: str, seed: int) -> dict[str, torch.Tensor]:
    return inputs | {key: _inputs(seed)[key]}


def test_actor_ignores_the_global_state(ppo: SkrlMarlPpoConfig) -> None:
    torch.manual_seed(0)
    policy = TeamPolicy(ppo, "cpu")
    inputs = _inputs(1)

    mean, _ = policy.compute(inputs)
    mean_with_other_state, _ = policy.compute(_with(inputs, "states", 2))

    assert torch.equal(mean, mean_with_other_state)


@pytest.mark.parametrize(
    ("algorithm", "read_key", "ignored_key"),
    [(MarlAlgorithm.IPPO, "observations", "states"), (MarlAlgorithm.MAPPO, "states", "observations")],
    ids=["ippo-critic-reads-observations", "mappo-critic-reads-global-state"],
)
def test_critic_reads_only_the_input_of_its_algorithm(
    ppo: SkrlMarlPpoConfig, algorithm: MarlAlgorithm, read_key: str, ignored_key: str
) -> None:
    torch.manual_seed(0)
    value = TeamValue(ppo, algorithm, num_agents=2, device="cpu")
    inputs = _inputs(1)

    baseline, _ = value.compute(inputs)
    changed_read, _ = value.compute(_with(inputs, read_key, 2))
    changed_ignored, _ = value.compute(_with(inputs, ignored_key, 2))

    assert not torch.equal(baseline, changed_read)
    assert torch.equal(baseline, changed_ignored)


def test_wrapped_env_splits_time_out_into_truncated(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()
    wrapped = SkrlTeamEnv(env, NAMES)
    stand_still = {name: torch.zeros(TEAM_NUM_ENVS, 3) for name in NAMES}

    for _ in range(env.max_episode_length - 1):
        _, rewards, terminated, truncated, _ = wrapped.step(stand_still)
    _, _, terminated, truncated, infos = wrapped.step(stand_still)

    assert rewards["go2_a"].shape == (TEAM_NUM_ENVS, 1)
    assert truncated["go2_a"].all() and not terminated["go2_a"].any()
    assert "termination_time_out" in infos["episode"]
    assert infos["episode"]["termination_time_out"].item() == pytest.approx(1.0)


def test_wrapped_env_exposes_actor_observation_and_state_spaces(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()
    wrapped = SkrlTeamEnv(env, NAMES)

    observations, _ = wrapped.reset()
    states = wrapped.state()

    assert wrapped.possible_agents == list(NAMES)
    assert observations["go2_b"].shape == (TEAM_NUM_ENVS, wrapped.observation_spaces["go2_b"].shape[0])
    assert states["go2_b"].shape == (TEAM_NUM_ENVS, wrapped.state_spaces["go2_b"].shape[0])


@pytest.mark.parametrize("shared", [True, False], ids=["shared", "separate"])
def test_parameter_sharing_uses_one_model_and_one_optimizer(
    make_team_marl_env: TeamMarlEnvFactory, ppo: SkrlMarlPpoConfig, tmp_path: Path, shared: bool
) -> None:
    env, _ = make_team_marl_env()
    wrapped = SkrlTeamEnv(env, NAMES)

    agent = build_team_agent(wrapped, ppo, MarlVariant(MarlAlgorithm.MAPPO, shared), tmp_path / "run")

    assert (agent.policies["go2_a"] is agent.policies["go2_b"]) is shared
    assert (agent.values["go2_a"] is agent.values["go2_b"]) is shared
    assert (agent.optimizers["go2_a"] is agent.optimizers["go2_b"]) is shared


def test_wrapped_env_rejects_a_wrong_number_of_agent_names(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()

    with pytest.raises(TeamSizeError):
        SkrlTeamEnv(env, ("go2_a",))


def test_time_out_passes_the_state_before_reset_to_the_agent(make_team_marl_env: TeamMarlEnvFactory) -> None:
    env, _ = make_team_marl_env()
    wrapped = SkrlTeamEnv(env, NAMES)
    stand_still = {name: torch.zeros(TEAM_NUM_ENVS, 3) for name in NAMES}

    for _ in range(env.max_episode_length - 1):
        wrapped.step(stand_still)
    _, _, _, _, infos = wrapped.step(stand_still)

    # Critic の状態の最後から 2 番目はエピソードの経過。リセット前は 1、リセット後は 0
    assert torch.allclose(infos[TERMINAL_STATES_INFO_KEY]["go2_a"][:, -2], torch.ones(TEAM_NUM_ENVS))
    assert torch.allclose(wrapped.state()["go2_a"][:, -2], torch.zeros(TEAM_NUM_ENVS))


@pytest.mark.parametrize("algorithm", [MarlAlgorithm.IPPO, MarlAlgorithm.MAPPO], ids=["ippo", "mappo"])
def test_time_out_reward_is_bootstrapped_from_the_terminal_state(
    make_team_marl_env: TeamMarlEnvFactory, ppo: SkrlMarlPpoConfig, tmp_path: Path, algorithm: MarlAlgorithm
) -> None:
    env, _ = make_team_marl_env()
    wrapped = SkrlTeamEnv(env, NAMES)
    agent = build_team_agent(wrapped, ppo, MarlVariant(algorithm, shared_parameters=False), tmp_path / "run")
    agent.init(trainer_cfg=SequentialTrainerCfg(timesteps=ppo.rollouts))
    # 学習の流れ（SequentialTrainer）と同じく学習中にする。学習中でなければ遷移を記録しない
    agent.enable_training_mode(True)
    observations, _ = wrapped.reset()
    states = wrapped.state()
    actions, _ = agent.act(observations, states, timestep=0, timesteps=ppo.rollouts)
    terminal = {"observations": _inputs(1)["observations"][:TEAM_NUM_ENVS], "states": _inputs(1)["states"][:TEAM_NUM_ENVS]}
    truncated = {name: torch.tensor([[True], [False]]) for name in NAMES}
    zeros = {name: torch.zeros(TEAM_NUM_ENVS, 1) for name in NAMES}

    with torch.no_grad():
        agent.record_transition(
            observations=observations, states=states, actions=actions, rewards=dict(zeros),
            next_observations=observations, next_states=states,
            terminated={name: torch.zeros(TEAM_NUM_ENVS, 1, dtype=torch.bool) for name in NAMES},
            truncated=truncated,
            infos={"terminal_observations": {name: terminal["observations"] for name in NAMES},
                   "terminal_states": {name: terminal["states"] for name in NAMES}},
            timestep=0, timesteps=ppo.rollouts,
        )  # fmt: skip
        expected, _ = agent.values["go2_a"].act(terminal, role="value")

    recorded = agent.memories["go2_a"].get_tensor_by_name("rewards")[0]
    assert recorded[0].item() == pytest.approx(ppo.discount_factor * expected[0].item(), rel=1e-5)
    assert recorded[1].item() == 0.0
