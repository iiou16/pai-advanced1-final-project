"""2 台の協調学習の環境（Go2TeamMarlEnv）を、学習済みの歩行方策と Genesis のシーン S3_team で動かすスモークテスト。

ランダムな速度指令で 2 台分のエピソードを回し、例外なく進むこと・観測と状態と報酬の形・終了理由を確かめ、
並列環境の数ごとの速さ（高レベル方策のステップ/秒）と GPU のメモリの最大を表示する。学習の並列環境の数を決めるのに使う。

実行（Genesis を cuda で使う前に、同じシェルで source scripts/setup/env.sh する）:
    uv run python scripts/p5_marl_check/smoke_team_marl_env.py --device cuda --num-envs 256 --steps 100
"""

import argparse
import dataclasses
import time
from collections import Counter
from pathlib import Path

import genesis as gs
import torch

from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.team_marl_observation import TEAM_ACTOR_OBSERVATION_DIM, team_state_dim
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_team_marl_run_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv, TeamMarlEnvSetup

CONFIG_PATH = Path("configs/push_multi.yaml")
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}
# 速さを測る前に捨てるステップ数（Genesis の最初のステップはカーネルのコンパイルを含む）
WARMUP_STEPS = 5


class SmokeCheckError(RuntimeError):
    """協調学習の環境が期待どおりに動かなかったことを表す。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True, help="高レベル方策のステップ数")
    parser.add_argument(
        "--episode-length-s", type=float, default=None,
        help="エピソードの長さを短くしてリセットを起こす [s]。省略すれば設定どおり",
    )  # fmt: skip
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= WARMUP_STEPS:
        raise ValueError(f"--steps は速さを測るために {WARMUP_STEPS} より大きくする: {args.steps}")
    run = load_team_marl_run_settings(CONFIG_PATH)
    settings = run.settings
    gait_run_dir = Path(settings.gait_policy.run_dir)
    gait = load_gait_settings(gait_run_dir / RUN_CONFIG_FILENAME).gait
    episode = dataclasses.replace(settings.task.episode, num_envs=args.num_envs)
    if args.episode_length_s is not None:
        episode = dataclasses.replace(episode, episode_length_s=args.episode_length_s)
    task = dataclasses.replace(settings.task, episode=episode)

    GENESIS_INITIALIZERS[args.device](args.seed)
    generator = make_seeded_generator(args.seed)
    simulation = GenesisCorridorSimulation.build(gait, run.scene, CorridorSceneRequest(args.num_envs, camera=None))
    controllers = tuple(
        load_gait_controller(gait_run_dir, settings.gait_policy.checkpoint, args.num_envs, gs.device)
        for _ in simulation.robot_names
    )
    env = Go2TeamMarlEnv(TeamMarlEnvSetup(task, gait, run.scene.scene, generator), simulation, controllers)

    reasons: Counter[str] = Counter()
    span = env.command_high - env.command_low
    start = time.perf_counter()
    for step in range(args.steps):
        if step == WARMUP_STEPS:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
        actions = tuple(
            env.command_low + span * torch.rand((args.num_envs, 3), generator=generator, device=generator.device)
            for _ in range(env.num_agents)
        )
        outcome = env.step(actions)
        for observation, state in zip(outcome.observations, outcome.states, strict=True):
            if observation.shape != (args.num_envs, TEAM_ACTOR_OBSERVATION_DIM):
                raise SmokeCheckError(f"Actor の観測の形が違う: {tuple(observation.shape)}")
            if state.shape != (args.num_envs, team_state_dim(env.num_agents)):
                raise SmokeCheckError(f"Critic の状態の形が違う: {tuple(state.shape)}")
            if not (torch.isfinite(observation).all() and torch.isfinite(state).all()):
                raise SmokeCheckError(f"観測か状態に有限でない値がある（ステップ {step}）")
        if outcome.rewards.shape != (env.num_agents, args.num_envs) or not torch.isfinite(outcome.rewards).all():
            raise SmokeCheckError(f"報酬の形か値が不正: {tuple(outcome.rewards.shape)}")
        for reason in outcome.termination_reasons[outcome.done].tolist():
            reasons[PushTerminationReason(reason).name] += 1
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_s = time.perf_counter() - start
    measured_steps = args.steps - WARMUP_STEPS
    steps_per_s = measured_steps * args.num_envs / elapsed_s
    print(f"並列環境 {args.num_envs}、{args.steps} ステップ: 例外なし。終了理由 {dict(reasons)}")
    print(
        f"速さ: {steps_per_s:.0f} 環境ステップ/秒（高レベル方策の 1 ステップ = 物理 {env.gait_steps_per_high_level_step} 回）"
    )
    if torch.cuda.is_available():
        print(f"GPU のメモリの最大（PyTorch の確保分）: {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")


if __name__ == "__main__":
    main()
