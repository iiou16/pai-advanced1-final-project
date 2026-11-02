"""箱押しの学習環境（Go2PushEnv）を、学習済みの歩行方策と Genesis のシーン S1_push で動かすスモークテスト。

ランダムな速度指令で複数のエピソードを回し、例外なく進むこと、観測・行動・報酬の形、終了理由が記録されることを確かめる。
GPU で学習が走っていても取り合わないよう、Genesis を CPU で初期化し、少数の環境・短いエピソードで動かす。

実行: uv run python scripts/p2_scene_check/smoke_push_env.py
"""

import dataclasses
from collections import Counter
from pathlib import Path

import torch

from cooperative_go2.entities.push_observation import PUSH_OBSERVATION_DIM
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import (
    load_gait_settings,
    load_push_scene_settings,
    load_push_training_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_push_simulation import GenesisPushSimulation
from cooperative_go2.interface_adapters.push_env import Go2PushEnv, PushEnvSetup

PUSH_CONFIG_PATH = Path("configs/push_single.yaml")
SEED = 0
NUM_ENVS = 4
# 短いエピソードで、時間切れによるリセットを何度も起こす
EPISODE_LENGTH_S = 2.0
NUM_HIGH_LEVEL_STEPS = 60
MIN_COMPLETED_EPISODES = NUM_ENVS


class SmokeCheckError(RuntimeError):
    """箱押しの学習環境が期待どおりに動かなかったことを表す。"""


def main() -> None:
    settings = load_push_training_settings(PUSH_CONFIG_PATH)
    run_dir = Path(settings.gait_policy.run_dir)
    gait = load_gait_settings(run_dir / RUN_CONFIG_FILENAME).gait
    scene_settings = load_push_scene_settings(Path(settings.scene_config))
    episode = dataclasses.replace(settings.task.episode, num_envs=NUM_ENVS, episode_length_s=EPISODE_LENGTH_S)
    task = dataclasses.replace(settings.task, episode=episode)

    init_genesis_on_cpu(SEED)
    generator = make_seeded_generator(SEED)
    simulation = GenesisPushSimulation.build(gait, scene_settings, NUM_ENVS)
    controller = load_gait_controller(run_dir, settings.gait_policy.checkpoint, NUM_ENVS, generator.device)
    setup = PushEnvSetup(task=task, gait=gait, base_box_xy_m=scene_settings.box.initial_xy_m, generator=generator)
    env = Go2PushEnv(setup, simulation, controller)

    reasons: Counter[str] = Counter()
    low = torch.tensor([limit.low for limit in task.commands.as_ranges()], device=generator.device)
    high = torch.tensor([limit.high for limit in task.commands.as_ranges()], device=generator.device)
    for _ in range(NUM_HIGH_LEVEL_STEPS):
        unit = torch.rand((NUM_ENVS, VELOCITY_COMMAND_SIZE), generator=generator, device=generator.device)
        outcome = env.step_with_outcome(low + (high - low) * unit)
        if outcome.observation.shape != (NUM_ENVS, PUSH_OBSERVATION_DIM) or outcome.reward.shape != (NUM_ENVS,):
            raise SmokeCheckError(
                f"形が想定と異なる: 観測 {tuple(outcome.observation.shape)} 報酬 {tuple(outcome.reward.shape)}"
            )
        if not bool(torch.isfinite(outcome.observation).all()) or not bool(torch.isfinite(outcome.reward).all()):
            raise SmokeCheckError("観測または報酬に NaN・無限大が含まれる")
        finished = outcome.termination_reasons[outcome.done]
        reasons.update(PushTerminationReason(int(reason)).name for reason in finished)

    completed = sum(reasons.values())
    print(f"高レベル方策 {NUM_HIGH_LEVEL_STEPS} ステップで終わったエピソード {completed} 件: {dict(reasons)}")
    if completed < MIN_COMPLETED_EPISODES:
        raise SmokeCheckError(f"終わったエピソードが {completed} 件しかない（{MIN_COMPLETED_EPISODES} 件以上を期待）")
    print("箱押しの学習環境のスモークテストに合格")


if __name__ == "__main__":
    main()
