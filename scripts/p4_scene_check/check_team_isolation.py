"""2 台のシーン（S3_team）で、片方のロボットへの指令がそのロボットだけを動かすことを確かめる。

ROS 2 の配信・受信の側で混線しないことは tests/ros2/test_team_topic_isolation.py で確かめる。

片方のロボットにだけ後退の指令を出し、もう片方には停止の指令を出して物理を進める。指令を出したロボットだけが
動き、止めたロボットは動かないことを、両方の組み合わせ（go2_a だけ・go2_b だけ）で確かめる。前進では箱にすぐ
当たるので、箱から離れる後退の指令を使う。どちらかの組み合わせで判定を満たさなければ終了コード 1 で終わる。

実行（Genesis を cuda で使う前に、同じシェルで source scripts/setup/env.sh する）:
    uv run python scripts/p4_scene_check/check_team_isolation.py --device cuda
"""

import argparse
import sys
from pathlib import Path

import genesis as gs
import torch

from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_team_run_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation

CONFIG = Path("configs/team_rule_based.yaml")
SEED = 4201
# 後退の指令 [m/s]（歩行方策が学習した範囲 -0.5〜1.0 の内側）と、指令を出す時間 [s]
BACKWARD_MPS = -0.3
DURATION_S = 3.0
# 指令を出したロボットが動いたとみなす移動量 [m]（理想は 0.3 × 3 = 0.9 m）と、止めたロボットが動かなかったとみなす移動量 [m]
MOVED_M = 0.5
STILL_M = 0.05
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}


def run_case(simulation: GenesisCorridorSimulation, controllers: list, moving_index: int, control_dt_s: float
             ) -> list[float]:  # fmt: skip
    """moving_index のロボットにだけ後退の指令を出して物理を進め、ロボットごとの水平の移動量 [m] を返す。"""
    all_envs = torch.ones(1, dtype=torch.bool, device=gs.device)
    dof_pos = tuple(simulation.default_dof_pos.unsqueeze(0).clone() for _ in controllers)
    simulation.reset(all_envs, dof_pos)
    for controller in controllers:
        controller.reset_where(all_envs)
    start = [planar.position_xy_m.clone() for planar in simulation.read_robot_planars()]
    pending = [simulation.default_dof_pos.unsqueeze(0).clone() for _ in controllers]
    for _ in range(round(DURATION_S / control_dt_s)):
        states = simulation.read_go2_states()
        targets = []
        for index, (state, controller) in enumerate(zip(states, controllers, strict=True)):
            vx = BACKWARD_MPS if index == moving_index else 0.0
            commands = torch.tensor([[vx, 0.0, 0.0]], device=gs.device)
            targets.append(controller.compute_joint_targets(state.proprioception, commands))
        # 歩行方策の学習時と同じく、1 制御周期前に決めた目標を適用する
        simulation.apply_joint_targets(tuple(pending))
        pending = targets
        simulation.step()
    end = [planar.position_xy_m for planar in simulation.read_robot_planars()]
    return [float(torch.linalg.vector_norm(after - before)) for before, after in zip(start, end, strict=True)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    args = parser.parse_args()
    run = load_team_run_settings(CONFIG)
    gait_policy = run.settings.gait_policy
    gait = load_gait_settings(Path(gait_policy.run_dir) / RUN_CONFIG_FILENAME).gait
    GENESIS_INITIALIZERS[args.device](SEED)
    simulation = GenesisCorridorSimulation.build(gait, run.scene, CorridorSceneRequest(num_envs=1, camera=None))
    names = simulation.robot_names
    controllers = [load_gait_controller(Path(gait_policy.run_dir), gait_policy.checkpoint, 1, gs.device) for _ in names]
    passed = True
    for moving_index, moving_name in enumerate(names):
        displacements = run_case(simulation, controllers, moving_index, gait.env.control_dt_s)
        moved = displacements[moving_index] >= MOVED_M
        others_still = all(
            displacement <= STILL_M for index, displacement in enumerate(displacements) if index != moving_index
        )
        pairs = zip(names, displacements, strict=True)
        summary = "、".join(f"{name} {displacement:.3f} m" for name, displacement in pairs)
        print(f"{moving_name} だけに後退の指令: {summary} → {'混線なし' if moved and others_still else '判定を満たさない'}")
        passed = passed and moved and others_still
    if not passed:
        print("指令を出していないロボットが動いた、または指令を出したロボットが動かなかった", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
