"""箱の質量と摩擦を変えて、水平の外力で押した箱が滑るか倒れるかを確かめる。

シーン S1_push の箱（0.4 × 0.4 × 0.3 m）の中心（重心）に、Go2 から離れる向き（+x）の外力を 1 秒かけ、箱の移動量・
速さと、押している間の傾き・浮き上がり・床との接触が途切れたステップの数を出す。重心を押すとき、摩擦係数が
箱の幅 / 高さ（約 1.33）を超えると、箱は滑らずに倒れる。頑健性の評価の摩擦の条件が「滑りにくい箱」になって
いるかの確認に使う。Go2 は重心より高い位置を押すので、Go2 が押したときの傾きはこの確認とは別に見る必要がある。
Genesis を CPU で初期化し、並列環境 1 つで動かす。1 回の実行で 1 つの条件を測る（Genesis は 1 つのプロセスで 1 回だけ初期化する）。

実行例（リポジトリのルートで）:
    uv run --frozen python scripts/p6_robustness/probe_box_friction.py --mass-kg 2.0 --friction 1.3 --force-n 35
"""

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "p2_scene_check"))
from check_s1_push_scene import GAIT_CONFIG_PATH, NUM_ENVS, SCENE_CONFIG_PATH, SEED, SceneProbe  # noqa: E402

from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu  # noqa: E402
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_scene_settings  # noqa: E402
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body  # noqa: E402
from cooperative_go2.interface_adapters.genesis_push_scene import PushSceneRequest, build_push_scene  # noqa: E402

SETTLE_S = 1.0
PUSH_S = 1.0


def main() -> None:
    """1 つの条件で箱を押し、結果を 1 行で表示する。"""
    parser = argparse.ArgumentParser(description="箱を水平の外力で押し、滑るか倒れるかを確かめる")
    parser.add_argument("--mass-kg", type=float, required=True, help="箱の質量 [kg]")
    parser.add_argument("--friction", type=float, required=True, help="箱の表面の摩擦係数（床は 1.0）")
    parser.add_argument("--force-n", type=float, required=True, help="箱にかける水平の外力 [N]")
    args = parser.parse_args()
    gait = load_gait_settings(GAIT_CONFIG_PATH).gait
    settings = load_push_scene_settings(SCENE_CONFIG_PATH)
    box = dataclasses.replace(settings.box, mass_kg=args.mass_kg, friction=args.friction)
    settings = dataclasses.replace(settings, box=box)
    init_genesis_on_cpu(SEED)
    scene = build_push_scene(gait, settings, PushSceneRequest(NUM_ENVS, video=None))
    body = GenesisGo2Body(scene.robot, gait.robot)
    probe = SceneProbe(scene, body, gait.env.control_dt_s, resting_height_m=box.size_m[2] / 2)
    probe.run(SETTLE_S, (0.0, 0.0, 0.0))
    start_x = float(probe.box_position()[0])
    excursion = probe.run(PUSH_S, (args.force_n, 0.0, 0.0))
    moved_x = float(probe.box_position()[0]) - start_x
    print(
        f"質量 {args.mass_kg} kg・摩擦 {args.friction}・外力 {args.force_n} N: 移動 {moved_x:.3f} m、"
        f"速さ {probe.box_speed():.3f} m/s、傾きの最大 {excursion.max_tilt_deg:.1f}°、"
        f"浮き上がりの最大 {excursion.max_lift_m:.3f} m、床との接触が途切れたステップ {excursion.floor_contact_lost_steps}"
    )


if __name__ == "__main__":
    main()
