"""摩擦円錐のモデル（pyramidal / elliptic）による、押された箱の動きの違いを動画に撮る（レポート・提出動画の素材）。

シーン S1_push の箱に、静止摩擦力の 2 倍の水平な力を 1 秒かけてから離す。Go2 は既定姿勢で立たせておく。
pyramidal（Genesis の既定）では箱が浮いて飛び、転がる。elliptic では床を滑って止まる。

実行: uv run python scripts/p2_scene_check/record_friction_cone_demo.py <pyramidal|elliptic> <出力する mp4>
"""

import dataclasses
import sys
from pathlib import Path

import genesis as gs
import torch
from genesis.utils.geom import default_friction

from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_scene_settings
from cooperative_go2.interface_adapters.genesis_flat_scene import CameraSpec
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body
from cooperative_go2.interface_adapters.genesis_push_scene import PushSceneRequest, PushVideoSpec, build_push_scene

GAIT_CONFIG_PATH = Path("configs/gait.yaml")
SCENE_CONFIG_PATH = Path("configs/scenes/s1_push.yaml")
SEED = 0
NUM_ENVS = 1
GRAVITY_MPS2 = 9.81
SETTLE_S = 0.5
PUSH_S = 1.0
RELEASE_S = 2.0
PUSH_FORCE_RATIO = 2.0
# Go2（立ったまま動かない）の斜め横から、箱が押される方向（+x）の数 m 先まで入るように広く撮る
DEMO_CAMERA = CameraSpec(resolution=(960, 540), offset_m=(2.5, -6.5, 1.6), fov_deg=70.0)
# この動画には目標領域は無いので、目標を示す円板は見えない大きさにする
HIDDEN_GOAL_MARKER_RADIUS_M = 0.001


def main() -> None:
    cone, output = sys.argv[1], Path(sys.argv[2])
    gait = load_gait_settings(GAIT_CONFIG_PATH).gait
    settings = load_push_scene_settings(SCENE_CONFIG_PATH)
    settings = dataclasses.replace(settings, genesis=dataclasses.replace(settings.genesis, friction_cone=cone))
    init_genesis_on_cuda(SEED)
    video = PushVideoSpec(DEMO_CAMERA, HIDDEN_GOAL_MARKER_RADIUS_M)
    scene = build_push_scene(gait, settings, PushSceneRequest(NUM_ENVS, video=video))
    body = GenesisGo2Body(scene.robot, gait.robot)
    camera = scene.ground_and_robot.require_camera()
    standing = body.default_dof_pos.expand(NUM_ENVS, -1)
    friction_force = max(settings.box.friction, default_friction()) * settings.box.mass_kg * GRAVITY_MPS2
    dt = gait.env.control_dt_s
    output.parent.mkdir(parents=True, exist_ok=True)
    camera.start_recording(save_to_filename=str(output), fps=round(1.0 / dt))
    for phase_s, force_x in ((SETTLE_S, 0.0), (PUSH_S, PUSH_FORCE_RATIO * friction_force), (RELEASE_S, 0.0)):
        force = torch.tensor([[[force_x, 0.0, 0.0]]], dtype=gs.tc_float, device=gs.device)
        for _ in range(round(phase_s / dt)):
            scene.box.apply_links_external_wrench(force=force)
            body.apply_joint_targets(standing)
            # 録画中は物理ステップのたびにカメラが 1 フレーム描画する
            scene.step()
    camera.stop_recording()
    print(f"{cone}: 動画を保存した: {output}")


if __name__ == "__main__":
    main()
