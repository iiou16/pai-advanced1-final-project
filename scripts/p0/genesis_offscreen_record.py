"""Genesis のオフスクリーンカメラで mp4 を録画できることを確かめるスモークテスト.

Viewer (WSLg) に依存しない評価動画の取得経路を検証する。
出力: outputs/p0/offscreen_record.mp4
"""

from pathlib import Path

import genesis as gs

OUTPUT_PATH = Path("outputs/p0/offscreen_record.mp4")
SIM_DT_S = 0.01
SIM_STEPS = 150
VIDEO_FPS = int(round(1 / SIM_DT_S))  # 1 step = 1 フレームで実時間再生にする
CAMERA_RES = (640, 480)
CAMERA_POS = (2.5, -2.5, 1.5)
CAMERA_LOOKAT = (0.0, 0.0, 0.3)
CAMERA_FOV_DEG = 40
BOX_SIZE_M = 0.2
BOX_DROP_HEIGHT_M = 1.0


class RecordingError(RuntimeError):
    """録画ファイルが生成されなかったことを表す。"""


def main() -> None:
    """箱の落下を録画し、mp4 が生成されたことを検証する。"""
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT_S),
        renderer=gs.renderers.Rasterizer(),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(BOX_SIZE_M,) * 3, pos=(0.0, 0.0, BOX_DROP_HEIGHT_M)))
    camera = scene.add_camera(
        res=CAMERA_RES, pos=CAMERA_POS, lookat=CAMERA_LOOKAT, fov=CAMERA_FOV_DEG, GUI=False
    )
    scene.build()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.unlink(missing_ok=True)
    camera.start_recording(save_to_filename=str(OUTPUT_PATH), fps=VIDEO_FPS)
    for _ in range(SIM_STEPS):
        scene.step()
    camera.stop_recording()

    if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
        raise RecordingError(f"録画ファイルが生成されていない: {OUTPUT_PATH}")
    print(f"saved {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes)")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
