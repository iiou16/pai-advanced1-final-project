"""Genesis のオフスクリーンカメラで mp4 を録画できることを確かめるスモークテスト.

Viewer (WSLg) に依存しない評価動画の取得経路を検証する。
mp4 を読み戻し、フレーム数・解像度と、箱の落下で映像が変化していることを確かめる。
出力: outputs/p0_env_check/offscreen_record.mp4
"""

from pathlib import Path

import av
import genesis as gs
import numpy as np

OUTPUT_PATH = Path("outputs/p0_env_check/offscreen_record.mp4")
SIM_DT_S = 0.01
SIM_STEPS = 150
VIDEO_FPS = int(round(1 / SIM_DT_S))  # 1 step = 1 フレームで実時間再生にする
CAMERA_RES = (640, 480)
CAMERA_POS = (2.5, -2.5, 1.5)
CAMERA_LOOKAT = (0.0, 0.0, 0.3)
CAMERA_FOV_DEG = 40
BOX_SIZE_M = 0.2
BOX_DROP_HEIGHT_M = 1.0
# 最初と最後のフレームの画素差の平均 (0-255)。箱の落下で約 2、静止した連続フレームでは 0 になる。
# 圧縮ノイズで超えず、落下が描画されていれば確実に超える値にする
MIN_MEAN_FRAME_CHANGE = 0.5


class RecordingError(RuntimeError):
    """録画ファイルが生成されなかった、または中身が描画結果になっていないことを表す。"""


def load_video_frames(path: Path) -> np.ndarray:
    """mp4 の全フレームを (フレーム数, 高さ, 幅, 3) の RGB 配列として読み込む。"""
    with av.open(str(path)) as container:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    return np.stack(frames)


def verify_recording(path: Path) -> None:
    """録画が期待どおりのフレーム数・解像度で、箱の落下による映像の変化を含むことを検証する。

    Raises:
        RecordingError: ファイルが無い、フレーム数・解像度が異なる、または映像が変化していない。
    """
    if not path.is_file() or path.stat().st_size == 0:
        raise RecordingError(f"録画ファイルが生成されていない: {path}")
    frames = load_video_frames(path)
    frame_count, height, width, _ = frames.shape
    if frame_count != SIM_STEPS or (width, height) != CAMERA_RES:
        raise RecordingError(
            f"録画の形式が異なる: frames={frame_count} size={width}x{height} "
            f"(期待 frames={SIM_STEPS} size={CAMERA_RES[0]}x{CAMERA_RES[1]})"
        )
    mean_change = float(np.abs(frames[-1].astype(np.int16) - frames[0].astype(np.int16)).mean())
    if mean_change < MIN_MEAN_FRAME_CHANGE:
        raise RecordingError(
            f"最初と最後のフレームがほぼ同じで、落下が描画されていない: "
            f"mean_change={mean_change:.3f} (下限 {MIN_MEAN_FRAME_CHANGE})"
        )
    print(f"saved {path}: frames={frame_count} size={width}x{height} mean_change={mean_change:.3f}")


def main() -> None:
    """箱の落下を録画し、mp4 が生成されたことを検証する。"""
    gs.init(backend=gs.cuda, logging_level="warning")
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

    verify_recording(OUTPUT_PATH)
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
