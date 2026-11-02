"""Genesis のオフスクリーンカメラで評価動画（mp4）を撮る。"""

from pathlib import Path

from genesis.vis.camera import Camera


class GenesisVideoRecorder:
    """カメラの録画を開始・終了する。録画中は物理ステップのたびにカメラが 1 フレーム描画する。"""

    def __init__(self, camera: Camera, video_path: Path, control_dt_s: float) -> None:
        """録画の保存先とフレームレートを決める。録画は start() まで始めない。

        Args:
            camera: シーンに置いたカメラ。
            video_path: 保存先の mp4。親ディレクトリが無ければ作る。
            control_dt_s: 1 制御ステップの時間 [s]。1 ステップを 1 フレームにして実時間で再生されるようにする。
        """
        self._camera = camera
        self._video_path = video_path
        self._fps = round(1.0 / control_dt_s)

    def start(self) -> None:
        """録画を始める。"""
        self._video_path.parent.mkdir(parents=True, exist_ok=True)
        self._camera.start_recording(save_to_filename=str(self._video_path), fps=self._fps)

    def stop(self) -> None:
        """録画を終えて mp4 を書き出す。"""
        self._camera.stop_recording()
