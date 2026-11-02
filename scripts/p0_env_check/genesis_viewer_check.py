"""Genesis の対話 Viewer が WSLg 上で開けることを確かめるスモークテスト.

数秒間ウィンドウを表示して自動終了する。Viewer の生成や描画に失敗した場合は
Genesis の例外がそのまま送出され、非ゼロ終了する。
"""

import genesis as gs

SIM_DT_S = 0.01
SIM_STEPS = 300  # 3 秒。ウィンドウ表示を目視確認できる長さ
BOX_SIZE_M = 0.2
BOX_DROP_HEIGHT_M = 1.0


def main() -> None:
    """Viewer 付きでシーンを構築し、一定ステップ描画して終了する。"""
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT_S),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.5, -2.5, 1.5), camera_lookat=(0.0, 0.0, 0.3)
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    scene.add_entity(gs.morphs.Box(size=(BOX_SIZE_M,) * 3, pos=(0.0, 0.0, BOX_DROP_HEIGHT_M)))
    scene.build()
    for _ in range(SIM_STEPS):
        scene.step()
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
