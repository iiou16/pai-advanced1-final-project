"""Genesis の CUDA バックエンドで剛体シーンを step できることを確かめるスモークテスト.

平面上に箱を落とし、重力と接触で床に静止するかを検証する。
CUDA が使えない場合は CPU に切り替えず、Genesis の初期化で例外を送出する。
失敗時は例外を送出して非ゼロ終了する。
"""

import math
import time

import genesis as gs

SIM_DT_S = 0.01
SIM_STEPS = 200  # 2 秒。1 m 落下と接触後の減衰に十分な長さ
BOX_SIZE_M = 0.2
BOX_DROP_HEIGHT_M = 1.0
# 接触ソルバの貫入・弾みを許容する誤差
REST_HEIGHT_TOLERANCE_M = 0.01


class RigidSmokeError(RuntimeError):
    """剛体シーンが物理的に妥当な状態に収束しなかったことを表す。"""


def build_scene() -> tuple[gs.Scene, object]:
    """平面と落下させる箱からなるシーンを構築して返す。"""
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=SIM_DT_S),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    box = scene.add_entity(
        gs.morphs.Box(size=(BOX_SIZE_M,) * 3, pos=(0.0, 0.0, BOX_DROP_HEIGHT_M)),
    )
    scene.build()
    return scene, box


def main() -> None:
    """CUDA 上で箱を落下させ、床に静止した高さを検証する。"""
    gs.init(backend=gs.cuda, logging_level="warning")
    scene, box = build_scene()

    started = time.perf_counter()
    for _ in range(SIM_STEPS):
        scene.step()
    wall_time_s = time.perf_counter() - started

    final_height_m = float(box.get_pos()[2])
    expected_height_m = BOX_SIZE_M / 2
    print(f"backend={gs.backend.name} steps={SIM_STEPS} wall_time_s={wall_time_s:.3f}")
    print(f"box z: final={final_height_m:.4f} expected={expected_height_m:.4f}")
    # 発散して NaN になると差の比較が常に偽になるため、有限値であることを先に確かめる
    is_at_rest = (
        math.isfinite(final_height_m)
        and abs(final_height_m - expected_height_m) <= REST_HEIGHT_TOLERANCE_M
    )
    if not is_at_rest:
        raise RigidSmokeError(
            f"箱が床に静止していない: z={final_height_m:.4f} m (期待 {expected_height_m:.4f} m)"
        )
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
