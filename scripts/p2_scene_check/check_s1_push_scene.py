"""シーン S1_push の箱の物理を確かめる（受け入れテスト T02）。

箱が重力で床に置かれ、押すと滑り、離すと止まること、Go2 と箱の接触が取れることを、
箱に外力をかけて確かめる。箱の位置は書き換えない（要件 3.1: 箱は接触と摩擦だけで動かす）。
押す速さは Go2 が実際に押す程度（1 m/s 未満）にとどめ、その間に箱が浮き上がったり傾いたりしないことも確かめる。
GPU で学習が走っていても取り合わないよう、Genesis を CPU で初期化し、並列環境 1 つで動かす。

実行: uv run python scripts/p2_scene_check/check_s1_push_scene.py
"""

from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch
from genesis.utils.geom import default_friction

from cooperative_go2.entities.push_evaluation import tilt_deg
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_scene_settings
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body
from cooperative_go2.interface_adapters.genesis_push_scene import PushScene, build_push_scene

GAIT_CONFIG_PATH = Path("configs/gait.yaml")
SCENE_CONFIG_PATH = Path("configs/scenes/s1_push.yaml")
SEED = 0
NUM_ENVS = 1
GRAVITY_MPS2 = 9.81
SETTLE_S = 1.0
HOLD_S = 1.0
SLIDE_S = 0.5
RELEASE_S = 1.0
# 静止摩擦力（摩擦係数 × 質量 × 重力加速度）に対する外力の比。下回れば動かず、上回れば滑るはず。
# 上回る側は、0.5 秒で約 1 m/s に達する程度にする（Go2 が押す速さと同程度）
BELOW_FRICTION_FORCE_RATIO = 0.5
ABOVE_FRICTION_FORCE_RATIO = 1.2
# 判定のしきい値
RESTING_HEIGHT_TOLERANCE_M = 0.01  # 床に置かれた箱の中心の高さのずれ
STATIC_MOVEMENT_TOLERANCE_M = 0.01  # 静止摩擦で止まっているとみなす移動量
RESTING_SPEED_MPS = 0.02  # 静止とみなす速さ
MOVED_DISTANCE_M = 0.05  # 滑ったとみなす移動量
# 水平に押された箱は理想的には浮かず傾かない（摩擦係数 1.0 < 箱の幅 / 高さ のため倒れない）。
# 接触の計算で生じるわずかなめり込み・揺れだけを許す
MAX_LIFT_M = 0.002  # 押している間に許す箱の浮き上がり
MAX_TILT_DEG = 0.5  # 押している間に許す箱の傾き
# 箱を Go2 へ押し戻して接触を待つ最長の時間
MAX_TIME_TO_CONTACT_S = 2.0
WORLD_UP_AXIS_INDEX = 2


class SceneCheckError(RuntimeError):
    """シーン S1_push の物理が期待どおりに振る舞わなかったことを表す。"""


@dataclass(frozen=True)
class BoxExcursion:
    """物理を進めた間に、床に置かれた姿勢から箱が最も離れた量。

    Attributes:
        max_lift_m: 箱の中心が床に置かれた高さより上がった量の最大 [m]。
        max_tilt_deg: 箱の上向きの軸の、鉛直からの傾きの最大 [deg]。
        floor_contact_lost_steps: 箱が床との接触を失った（浮いていた）制御ステップの数。
    """

    max_lift_m: float
    max_tilt_deg: float
    floor_contact_lost_steps: int


class SceneProbe:
    """Go2 を既定姿勢で立たせたまま、箱に外力をかけて物理を進め、箱の状態を読む。"""

    def __init__(self, scene: PushScene, body: GenesisGo2Body, control_dt_s: float, resting_height_m: float) -> None:
        self._scene = scene
        self._body = body
        self._control_dt_s = control_dt_s
        self._resting_height_m = resting_height_m
        self._standing_targets = body.default_dof_pos.expand(NUM_ENVS, -1)

    def run(self, duration_s: float, box_force_world_n: tuple[float, float, float]) -> BoxExcursion:
        """duration_s の間、毎ステップ箱に同じ外力をかけて物理を進め、その間の箱の浮き上がりと傾きの最大を返す。

        Genesis は外力を 1 ステップごとに消すため、毎ステップかけ直す。
        """
        force = torch.tensor([[box_force_world_n]], dtype=gs.tc_float, device=gs.device)
        max_lift_m = 0.0
        max_tilt_deg = 0.0
        floor_contact_lost_steps = 0
        for _ in range(round(duration_s / self._control_dt_s)):
            self._scene.box.apply_links_external_wrench(force=force)
            self._body.apply_joint_targets(self._standing_targets)
            self._scene.step()
            if bool(self._scene.diverged_mask().any()):
                raise SceneCheckError("物理計算が発散した")
            max_lift_m = max(max_lift_m, float(self.box_position()[WORLD_UP_AXIS_INDEX]) - self._resting_height_m)
            max_tilt_deg = max(max_tilt_deg, self.box_tilt_deg())
            floor_contact_lost_steps += int(not bool(self._scene.box_ground_contact_mask().all()))
        return BoxExcursion(
            max_lift_m=max_lift_m, max_tilt_deg=max_tilt_deg, floor_contact_lost_steps=floor_contact_lost_steps
        )

    def run_until_contact(self, timeout_s: float, box_force_world_n: tuple[float, float, float]) -> float:
        """Go2 と箱が接触するまで外力をかけ続け、接触までの時間 [s] を返す。"""
        for step_index in range(round(timeout_s / self._control_dt_s)):
            self.run(self._control_dt_s, box_force_world_n)
            if bool(self._scene.robot_box_contact_mask().all()):
                return (step_index + 1) * self._control_dt_s
        raise SceneCheckError(f"{timeout_s} 秒以内に Go2 と箱の接触を検出できなかった")

    @property
    def resting_height_m(self) -> float:
        """床に置かれた箱の中心の高さ [m]。"""
        return self._resting_height_m

    def box_position(self) -> torch.Tensor:
        """箱の中心の位置 (3,) [m]。"""
        return self._scene.box.get_pos()[0].cpu()

    def box_speed(self) -> float:
        """箱の並進の速さ [m/s]。"""
        return float(torch.linalg.vector_norm(self._scene.box.get_vel()[0]))

    def box_tilt_deg(self) -> float:
        """箱の上向きの軸の、鉛直からの傾き [deg]。"""
        _, x, y, _ = self._scene.box.get_quat()[0].cpu()
        # 四元数 (w, x, y, z) で回した z 軸の、ワールドの z 成分
        up_axis_world_z = 1 - 2 * (x * x + y * y)
        return float(tilt_deg(up_axis_world_z.unsqueeze(0))[0])


def require_box_stays_flat(excursion: BoxExcursion, phase: str) -> None:
    print(
        f"[{phase}] 浮き上がり最大 {excursion.max_lift_m:.4f} m、傾き最大 {excursion.max_tilt_deg:.2f}°、"
        f"床との接触を失ったステップ {excursion.floor_contact_lost_steps}"
    )
    if excursion.floor_contact_lost_steps > 0:
        raise SceneCheckError(f"{phase}の間に箱が床との接触を失った（接触の欠落）")
    if excursion.max_lift_m > MAX_LIFT_M or excursion.max_tilt_deg > MAX_TILT_DEG:
        raise SceneCheckError(f"{phase}の間に箱が浮き上がった、または傾いた")


def check_box_rests_on_floor(probe: SceneProbe) -> None:
    probe.run(SETTLE_S, (0.0, 0.0, 0.0))
    height_error = abs(float(probe.box_position()[WORLD_UP_AXIS_INDEX]) - probe.resting_height_m)
    speed = probe.box_speed()
    print(f"[置く] 中心の高さのずれ {height_error:.4f} m、速さ {speed:.4f} m/s")
    if height_error > RESTING_HEIGHT_TOLERANCE_M or speed > RESTING_SPEED_MPS:
        raise SceneCheckError("箱が床の上で静止していない")


def check_static_friction_holds(probe: SceneProbe, friction_force_n: float) -> None:
    start = probe.box_position()
    probe.run(HOLD_S, (BELOW_FRICTION_FORCE_RATIO * friction_force_n, 0.0, 0.0))
    moved = float(torch.linalg.vector_norm(probe.box_position()[:2] - start[:2]))
    print(f"[静止摩擦] 静止摩擦力の {BELOW_FRICTION_FORCE_RATIO} 倍で {HOLD_S} 秒押した移動量 {moved:.4f} m")
    if moved > STATIC_MOVEMENT_TOLERANCE_M:
        raise SceneCheckError("静止摩擦力より弱い力で箱が動いた")


def check_box_slides_and_stops(probe: SceneProbe, friction_force_n: float) -> None:
    start = probe.box_position()
    excursion = probe.run(SLIDE_S, (ABOVE_FRICTION_FORCE_RATIO * friction_force_n, 0.0, 0.0))
    pushed = float(probe.box_position()[0] - start[0])
    print(
        f"[滑る] 静止摩擦力の {ABOVE_FRICTION_FORCE_RATIO} 倍で {SLIDE_S} 秒押した移動量 {pushed:.4f} m、"
        f"離す直前の速さ {probe.box_speed():.3f} m/s"
    )
    if pushed < MOVED_DISTANCE_M:
        raise SceneCheckError("静止摩擦力より強い力で押しても箱が滑らなかった")
    require_box_stays_flat(excursion, "滑る")
    probe.run(RELEASE_S, (0.0, 0.0, 0.0))
    speed = probe.box_speed()
    print(f"[止まる] 離して {RELEASE_S} 秒後の速さ {speed:.4f} m/s")
    if speed > RESTING_SPEED_MPS:
        raise SceneCheckError("外力を除いても箱が止まらない")


def check_robot_box_contact(probe: SceneProbe, friction_force_n: float) -> None:
    # 箱を Go2（原点）へ向けて押し戻し、接触が検出されること、衝突候補が上限を超えないことを確かめる
    elapsed = probe.run_until_contact(MAX_TIME_TO_CONTACT_S, (-ABOVE_FRICTION_FORCE_RATIO * friction_force_n, 0.0, 0.0))
    print(f"[接触] 押し戻し始めて {elapsed:.2f} 秒で Go2 と箱の接触を検出")


def main() -> None:
    gait_config = load_gait_settings(GAIT_CONFIG_PATH).gait
    scene_settings = load_push_scene_settings(SCENE_CONFIG_PATH)
    init_genesis_on_cpu(SEED)
    scene = build_push_scene(gait_config, scene_settings, NUM_ENVS)
    body = GenesisGo2Body(scene.robot, gait_config.robot)
    box = scene_settings.box
    probe = SceneProbe(scene, body, gait_config.env.control_dt_s, resting_height_m=box.size_m[2] / 2)

    # 接触する 2 面の摩擦は大きい方が使われる。床は Genesis の既定値のまま
    effective_friction = max(box.friction, default_friction())
    friction_force_n = effective_friction * box.mass_kg * GRAVITY_MPS2
    print(f"箱: {box.size_m} m、{box.mass_kg} kg、床との摩擦 {effective_friction}、静止摩擦力 {friction_force_n:.1f} N")

    check_box_rests_on_floor(probe)
    check_static_friction_holds(probe, friction_force_n)
    check_box_slides_and_stops(probe, friction_force_n)
    check_robot_box_contact(probe, friction_force_n)
    print("T02: すべての確認に合格")


if __name__ == "__main__":
    main()
