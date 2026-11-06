"""Go2 が 1 台・2 台で押して動かせる箱の質量を、壁の無い平面で測る（重い箱の質量を決めるため）。

協調の手順（位置合わせ・同時開始）の出来に左右されずに物理的な押す力の上限だけを比べるため、
ロボットを箱の後ろに並べて置き、全員に同じ前進の指令を出し続ける（向きだけを +x に保つ）。
質量と指令の速さを変えて、決めた時間内の箱の移動量・回転・ロボットの転倒を並列環境ごとに記録する。

- 箱は 2 台が横に並んで押せる幅にする。1 台の条件も同じ箱を中央から押す（1 台と 2 台を同じ箱で比べるため）。
- 箱と床の摩擦は床（歩行の学習時の Genesis の既定値 1.0）と同じにする。大きい方が使われるため、これより下げられない。
- 指令の速さは、規則ベースの押す速さ（0.3 m/s）、高レベル方策の指令の上限（0.6 m/s）、歩行方策が学習した上限（1.0 m/s）。
  1 台が押せないことを、弱い指令のときだけで判断しないため。
- 質量はシーンを作り直さずに箱の総質量を書き換えて変える（慣性も同じ比で変わる）。

実行: uv run python scripts/p4_physics_check/sweep_team_push_capacity.py --num-robots 2 --device cuda \\
        --output-csv outputs/p4_physics_check/capacity-2robots.csv
"""

import argparse
import csv
import dataclasses
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch
from genesis.engine.entities import RigidEntity

from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.planar_frame import wrap_angle_rad, yaw_from_quat_wxyz
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import load_gait_settings, load_push_scene_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_flat_scene import ENABLE_SELF_COLLISION, GO2_URDF, PLANE_URDF, FlatScene
from cooperative_go2.interface_adapters.genesis_go2_body import GenesisGo2Body

GAIT_RUN_DIR = Path("outputs/gait/gait-command-tracking")
GAIT_CHECKPOINT = 1500
# 摩擦円錐のモデルは箱押しのシーン（S1_push）と同じにする
SCENE_CONFIG_PATH = Path("configs/scenes/s1_push.yaml")
SEED = 4001
# 奥行き (x) × 幅 (y) × 高さ (z)。幅は Go2（胴体の幅 約 0.31 m）2 台が 0.5 m 間隔で並んで収まる大きさ
BOX_SIZE_M = (0.4, 1.0, 0.3)
BOX_FRICTION = 1.0
BOX_CENTER_X_M = 1.0
# Go2 の胴体の原点から前端までの距離 [m] と、押し始める前の前端と箱の間のすき間 [m]
ROBOT_FRONT_FROM_BASE_M = 0.35
START_GAP_M = 0.15
# 台数ごとの、箱の中心線からの各ロボットの横の位置 [m]
LATERAL_OFFSETS_M = {1: (0.0,), 2: (0.25, -0.25)}
# 1 台の歩行の 20 に、箱と床・箱と Go2・Go2 どうしの衝突候補の分を足す。超えたら例外で止まる
MAX_COLLISION_PAIRS = 80
# 向きを +x に保つための旋回の比例ゲイン [1/s] と上限 [rad/s]（規則ベースの箱押しと同じ値）
HEADING_GAIN_PER_S = 2.0
MAX_YAW_RATE_RADPS = 1.0
# 歩行方策の学習時の転倒の判定と同じ傾き
FALL_TILT_DEG = 10.0
X_AXIS, Y_AXIS = 0, 1
ROLL_AXIS, PITCH_AXIS = 0, 1
UPRIGHT_QUAT_WXYZ = (1.0, 0.0, 0.0, 0.0)


class CapacityCheckError(RuntimeError):
    """計測を続けられない状態（物理計算の発散、質量の設定の失敗など）を表す。"""


@dataclass(frozen=True)
class SweepRequest:
    """計測の条件。

    Attributes:
        num_robots: 押すロボットの数（1 か 2）。
        masses_kg: 試す箱の質量 [kg] の並び。
        speeds_mps: 試す前進の指令の速さ [m/s] の並び。
        repeats: 質量と速さの組ごとの試行数（リセット時の関節角のノイズだけが違う）。
        duration_s: 1 試行の長さ [s]（シミュレーション時間）。
    """

    num_robots: int
    masses_kg: tuple[float, ...]
    speeds_mps: tuple[float, ...]
    repeats: int
    duration_s: float


@dataclass(frozen=True)
class TrialRow:
    """1 試行（並列環境 1 つ）の結果。CSV の 1 行になる。

    Attributes:
        num_robots: 押したロボットの数。
        mass_kg: 箱の質量 [kg]。
        command_vx_mps: 全ロボットに出した前進の指令 [m/s]。
        repeat: 同じ質量・速さの中の試行の番号。
        box_dx_half_m: 試行の前半（duration_s / 2）での箱の +x 方向の移動量 [m]。
        box_dx_m: 試行の終わりでの箱の +x 方向の移動量 [m]。
        box_dy_m: 試行の終わりでの箱の横方向の移動量 [m]。
        box_final_yaw_deg: 試行の終わりの箱のヨー角 [deg]。
        box_max_abs_yaw_deg: 試行中の箱のヨー角の絶対値の最大 [deg]。
        robot_fell: いずれかのロボットが試行中に転倒の傾きを超えたか。
    """

    num_robots: int
    mass_kg: float
    command_vx_mps: float
    repeat: int
    box_dx_half_m: float
    box_dx_m: float
    box_dy_m: float
    box_final_yaw_deg: float
    box_max_abs_yaw_deg: float
    robot_fell: bool


@dataclass(frozen=True)
class TeamScene:
    """床・Go2 を num_robots 台・箱を置いた構築済みのシーン。"""

    stepper: FlatScene
    robots: tuple[RigidEntity, ...]
    box: RigidEntity


def build_team_scene(gait_config: GaitConfig, num_robots: int, num_envs: int, first_mass_kg: float) -> TeamScene:
    """箱の後ろに Go2 を横に並べたシーンを num_envs 個の並列環境で構築する。前提: gs.init() 済み。"""
    scene_settings = load_push_scene_settings(SCENE_CONFIG_PATH)
    env_config = gait_config.env
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=env_config.control_dt_s, substeps=env_config.substeps),
        rigid_options=gs.options.RigidOptions(
            enable_self_collision=ENABLE_SELF_COLLISION,
            max_collision_pairs=MAX_COLLISION_PAIRS,
            friction_cone=getattr(gs.friction_cone, scene_settings.genesis.friction_cone),
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.URDF(file=PLANE_URDF, fixed=True))
    robots = tuple(
        scene.add_entity(
            gs.morphs.URDF(file=GO2_URDF, pos=position, quat=gait_config.robot.base_init_quat_wxyz)
        )
        for position in robot_initial_positions(gait_config, num_robots)
    )
    box_volume_m3 = BOX_SIZE_M[0] * BOX_SIZE_M[1] * BOX_SIZE_M[2]
    box = scene.add_entity(
        gs.morphs.Box(size=BOX_SIZE_M, pos=box_initial_position()),
        material=gs.materials.Rigid(rho=first_mass_kg / box_volume_m3, friction=BOX_FRICTION),
    )
    scene.build(n_envs=num_envs)
    return TeamScene(stepper=FlatScene(scene=scene, robot=robots[0], camera=None), robots=robots, box=box)


def robot_initial_positions(gait_config: GaitConfig, num_robots: int) -> list[tuple[float, float, float]]:
    """各ロボットの胴体の初期位置。前端と箱の後面の間を START_GAP_M あける。"""
    box_rear_x_m = BOX_CENTER_X_M - BOX_SIZE_M[X_AXIS] / 2
    base_x_m = box_rear_x_m - START_GAP_M - ROBOT_FRONT_FROM_BASE_M
    base_z_m = gait_config.robot.base_init_pos_m[2]
    return [(base_x_m, offset, base_z_m) for offset in LATERAL_OFFSETS_M[num_robots]]


def box_initial_position() -> tuple[float, float, float]:
    """箱の中心の初期位置（床に置いた高さ）。"""
    return (BOX_CENTER_X_M, 0.0, BOX_SIZE_M[2] / 2)


class CapacitySweep:
    """構築済みのシーンで、質量ごとに全環境をリセットして押させ、環境ごとの結果を返す。"""

    def __init__(self, gait_config: GaitConfig, team: TeamScene, request: SweepRequest) -> None:
        self._gait_config = gait_config
        self._team = team
        self._request = request
        self._num_envs = len(request.speeds_mps) * request.repeats
        self._bodies = [
            GenesisGo2Body(robot, dataclasses.replace(gait_config.robot, base_init_pos_m=position))
            for robot, position in zip(
                team.robots, robot_initial_positions(gait_config, request.num_robots), strict=True
            )
        ]
        self._controllers = [
            load_gait_controller(GAIT_RUN_DIR, GAIT_CHECKPOINT, self._num_envs, gs.device) for _ in team.robots
        ]
        self._generator = make_seeded_generator(SEED)
        # 環境 i の指令の速さは speeds_mps[i // repeats]
        self._speeds = torch.tensor(request.speeds_mps, dtype=gs.tc_float, device=gs.device).repeat_interleave(
            request.repeats
        )

    def run(self, mass_kg: float) -> list[TrialRow]:
        """箱の質量を mass_kg にして全環境で 1 試行ずつ押させ、環境ごとの結果の行を返す。"""
        self._set_box_mass(mass_kg)
        self._reset()
        box = self._team.box
        start_xy = box.get_pos()[:, : Y_AXIS + 1].clone()
        fell = torch.zeros(self._num_envs, dtype=torch.bool, device=gs.device)
        max_abs_box_yaw_deg = torch.zeros(self._num_envs, device=gs.device)
        num_steps = round(self._request.duration_s / self._gait_config.env.control_dt_s)
        half_dx = torch.zeros(self._num_envs, device=gs.device)
        pending_targets = [body.default_dof_pos.expand(self._num_envs, -1).clone() for body in self._bodies]
        for step_index in range(num_steps):
            for robot_index, (robot, body, controller) in enumerate(
                zip(self._team.robots, self._bodies, self._controllers, strict=True)
            ):
                state = body.read_state()
                fell |= (state.base_euler_deg[:, [ROLL_AXIS, PITCH_AXIS]].abs() > FALL_TILT_DEG).any(dim=1)
                targets = controller.compute_joint_targets(state.proprioception, self._commands(robot))
                # 歩行方策の学習時と同じく、1 制御周期前に決めた目標を適用する
                body.apply_joint_targets(pending_targets[robot_index])
                pending_targets[robot_index] = targets
            self._team.stepper.step()
            if bool(self._team.stepper.diverged_mask().any()):
                raise CapacityCheckError(f"質量 {mass_kg} kg で物理計算が発散した（ステップ {step_index}）")
            max_abs_box_yaw_deg = torch.maximum(
                max_abs_box_yaw_deg, torch.rad2deg(yaw_from_quat_wxyz(box.get_quat()).abs())
            )
            if step_index + 1 == num_steps // 2:
                half_dx = box.get_pos()[:, X_AXIS] - start_xy[:, X_AXIS]
        displacement = box.get_pos()[:, : Y_AXIS + 1] - start_xy
        final_yaw_deg = torch.rad2deg(yaw_from_quat_wxyz(box.get_quat()))
        return [
            TrialRow(
                num_robots=self._request.num_robots,
                mass_kg=mass_kg,
                command_vx_mps=round(float(self._speeds[env]), 3),
                repeat=env % self._request.repeats,
                box_dx_half_m=round(float(half_dx[env]), 4),
                box_dx_m=round(float(displacement[env, X_AXIS]), 4),
                box_dy_m=round(float(displacement[env, Y_AXIS]), 4),
                box_final_yaw_deg=round(float(final_yaw_deg[env]), 2),
                box_max_abs_yaw_deg=round(float(max_abs_box_yaw_deg[env]), 2),
                robot_fell=bool(fell[env]),
            )
            for env in range(self._num_envs)
        ]

    def _commands(self, robot: RigidEntity) -> torch.Tensor:
        heading_error = wrap_angle_rad(-yaw_from_quat_wxyz(robot.get_quat()))
        yaw_rate = torch.clamp(HEADING_GAIN_PER_S * heading_error, -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS)
        return torch.stack((self._speeds, torch.zeros_like(self._speeds), yaw_rate), dim=1)

    def _set_box_mass(self, mass_kg: float) -> None:
        box = self._team.box
        box.set_mass(mass_kg)
        actual_kg = float(box.get_mass().flatten()[0])
        if abs(actual_kg - mass_kg) > 1e-3 * mass_kg:
            raise CapacityCheckError(f"箱の質量を {mass_kg} kg にできなかった（読み出し {actual_kg} kg）")

    def _reset(self) -> None:
        all_envs = torch.ones(self._num_envs, dtype=torch.bool, device=gs.device)
        noise_rad = self._gait_config.env.reset_joint_pos_noise_rad
        for body, controller in zip(self._bodies, self._controllers, strict=True):
            dof_pos = sample_reset_dof_pos(body.default_dof_pos, self._num_envs, noise_rad, self._generator)
            body.reset(all_envs, dof_pos)
            controller.reset_where(all_envs)
        box = self._team.box
        position = torch.tensor(box_initial_position(), dtype=gs.tc_float, device=gs.device)
        box.set_pos(position.expand(self._num_envs, -1), envs_idx=all_envs, zero_velocity=True)
        upright = torch.tensor(UPRIGHT_QUAT_WXYZ, dtype=gs.tc_float, device=gs.device)
        box.set_quat(upright.expand(self._num_envs, -1), envs_idx=all_envs, zero_velocity=True)


def summarize(rows: list[TrialRow]) -> None:
    """質量と速さの組ごとに、箱の移動量の平均・最小・最大と転倒の割合を表示する。"""
    groups: dict[tuple[float, float], list[TrialRow]] = {}
    for row in rows:
        groups.setdefault((row.mass_kg, row.command_vx_mps), []).append(row)
    print("台数 質量[kg] 指令[m/s] 移動量 平均/最小/最大[m] 前半[m] 箱のヨーの最大 試行平均/全体[deg] 転倒率")
    for (mass_kg, speed), group in sorted(groups.items()):
        dx = [row.box_dx_m for row in group]
        half = [row.box_dx_half_m for row in group]
        yaw = [row.box_max_abs_yaw_deg for row in group]
        fall_rate = sum(row.robot_fell for row in group) / len(group)
        print(
            f"{group[0].num_robots} {mass_kg:>5} {speed:>5} "
            f"{sum(dx) / len(dx):6.3f}/{min(dx):6.3f}/{max(dx):6.3f} {sum(half) / len(half):6.3f} "
            f"{sum(yaw) / len(yaw):6.1f}/{max(yaw):6.1f} {fall_rate:5.2f}"
        )


GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}


@dataclass(frozen=True)
class CommandLine:
    """コマンドラインで受け取った計測の条件・出力先・デバイス。"""

    request: SweepRequest
    output_csv: Path
    device: str


def parse_args() -> CommandLine:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-robots", type=int, choices=sorted(LATERAL_OFFSETS_M), required=True)
    parser.add_argument("--masses-kg", type=float, nargs="+", default=[2.0, 3.0, 4.0, 5.0, 6.0, 8.0])
    parser.add_argument("--speeds-mps", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    # 学習・評価と同じ cuda を基本にする。cpu はコードの動作を少数の環境で確かめるために使う
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    args = parser.parse_args()
    request = SweepRequest(
        num_robots=args.num_robots,
        masses_kg=tuple(args.masses_kg),
        speeds_mps=tuple(args.speeds_mps),
        repeats=args.repeats,
        duration_s=args.duration_s,
    )
    return CommandLine(request=request, output_csv=args.output_csv, device=args.device)


def main() -> None:
    command_line = parse_args()
    request, output_csv = command_line.request, command_line.output_csv
    if output_csv.exists():
        raise FileExistsError(f"出力先が既にある（上書きしない）: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    GENESIS_INITIALIZERS[command_line.device](SEED)
    gait_config = load_gait_settings(GAIT_RUN_DIR / RUN_CONFIG_FILENAME).gait
    num_envs = len(request.speeds_mps) * request.repeats
    team = build_team_scene(gait_config, request.num_robots, num_envs, request.masses_kg[0])
    sweep = CapacitySweep(gait_config, team, request)
    rows: list[TrialRow] = []
    for mass_kg in request.masses_kg:
        rows.extend(sweep.run(mass_kg))
        print(f"質量 {mass_kg} kg を計測した")
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in dataclasses.fields(TrialRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    summarize(rows)


if __name__ == "__main__":
    main()
