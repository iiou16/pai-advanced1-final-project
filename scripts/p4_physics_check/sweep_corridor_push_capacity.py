"""通路のシーンで、Go2 が押して動かせる箱の質量と、通路を開けられるかを測る（D30 の重い箱を壁のある通路で確かめる）。

平面での計測（sweep_team_push_capacity.py）と同じく、協調の手順の出来に左右されずに押す力の上限だけを比べるため、
全ロボットに同じ前進の指令を出し続ける（向きだけを +x に保つ）。通路の壁との接触は箱を止める摩擦にも、
回転を抑えるガイドにもなりうるため、E3〜E5 を行う通路で測り直す。
台数とロボットの配置はシーンの設定（s2_corridor.yaml は 1 台、s3_team.yaml は 2 台）で決め、質量だけを書き換えて変える。

実行（Genesis を cuda で使う前に、同じシェルで source scripts/setup/env.sh する）:
    uv run python scripts/p4_physics_check/sweep_corridor_push_capacity.py --device cuda \\
        --scene-config configs/scenes/s3_team.yaml --output-csv outputs/p4_physics_check/corridor-2robots.csv
"""

import argparse
import csv
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch

from cooperative_go2.entities.corridor import is_corridor_open
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.planar_frame import wrap_angle_rad
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import CorridorSceneSettings, load_corridor_scene_settings, load_gait_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation

GAIT_RUN_DIR = Path("outputs/gait/gait-command-tracking")
GAIT_CHECKPOINT = 1500
SEED = 4002
# 向きを +x に保つための旋回の比例ゲイン [1/s] と上限 [rad/s]（規則ベースの箱押しと同じ値）
HEADING_GAIN_PER_S = 2.0
MAX_YAW_RATE_RADPS = 1.0
# 歩行方策の学習時の転倒の判定と同じ傾き
FALL_TILT_DEG = 10.0
X_AXIS, Y_AXIS = 0, 1
ROLL_AXIS, PITCH_AXIS = 0, 1
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}


class CapacityCheckError(RuntimeError):
    """計測を続けられない状態（物理計算の発散、質量の設定の失敗など）を表す。"""


@dataclass(frozen=True)
class SweepRequest:
    """計測の条件。

    Attributes:
        masses_kg: 試す箱の質量 [kg] の並び。
        speeds_mps: 試す前進の指令の速さ [m/s] の並び。
        repeats: 質量と速さの組ごとの試行数（リセット時の関節角のノイズだけが違う）。
        duration_s: 1 試行の長さ [s]（シミュレーション時間）。
    """

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
        box_dx_m: 試行の終わりでの箱の +x 方向の移動量 [m]。
        box_dy_m: 試行の終わりでの箱の横方向の移動量 [m]。
        box_final_yaw_deg: 試行の終わりの箱のヨー角 [deg]。
        box_max_abs_yaw_deg: 試行中の箱のヨー角の絶対値の最大 [deg]。
        corridor_opened: 試行中に通路が開いた（箱が通行領域から出た）か。
        open_time_s: 通路が初めて開いた時刻 [s]。開かなければ None（CSV では空欄）。
        robot_fell: いずれかのロボットが試行中に転倒の傾きを超えたか。
    """

    num_robots: int
    mass_kg: float
    command_vx_mps: float
    repeat: int
    box_dx_m: float
    box_dy_m: float
    box_final_yaw_deg: float
    box_max_abs_yaw_deg: float
    corridor_opened: bool
    open_time_s: float | None
    robot_fell: bool


class CorridorCapacitySweep:
    """構築済みの通路のシーンで、質量ごとに全環境をリセットして押させ、環境ごとの結果を返す。"""

    def __init__(
        self, gait_config: GaitConfig, settings: CorridorSceneSettings, request: SweepRequest
    ) -> None:
        self._gait_config = gait_config
        self._settings = settings
        self._request = request
        self._num_envs = len(request.speeds_mps) * request.repeats
        self._simulation = GenesisCorridorSimulation.build(
            gait_config, settings, CorridorSceneRequest(num_envs=self._num_envs, camera=None)
        )
        self._num_robots = len(self._simulation.robot_names)
        self._controllers = tuple(
            load_gait_controller(GAIT_RUN_DIR, GAIT_CHECKPOINT, self._num_envs, gs.device)
            for _ in range(self._num_robots)
        )
        self._generator = make_seeded_generator(SEED)
        # 環境 i の指令の速さは speeds_mps[i // repeats]
        self._speeds = torch.tensor(request.speeds_mps, dtype=gs.tc_float, device=gs.device).repeat_interleave(
            request.repeats
        )

    def run(self, mass_kg: float) -> list[TrialRow]:
        """箱の質量を mass_kg にして全環境で 1 試行ずつ押させ、環境ごとの結果の行を返す。"""
        simulation = self._simulation
        simulation.set_box_mass(mass_kg)
        self._reset()
        corridor = self._settings.scene.corridor
        control_dt_s = self._gait_config.env.control_dt_s
        start_xy = simulation.read_box_planar().position_xy_m.clone()
        fell = torch.zeros(self._num_envs, dtype=torch.bool, device=gs.device)
        max_abs_box_yaw_deg = torch.zeros(self._num_envs, device=gs.device)
        open_step = torch.full((self._num_envs,), -1, dtype=torch.int64, device=gs.device)
        num_steps = round(self._request.duration_s / control_dt_s)
        pending_targets = tuple(simulation.default_dof_pos.expand(self._num_envs, -1).clone() for _ in self._controllers)
        for step_index in range(num_steps):
            states = simulation.read_go2_states()
            planars = simulation.read_robot_planars()
            new_targets = []
            for state, planar, controller in zip(states, planars, self._controllers, strict=True):
                fell |= (state.base_euler_deg[:, [ROLL_AXIS, PITCH_AXIS]].abs() > FALL_TILT_DEG).any(dim=1)
                new_targets.append(controller.compute_joint_targets(state.proprioception, self._commands(planar.yaw_rad)))
            # 歩行方策の学習時と同じく、1 制御周期前に決めた目標を適用する
            simulation.apply_joint_targets(pending_targets)
            pending_targets = tuple(new_targets)
            simulation.step()
            if bool(simulation.diverged_mask().any()):
                raise CapacityCheckError(f"質量 {mass_kg} kg で物理計算が発散した（ステップ {step_index}）")
            footprint = simulation.read_box_footprint()
            max_abs_box_yaw_deg = torch.maximum(max_abs_box_yaw_deg, torch.rad2deg(footprint.yaw_rad.abs()))
            newly_open = is_corridor_open(footprint, corridor) & (open_step < 0)
            open_step = torch.where(newly_open, step_index + 1, open_step)
        box = simulation.read_box_planar()
        displacement = box.position_xy_m - start_xy
        return [
            TrialRow(
                num_robots=self._num_robots,
                mass_kg=mass_kg,
                command_vx_mps=round(float(self._speeds[env]), 3),
                repeat=env % self._request.repeats,
                box_dx_m=round(float(displacement[env, X_AXIS]), 4),
                box_dy_m=round(float(displacement[env, Y_AXIS]), 4),
                box_final_yaw_deg=round(math.degrees(float(box.yaw_rad[env])), 2),
                box_max_abs_yaw_deg=round(float(max_abs_box_yaw_deg[env]), 2),
                corridor_opened=bool(open_step[env] >= 0),
                open_time_s=round(int(open_step[env]) * control_dt_s, 2) if open_step[env] >= 0 else None,
                robot_fell=bool(fell[env]),
            )
            for env in range(self._num_envs)
        ]

    def _commands(self, robot_yaw_rad: torch.Tensor) -> torch.Tensor:
        heading_error = wrap_angle_rad(-robot_yaw_rad)
        yaw_rate = torch.clamp(HEADING_GAIN_PER_S * heading_error, -MAX_YAW_RATE_RADPS, MAX_YAW_RATE_RADPS)
        return torch.stack((self._speeds, torch.zeros_like(self._speeds), yaw_rate), dim=1)

    def _reset(self) -> None:
        all_envs = torch.ones(self._num_envs, dtype=torch.bool, device=gs.device)
        noise_rad = self._gait_config.env.reset_joint_pos_noise_rad
        default_dof_pos = self._simulation.default_dof_pos
        dof_pos = tuple(
            sample_reset_dof_pos(default_dof_pos, self._num_envs, noise_rad, self._generator)
            for _ in self._controllers
        )
        self._simulation.reset(all_envs, dof_pos)
        for controller in self._controllers:
            controller.reset_where(all_envs)


def summarize(rows: list[TrialRow]) -> None:
    """質量と速さの組ごとに、箱の移動量・通路が開いた割合・開くまでの時間・転倒の割合を表示する。"""
    groups: dict[tuple[float, float], list[TrialRow]] = {}
    for row in rows:
        groups.setdefault((row.mass_kg, row.command_vx_mps), []).append(row)
    print("台数 質量[kg] 指令[m/s] 移動量 平均/最小/最大[m] 最大ヨー[deg] 開通率 開通までの平均[s] 転倒率")
    for (mass_kg, speed), group in sorted(groups.items()):
        dx = [row.box_dx_m for row in group]
        yaw = [row.box_max_abs_yaw_deg for row in group]
        open_times = [row.open_time_s for row in group if row.open_time_s is not None]
        mean_open_time = f"{sum(open_times) / len(open_times):6.2f}" if open_times else "     -"
        fall_rate = sum(row.robot_fell for row in group) / len(group)
        print(
            f"{group[0].num_robots} {mass_kg:>5} {speed:>5} "
            f"{sum(dx) / len(dx):6.3f}/{min(dx):6.3f}/{max(dx):6.3f} {sum(yaw) / len(yaw):6.1f} "
            f"{len(open_times) / len(group):5.2f} {mean_open_time} {fall_rate:5.2f}"
        )


@dataclass(frozen=True)
class CommandLine:
    """コマンドラインで受け取った計測の条件・シーン・出力先・デバイス。"""

    request: SweepRequest
    scene_config: Path
    output_csv: Path
    device: str


def parse_args() -> CommandLine:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scene-config", type=Path, required=True)
    parser.add_argument("--masses-kg", type=float, nargs="+", default=[2.0, 4.0, 5.0, 6.0])
    parser.add_argument("--speeds-mps", type=float, nargs="+", default=[0.6])
    parser.add_argument("--repeats", type=int, default=16)
    # 2 台で 5 kg を押した平面での速さ（10 秒で約 1.6 m）でも、箱が通行領域を出る（約 1.5 m）までに収まる長さ
    parser.add_argument("--duration-s", type=float, default=20.0)
    parser.add_argument("--output-csv", type=Path, required=True)
    # 学習・評価と同じ cuda を基本にする。cpu はコードの動作を少数の環境で確かめるために使う
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    args = parser.parse_args()
    request = SweepRequest(
        masses_kg=tuple(args.masses_kg),
        speeds_mps=tuple(args.speeds_mps),
        repeats=args.repeats,
        duration_s=args.duration_s,
    )
    return CommandLine(
        request=request, scene_config=args.scene_config, output_csv=args.output_csv, device=args.device
    )


def main() -> None:
    command_line = parse_args()
    output_csv = command_line.output_csv
    if output_csv.exists():
        raise FileExistsError(f"出力先が既にある（上書きしない）: {output_csv}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    settings = load_corridor_scene_settings(command_line.scene_config)
    GENESIS_INITIALIZERS[command_line.device](SEED)
    gait_config = load_gait_settings(GAIT_RUN_DIR / RUN_CONFIG_FILENAME).gait
    sweep = CorridorCapacitySweep(gait_config, settings, command_line.request)
    rows: list[TrialRow] = []
    for mass_kg in command_line.request.masses_kg:
        rows.extend(sweep.run(mass_kg))
        print(f"質量 {mass_kg} kg を計測した")
    with output_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in dataclasses.fields(TrialRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    summarize(rows)


if __name__ == "__main__":
    main()
