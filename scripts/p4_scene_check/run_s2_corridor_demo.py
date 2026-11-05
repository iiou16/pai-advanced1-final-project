"""シーン S2_corridor で、1 台の Go2 が通路を開通させ、そのあと通路を通過できることを確かめる。

規則ベースの制御で次の 3 段を順に行い、通路の開通と Go2 の通過を別々に判定する。

1. 押す: 退避領域の中心へ向かって前進し、間にある箱を押し出す。通路が開いたら次へ進む。
2. 戻る: 通行領域の入口側の端より手前（入口側）まで戻る。通路が開いた状態で入口側に着いたら次へ進む。
3. 通る: 通行領域の出口側の端の先へ向かって通路を通り抜ける。通過と判定されたら止まる。

押す段で箱の横へ回り込んで位置を合わせ直さないのは、通路では回り込む先が壁になり、出口の角で止まってしまうため。
押していたロボットは通路の中にいるまま通路が開くため、通過として数えるにはいったん入口側へ戻る必要がある
（「通路が開いたあとに入口から出口まで通った」を通過とする判定。cooperative_go2.entities.corridor）。
転倒したロボットはその場で止め、失敗として数える。

実行（Genesis を cuda で使う前に、同じシェルで source scripts/setup/env.sh する）:
    uv run python scripts/p4_scene_check/run_s2_corridor_demo.py --device cuda --output-dir outputs/p4_scene_check/s2-demo
"""

import argparse
import csv
import dataclasses
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path

import genesis as gs
import torch

from cooperative_go2.entities.corridor import PassageStage, advance_passage_stage, is_corridor_open
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.rule_based_push import compute_waypoint_commands
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import (
    CorridorSceneSettings,
    load_corridor_scene_settings,
    load_gait_settings,
    load_push_training_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest, FixedCameraSpec
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder

SCENE_CONFIG_PATH = Path("configs/scenes/s2_corridor.yaml")
# 旋回のゲイン・上限と、戻る・通るときの速さは、E1 の規則ベースの制御（rule_based 節）と同じにする
RULE_BASED_CONFIG_PATH = Path("configs/push_single_wide.yaml")
GAIT_RUN_DIR = Path("outputs/gait/gait-command-tracking")
GAIT_CHECKPOINT = 1500
SEED = 4101
# 押すときの前進の指令。重い箱を押す指令と同じ、高レベル方策の指令の上限
PUSH_SPEED_MPS = 0.6
# 高レベルの指令は E1 と同じ 10 Hz で決め直し、その間は同じ指令で歩かせる
HIGH_LEVEL_PERIOD_S = 0.1
# 戻る・通るときの目標点を、通行領域の端からさらに離す距離 [m]。端の上で止まらず、判定の線を確実に越えるため
WAYPOINT_MARGIN_M = 0.3
# 歩行方策の学習時の転倒の判定と同じ傾き
FALL_TILT_DEG = 10.0
ROLL_AXIS, PITCH_AXIS = 0, 1
VIDEO_FILENAME = "s2_corridor_env0.mp4"
TRIALS_FILENAME = "trials.csv"
# 通路の全体（スタートから退避領域まで）を、壁で通路の中が隠れないよう真上に近い角度から 1 つの画角で撮る
VIDEO_CAMERA = FixedCameraSpec(
    resolution=(1280, 720), position_m=(1.8, -2.5, 6.5), lookat_m=(1.8, 0.0, 0.0), fov_deg=50.0
)
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}


class DemoError(RuntimeError):
    """確認を続けられない状態（物理計算の発散など）を表す。"""


class DemoStage(IntEnum):
    """確認の段階。"""

    PUSH = 0
    RETURN = 1
    PASS = 2
    DONE = 3
    FELL = 4


@dataclass(frozen=True)
class TrialRow:
    """1 試行（並列環境 1 つ）の結果。CSV の 1 行になる。

    Attributes:
        env: 並列環境の番号。
        corridor_opened: 試行中に通路が開いたか。
        open_time_s: 通路が初めて開いた時刻 [s]。開かなければ None（CSV では空欄）。
        passed: Go2 が通過したか。
        pass_time_s: 通過と判定された時刻 [s]。通過しなければ None。
        fell: 転倒したか。
        final_stage: 試行の終わりの段階（DemoStage の名前）。
    """

    env: int
    corridor_opened: bool
    open_time_s: float | None
    passed: bool
    pass_time_s: float | None
    fell: bool
    final_stage: str


class CorridorDemo:
    """1 台の Go2 を、押す → 戻る → 通るの順に動かし、環境ごとに開通と通過を判定する。"""

    def __init__(self, gait_config: GaitConfig, settings: CorridorSceneSettings, num_envs: int, video: bool) -> None:
        self._gait_config = gait_config
        self._layout = settings.scene
        self._num_envs = num_envs
        camera = VIDEO_CAMERA if video else None
        self._simulation = GenesisCorridorSimulation.build(
            gait_config, settings, CorridorSceneRequest(num_envs=num_envs, camera=camera)
        )
        if len(self._simulation.robot_names) != 1:
            raise DemoError(f"このシーンは 1 台で確かめる: {self._simulation.robot_names}")
        self._controller = load_gait_controller(GAIT_RUN_DIR, GAIT_CHECKPOINT, num_envs, gs.device)
        self._rule_config = load_push_training_settings(RULE_BASED_CONFIG_PATH).rule_based
        region = self._layout.corridor.passage_region()
        center_y = self._layout.corridor.center_y_m
        self._entrance_waypoint = torch.tensor([region.min_x_m - WAYPOINT_MARGIN_M, center_y], device=gs.device)
        self._exit_waypoint = torch.tensor([region.max_x_m + WAYPOINT_MARGIN_M, center_y], device=gs.device)
        self._retreat_center = torch.tensor(self._layout.retreat_area.center_xy_m, device=gs.device)
        self._generator = make_seeded_generator(SEED)

    @property
    def simulation(self) -> GenesisCorridorSimulation:
        """構築済みのシミュレーション（評価動画のカメラを取り出すのに使う）。"""
        return self._simulation

    def run(self, duration_s: float) -> list[TrialRow]:
        """全環境で 1 試行ずつ、duration_s [s] まで（全環境が終わればそこまで）動かし、環境ごとの結果を返す。"""
        simulation = self._simulation
        all_envs = torch.ones(self._num_envs, dtype=torch.bool, device=gs.device)
        dof_pos = sample_reset_dof_pos(
            simulation.default_dof_pos, self._num_envs, self._gait_config.env.reset_joint_pos_noise_rad, self._generator
        )
        simulation.reset(all_envs, (dof_pos,))
        self._controller.reset_where(all_envs)
        control_dt_s = self._gait_config.env.control_dt_s
        gait_steps_per_command = round(HIGH_LEVEL_PERIOD_S / control_dt_s)
        stages = torch.full((self._num_envs,), int(DemoStage.PUSH), dtype=torch.int64, device=gs.device)
        passage = torch.full((self._num_envs,), int(PassageStage.WAITING), dtype=torch.int64, device=gs.device)
        open_step = torch.full((self._num_envs,), -1, dtype=torch.int64, device=gs.device)
        pass_step = torch.full((self._num_envs,), -1, dtype=torch.int64, device=gs.device)
        pending_targets = simulation.default_dof_pos.expand(self._num_envs, -1).clone()
        commands = torch.zeros(self._num_envs, 3, device=gs.device)
        corridor = self._layout.corridor
        for step_index in range(round(duration_s / control_dt_s)):
            (state,) = simulation.read_go2_states()
            fell = (state.base_euler_deg[:, [ROLL_AXIS, PITCH_AXIS]].abs() > FALL_TILT_DEG).any(dim=1)
            stages = torch.where(fell & (stages != DemoStage.DONE), int(DemoStage.FELL), stages)
            if step_index % gait_steps_per_command == 0:
                commands = self._commands(stages)
            targets = self._controller.compute_joint_targets(state.proprioception, commands)
            # 歩行方策の学習時と同じく、1 制御周期前に決めた目標を適用する
            simulation.apply_joint_targets((pending_targets,))
            pending_targets = targets
            simulation.step()
            if bool(simulation.diverged_mask().any()):
                raise DemoError(f"物理計算が発散した（ステップ {step_index}）")
            corridor_open = is_corridor_open(simulation.read_box_footprint(), corridor)
            open_step = torch.where(corridor_open & (open_step < 0), step_index + 1, open_step)
            (robot,) = simulation.read_robot_planars()
            passage = advance_passage_stage(passage, robot.position_xy_m, corridor_open, corridor)
            pass_step = torch.where((passage == PassageStage.PASSED) & (pass_step < 0), step_index + 1, pass_step)
            stages = self._advance_stages(stages, corridor_open, passage)
            if bool(((stages == DemoStage.DONE) | (stages == DemoStage.FELL)).all()):
                break
        return [self._row(env, open_step, pass_step, stages, control_dt_s) for env in range(self._num_envs)]

    def _commands(self, stages: torch.Tensor) -> torch.Tensor:
        (robot,) = self._simulation.read_robot_planars()
        rule = self._rule_config
        push = compute_waypoint_commands(
            robot.position_xy_m, robot.yaw_rad, self._retreat_center.expand(self._num_envs, -1), PUSH_SPEED_MPS, rule
        )
        walk_back = compute_waypoint_commands(
            robot.position_xy_m, robot.yaw_rad, self._entrance_waypoint.expand(self._num_envs, -1), rule.approach_speed_mps, rule
        )
        walk_through = compute_waypoint_commands(
            robot.position_xy_m, robot.yaw_rad, self._exit_waypoint.expand(self._num_envs, -1), rule.approach_speed_mps, rule
        )
        by_stage = torch.zeros_like(push)
        for stage, stage_commands in (
            (DemoStage.PUSH, push),
            (DemoStage.RETURN, walk_back),
            (DemoStage.PASS, walk_through),
        ):
            by_stage = torch.where((stages == stage).unsqueeze(1), stage_commands, by_stage)
        return by_stage

    @staticmethod
    def _advance_stages(stages: torch.Tensor, corridor_open: torch.Tensor, passage: torch.Tensor) -> torch.Tensor:
        advanced = torch.where((stages == DemoStage.PUSH) & corridor_open, int(DemoStage.RETURN), stages)
        advanced = torch.where(
            (advanced == DemoStage.RETURN) & (passage == PassageStage.ELIGIBLE), int(DemoStage.PASS), advanced
        )
        return torch.where(
            (advanced == DemoStage.PASS) & (passage == PassageStage.PASSED), int(DemoStage.DONE), advanced
        )

    @staticmethod
    def _row(
        env: int, open_step: torch.Tensor, pass_step: torch.Tensor, stages: torch.Tensor, control_dt_s: float
    ) -> TrialRow:
        def time_or_none(step: torch.Tensor) -> float | None:
            return round(int(step) * control_dt_s, 2) if int(step) >= 0 else None

        stage = DemoStage(int(stages[env]))
        return TrialRow(
            env=env,
            corridor_opened=int(open_step[env]) >= 0,
            open_time_s=time_or_none(open_step[env]),
            passed=int(pass_step[env]) >= 0,
            pass_time_s=time_or_none(pass_step[env]),
            fell=stage is DemoStage.FELL,
            final_stage=stage.name,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--duration-s", type=float, default=60.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true", help="環境 0 の動画を撮らない")
    # 学習・評価と同じ cuda を基本にする。cpu はコードの動作を少数の環境で確かめるために使う
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"出力先が既にある（上書きしない）: {output_dir}")
    settings = load_corridor_scene_settings(SCENE_CONFIG_PATH)
    GENESIS_INITIALIZERS[args.device](SEED)
    gait_config = load_gait_settings(GAIT_RUN_DIR / RUN_CONFIG_FILENAME).gait
    demo = CorridorDemo(gait_config, settings, args.num_envs, video=not args.no_video)
    output_dir.mkdir(parents=True)
    camera = demo.simulation.scene.camera
    if camera is None:
        rows = demo.run(args.duration_s)
    else:
        recorder = GenesisVideoRecorder(camera, output_dir / VIDEO_FILENAME, gait_config.env.control_dt_s)
        recorder.start()
        rows = demo.run(args.duration_s)
        recorder.stop()
    with (output_dir / TRIALS_FILENAME).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in dataclasses.fields(TrialRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    num_opened = sum(row.corridor_opened for row in rows)
    num_passed = sum(row.passed for row in rows)
    num_fell = sum(row.fell for row in rows)
    print(f"{len(rows)} 試行: 開通 {num_opened}、通過 {num_passed}、転倒 {num_fell}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
