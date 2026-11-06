"""規則ベースで重い箱を押す制御（1 台・通信なしの 2 台・協調の 2 台）を、ROS 2 を通さずに並列環境で動かして確かめる。

ROS 2 の実行系に移す前に、押す位置の割り当て・同時開始・回転を抑える直進の制御の出来と、ゲインの効き方を
多数の試行で安く調べるため。判断（TeamPushAgent）は ROS 2 の協調のノードと同じものを使い、
相手の状態は 1 回前の判断の結果を渡す（通信の遅れ 1 回分）。開始位置は設定の reset に従ってシードで散らす。

試行は、全ロボットが DONE になったら成功、いずれかが FAILED になったら失敗として終える。
通路の開通（箱が通行領域から出たか）は状態機械と別に、シミュレーションの真値で毎周期判定する。

実行（Genesis を cuda で使う前に、同じシェルで source scripts/setup/env.sh する）:
    uv run python scripts/p4_coordination_check/run_team_push.py --device cuda --config configs/team_rule_based.yaml \\
        --mode cooperative --output-dir outputs/p4_coordination_check/e5-check
"""

import argparse
import csv
import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path

import genesis as gs
import torch

from cooperative_go2.entities.coordination_state_machine import TERMINAL_PHASES, TeamPhase
from cooperative_go2.entities.corridor import is_corridor_open
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.push_trial import PlanarPose
from cooperative_go2.entities.team_readiness import PartnerStatus
from cooperative_go2.entities.team_task_config import sample_team_start_positions
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import TeamRunSettings, load_gait_settings, load_team_run_settings
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest, FixedCameraSpec
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder
from cooperative_go2.use_cases.team_push_agent import (
    CoordinationMode,
    RetreatArea,
    TeamPushAgent,
    TeamPushAgentInputs,
)

ROLL_AXIS, PITCH_AXIS = 0, 1
TRIALS_FILENAME = "trials.csv"
# 判断ごとの段階と位置の記録（--trace のとき）。失敗した試行の経過を調べるため
TRACE_FILENAME = "trace.csv"
TRACE_COLUMNS = ("time_s", "env", "robot", "phase", "robot_x", "robot_y", "robot_yaw", "box_x", "box_y", "box_yaw")
VIDEO_FILENAME = "env0.mp4"
# 通路の全体を、壁で通路の中が隠れないよう真上に近い角度から撮る
VIDEO_CAMERA = FixedCameraSpec(
    resolution=(1280, 720), position_m=(1.8, -2.5, 6.5), lookat_m=(1.8, 0.0, 0.0), fov_deg=50.0
)
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}


class TeamPushCheckError(RuntimeError):
    """確認を続けられない状態（物理計算の発散など）を表す。"""


@dataclass(frozen=True)
class TrialRow:
    """1 試行（並列環境 1 つ）の結果。CSV の 1 行になる。

    Attributes:
        env: 並列環境の番号。
        success: 全ロボットが DONE になった（箱が退避領域に入ったまま続いた）。
        failure_reasons: FAILED になったロボットの理由（名前を | でつなぐ）。無ければ空。
        corridor_opened: 試行中に通路が開いた。
        open_time_s: 通路が初めて開いた時刻 [s]。開かなければ None（CSV では空欄）。
        end_time_s: 試行を終えた時刻 [s]（全員が DONE、いずれかが FAILED、または時間切れ）。
        push_start_times_s: ロボットごとの、初めて PUSH に入った時刻 [s]（| でつなぐ。入らなければ -）。
        push_start_spread_s: 押し始めた時刻の最大と最小の差 [s]。全員が押し始めなければ None。
        box_dx_m: 箱の +x 方向の移動量 [m]。
        box_max_abs_yaw_deg: 試行中の、押す方向から見た箱の回転の絶対値の最大 [deg]。
        falls: 転倒したロボットの数。
    """

    env: int
    success: bool
    failure_reasons: str
    corridor_opened: bool
    open_time_s: float | None
    end_time_s: float
    push_start_times_s: str
    push_start_spread_s: float | None
    box_dx_m: float
    box_max_abs_yaw_deg: float
    falls: int


class TeamPushCheck:
    """全環境で 1 試行ずつ、規則ベースの制御で箱を押させ、環境ごとの結果を返す。"""

    def __init__(self, run: TeamRunSettings, mode: CoordinationMode, num_envs: int, video: bool) -> None:
        self.trace: list[tuple[object, ...]] = []
        self._run = run
        self._mode = mode
        self._num_envs = num_envs
        gait_policy = run.settings.gait_policy
        self._gait_config = load_gait_settings(Path(gait_policy.run_dir) / RUN_CONFIG_FILENAME).gait
        camera = VIDEO_CAMERA if video else None
        self._simulation = GenesisCorridorSimulation.build(
            self._gait_config, run.scene, CorridorSceneRequest(num_envs=num_envs, camera=camera)
        )
        self._num_robots = len(self._simulation.robot_names)
        self._controllers = tuple(
            load_gait_controller(Path(gait_policy.run_dir), gait_policy.checkpoint, num_envs, gs.device)
            for _ in range(self._num_robots)
        )
        team = run.settings.team
        self._agents = [
            [TeamPushAgent(team, robot_index, mode) for robot_index in range(self._num_robots)]
            for _ in range(num_envs)
        ]
        retreat = run.scene.scene.retreat_area
        self._retreat = RetreatArea(center_xy_m=tuple(retreat.center_xy_m), radius_m=retreat.radius_m)

    @property
    def simulation(self) -> GenesisCorridorSimulation:
        """構築済みのシミュレーション（評価動画のカメラを取り出すのに使う）。"""
        return self._simulation

    def run(self, generator: torch.Generator) -> list[TrialRow]:
        """全環境をリセットして 1 試行ずつ動かし、環境ごとの結果を返す。"""
        simulation = self._simulation
        team = self._run.settings.team
        all_envs = torch.ones(self._num_envs, dtype=torch.bool, device=gs.device)
        initial_xy = tuple(tuple(robot.initial_xy_m) for robot in self._run.scene.scene.robots)
        start_xy = sample_team_start_positions(initial_xy, team.reset, self._num_envs, generator)
        noise_rad = self._gait_config.env.reset_joint_pos_noise_rad
        dof_pos = tuple(
            sample_reset_dof_pos(simulation.default_dof_pos, self._num_envs, noise_rad, generator)
            for _ in range(self._num_robots)
        )
        simulation.reset_at(all_envs, dof_pos, start_xy)
        for controller in self._controllers:
            controller.reset_where(all_envs)
        for env_agents in self._agents:
            for agent in env_agents:
                agent.start(self._retreat, now_s=0.0)
        return self._run_episode()

    def _run_episode(self) -> list[TrialRow]:
        simulation = self._simulation
        team = self._run.settings.team
        control_dt_s = self._gait_config.env.control_dt_s
        gait_steps = team.gait_steps_per_high_level_step(control_dt_s)
        max_ticks = math.ceil(team.coordination.timeouts.task_s / team.high_level_period_s) + 1
        corridor = self._run.scene.scene.corridor
        box_start_x = simulation.read_box_planar().position_xy_m[:, 0].clone()
        num_envs, num_robots = self._num_envs, self._num_robots
        published: list[list[PartnerStatus | None]] = [[None] * num_robots for _ in range(num_envs)]
        fallen = torch.zeros(num_robots, num_envs, dtype=torch.bool, device=gs.device)
        finished = [False] * num_envs
        end_time = [0.0] * num_envs
        open_time: list[float | None] = [None] * num_envs
        push_start: list[list[float | None]] = [[None] * num_robots for _ in range(num_envs)]
        max_yaw = torch.zeros(num_envs, device=gs.device)
        pending = [simulation.default_dof_pos.expand(num_envs, -1).clone() for _ in range(num_robots)]
        for tick in range(max_ticks):
            now_s = round(tick * team.high_level_period_s, 6)
            commands = self._decide_all(now_s, published, fallen, finished)
            for env in range(num_envs):
                if finished[env]:
                    continue
                phases = [agent.state.phase for agent in self._agents[env]]
                for robot_index, phase in enumerate(phases):
                    if phase is TeamPhase.PUSH and push_start[env][robot_index] is None:
                        push_start[env][robot_index] = now_s
                if all(phase in TERMINAL_PHASES for phase in phases) or TeamPhase.FAILED in phases:
                    finished[env] = True
                    end_time[env] = now_s
            if all(finished):
                break
            # 判断の結果を次の周期に相手へ渡す（通信の遅れ 1 回分）
            published = [
                [PartnerStatus(phase=agent.state.phase, stamp_s=now_s) for agent in env_agents]
                for env_agents in self._agents
            ]
            for _ in range(gait_steps):
                states = simulation.read_go2_states()
                targets = []
                for robot_index, (state, controller) in enumerate(zip(states, self._controllers, strict=True)):
                    tilted = (state.base_euler_deg[:, [ROLL_AXIS, PITCH_AXIS]].abs() > team.fall_tilt_deg).any(dim=1)
                    fallen[robot_index] |= tilted
                    targets.append(controller.compute_joint_targets(state.proprioception, commands[robot_index]))
                # 歩行方策の学習時と同じく、1 制御周期前に決めた目標を適用する
                simulation.apply_joint_targets(tuple(pending))
                pending = targets
                simulation.step()
                if bool(simulation.diverged_mask().any()):
                    raise TeamPushCheckError(f"物理計算が発散した（{now_s} s）")
            footprint = simulation.read_box_footprint()
            push_angle = self._push_angle(footprint.center_xy_m)
            box_yaw_error = torch.remainder(footprint.yaw_rad - push_angle + math.pi / 2, math.pi) - math.pi / 2
            active = torch.tensor([not done for done in finished], device=gs.device)
            max_yaw = torch.where(active, torch.maximum(max_yaw, box_yaw_error.abs()), max_yaw)
            opened = is_corridor_open(footprint, corridor)
            for env in range(num_envs):
                if not finished[env] and open_time[env] is None and bool(opened[env]):
                    open_time[env] = round(now_s + team.high_level_period_s, 6)
        timeout_s = round(max_ticks * team.high_level_period_s, 6)
        box_dx = simulation.read_box_planar().position_xy_m[:, 0] - box_start_x
        return [
            self._row(env, finished[env], end_time[env] if finished[env] else timeout_s, open_time[env],
                      push_start[env], float(box_dx[env]), float(max_yaw[env]), int(fallen[:, env].sum()))
            for env in range(num_envs)
        ]  # fmt: skip

    def _decide_all(
        self, now_s: float, published: list[list[PartnerStatus | None]], fallen: torch.Tensor, finished: list[bool]
    ) -> list[torch.Tensor]:
        # GPU の値を 1 つずつ読むと読むたびに同期が起きるので、周期ごとに 1 回まとめて CPU に移してから使う
        robots_xy = [robot.position_xy_m.cpu() for robot in self._simulation.read_robot_planars()]
        robots_yaw = [robot.yaw_rad.cpu() for robot in self._simulation.read_robot_planars()]
        box = self._simulation.read_box_planar()
        box_xy, box_yaw = box.position_xy_m.cpu(), box.yaw_rad.cpu()
        fallen_cpu = fallen.cpu()
        commands = [torch.zeros(self._num_envs, 3) for _ in range(self._num_robots)]
        for env in range(self._num_envs):
            if finished[env]:
                continue
            box_pose = _pose(box_xy[env], box_yaw[env])
            for robot_index, agent in enumerate(self._agents[env]):
                # 2 台のときの相手は、もう一方のロボット。1 台なら相手はいない
                partner = published[env][1 - robot_index] if self._num_robots == 2 else None
                decision = agent.decide(
                    TeamPushAgentInputs(
                        now_s=now_s,
                        robot=_pose(robots_xy[robot_index][env], robots_yaw[robot_index][env]),
                        box=box_pose,
                        partner=partner,
                        task_assigned=True,
                        fallen=bool(fallen_cpu[robot_index, env]),
                    )
                )
                commands[robot_index][env] = decision.commands[0]
                robot_xy, robot_yaw = robots_xy[robot_index][env], robots_yaw[robot_index][env]
                self.trace.append((
                    now_s, env, robot_index, decision.state.phase.name, round(float(robot_xy[0]), 3),
                    round(float(robot_xy[1]), 3), round(float(robot_yaw), 3), round(box_pose.x_m, 3),
                    round(box_pose.y_m, 3), round(box_pose.yaw_rad, 3),
                ))  # fmt: skip
        return [robot_commands.to(gs.device) for robot_commands in commands]

    def _push_angle(self, box_xy_m: torch.Tensor) -> torch.Tensor:
        retreat = torch.tensor(self._retreat.center_xy_m, device=box_xy_m.device)
        to_retreat = retreat - box_xy_m
        return torch.atan2(to_retreat[:, 1], to_retreat[:, 0])

    def _row(
        self, env: int, finished: bool, end_time_s: float, open_time_s: float | None,
        push_start: list[float | None], box_dx_m: float, max_yaw_rad: float, falls: int,
    ) -> TrialRow:  # fmt: skip
        states = [agent.state for agent in self._agents[env]]
        reasons = [state.failure_reason.name for state in states if state.failure_reason is not None]
        starts = [time for time in push_start if time is not None]
        return TrialRow(
            env=env,
            success=finished and all(state.phase is TeamPhase.DONE for state in states),
            failure_reasons="|".join(reasons),
            corridor_opened=open_time_s is not None,
            open_time_s=open_time_s,
            end_time_s=end_time_s,
            push_start_times_s="|".join("-" if time is None else f"{time:.1f}" for time in push_start),
            push_start_spread_s=round(max(starts) - min(starts), 2) if len(starts) == len(push_start) else None,
            box_dx_m=round(box_dx_m, 3),
            box_max_abs_yaw_deg=round(math.degrees(max_yaw_rad), 1),
            falls=falls,
        )


def _pose(xy_m: torch.Tensor, yaw_rad: torch.Tensor) -> PlanarPose:
    return PlanarPose(x_m=float(xy_m[0]), y_m=float(xy_m[1]), yaw_rad=float(yaw_rad))


def summarize(rows: list[TrialRow]) -> None:
    """成功率・開通率・時間・押し始めの時刻の差・箱の回転・転倒をまとめて表示する。"""
    num = len(rows)
    successes = [row for row in rows if row.success]
    opened = [row for row in rows if row.open_time_s is not None]
    spreads = [row.push_start_spread_s for row in rows if row.push_start_spread_s is not None]
    yaws = [row.box_max_abs_yaw_deg for row in rows]
    reasons: dict[str, int] = {}
    for row in rows:
        for reason in filter(None, row.failure_reasons.split("|")):
            reasons[reason] = reasons.get(reason, 0) + 1
    print(f"成功 {len(successes)}/{num}、開通 {len(opened)}/{num}、転倒したロボット {sum(row.falls for row in rows)}")
    if successes:
        print(f"成功までの平均 {sum(row.end_time_s for row in successes) / len(successes):.1f} s")
    if opened:
        open_times = [row.open_time_s for row in opened if row.open_time_s is not None]
        print(f"開通までの平均 {sum(open_times) / len(open_times):.1f} s")
    if spreads:
        print(f"押し始めの時刻の差: 平均 {sum(spreads) / len(spreads):.2f} s、最大 {max(spreads):.2f} s")
    print(f"箱の回転の最大: 平均 {sum(yaws) / num:.1f}°、全体 {max(yaws):.1f}°")
    print(f"移動量の平均 {sum(row.box_dx_m for row in rows) / num:.2f} m、失敗の理由 {reasons}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="作業の設定（configs/team_rule_based*.yaml）")
    parser.add_argument("--mode", choices=[mode.value for mode in CoordinationMode], required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=5001)
    parser.add_argument(
        "--push-control", nargs="*", default=[], metavar="NAME=VALUE",
        help="設定の push_control の値を置き換える（調整用。例: lookahead_m=0.25 yaw_balance_gain_per_rad=1.0）",
    )  # fmt: skip
    parser.add_argument("--video", action="store_true", help="環境 0 の動画を撮る")
    parser.add_argument("--trace", action="store_true", help=f"判断ごとの段階と位置を {TRACE_FILENAME} に書く")
    parser.add_argument("--output-dir", type=Path, required=True)
    # 学習・評価と同じ cuda を基本にする。cpu はコードの動作を少数の環境で確かめるために使う
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    if output_dir.exists():
        raise FileExistsError(f"出力先が既にある（上書きしない）: {output_dir}")
    run = load_team_run_settings(args.config)
    if args.push_control:
        overrides = {name: float(value) for name, value in (item.split("=", 1) for item in args.push_control)}
        push_control = dataclasses.replace(run.settings.team.push_control, **overrides)
        team = dataclasses.replace(run.settings.team, push_control=push_control)
        run = dataclasses.replace(run, settings=dataclasses.replace(run.settings, team=team))
    GENESIS_INITIALIZERS[args.device](args.seed)
    check = TeamPushCheck(run, CoordinationMode(args.mode), args.num_envs, video=args.video)
    output_dir.mkdir(parents=True)
    generator = make_seeded_generator(args.seed)
    camera = check.simulation.scene.camera
    if camera is None:
        rows = check.run(generator)
    else:
        gait_run_dir = Path(run.settings.gait_policy.run_dir)
        control_dt_s = load_gait_settings(gait_run_dir / RUN_CONFIG_FILENAME).gait.env.control_dt_s
        recorder = GenesisVideoRecorder(camera, output_dir / VIDEO_FILENAME, control_dt_s)
        recorder.start()
        rows = check.run(generator)
        recorder.stop()
    with (output_dir / TRIALS_FILENAME).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in dataclasses.fields(TrialRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    if args.trace:
        with (output_dir / TRACE_FILENAME).open("w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(TRACE_COLUMNS)
            writer.writerows(check.trace)
    summarize(rows)


if __name__ == "__main__":
    main()
