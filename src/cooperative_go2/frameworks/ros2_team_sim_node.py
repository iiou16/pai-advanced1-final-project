"""ROS 2 のシミュレーションノード（通路の開通・1 台と 2 台）。Genesis のシーンを進める唯一のプロセス。

/clock・各ロボットの関節と胴体・箱・TF を配信し、各ロボットの歩行方策のノードの関節目標を受けて PD 制御する。
試行ごとに各ロボットの開始位置と関節角を引き、全ロボットの歩行方策のノードをリセットし、タスク管理に通路の開通を依頼する。
通路の開通・箱が退避領域に入ったまま続いたか・転倒はシミュレーションの真値で判定し、タスク管理の結果と並べて記録する。

起動例（launch から起動する）:
    go2-ros2-team-sim --config configs/ros2_team.yaml --mode cooperative --seed 6001 --num-trials 16 --device cuda \\
        --run-name e5-seed6001 --video false --ros-args -r __node:=genesis_sim -p use_sim_time:=true

結果は outputs/ros2_team_eval/<run-name>/ に trials.csv・summary.json・ros2_metrics.json、--video true なら video.mp4 を
保存する。保存先が既にあれば止まる。
"""

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import rclpy
import torch
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from cooperative_go2.entities.command_expiry import seconds_to_ns
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.team_task_config import InvalidTeamTaskConfigError, sample_team_start_positions
from cooperative_go2.frameworks.evaluate_push import DEVICES
from cooperative_go2.frameworks.genesis_runtime import make_seeded_generator
from cooperative_go2.frameworks.git_revision import GitCommitUnavailableError, current_commit_sha
from cooperative_go2.frameworks.ros2_executor_thread import (
    ExecutorThread,
    ExecutorThreadFailedError,
    HealthCheckedPacer,
)
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    Ros2TeamSettings,
    TeamRunSettings,
    load_gait_settings,
    load_ros2_team_settings,
    load_team_run_settings,
)
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest, FixedCameraSpec
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.genesis_team_world import GenesisTeamWorld
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder, VideoRecordingError
from cooperative_go2.interface_adapters.real_time_pacer import RealTimePacer
from cooperative_go2.interface_adapters.ros2_metrics import RoundTripLatencyProbe, write_node_metrics
from cooperative_go2.interface_adapters.ros2_sim_ports import PeerNodeRejectedError, PeerNodeTimeoutError
from cooperative_go2.interface_adapters.ros2_team_deciders import LEARNED_MODE
from cooperative_go2.interface_adapters.ros2_team_sim_ports import (
    RosTeamStatePublisher,
    RosTeamTaskPeers,
    subscribe_team_joint_targets,
)
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, JointTargetTiming
from cooperative_go2.use_cases.ros_team_trial import (
    NoTaskResultError,
    NoTeamJointTargetsAppliedError,
    TeamTrialPorts,
    TeamTrialRecord,
    TeamTrialRunner,
    TeamTrialSetup,
)
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError
from cooperative_go2.use_cases.team_push_agent import CoordinationMode

NODE_NAME = "genesis_sim"
OUTPUT_ROOT = Path("outputs/ros2_team_eval")
TRIALS_FILENAME = "trials.csv"
SUMMARY_FILENAME = "summary.json"
METRICS_FILENAME = "ros2_metrics.json"
VIDEO_FILENAME = "video.mp4"
NODE_METRICS_DIRNAME = "nodes"
# 通路の全体を、壁で通路の中が隠れないよう真上に近い角度から撮る
VIDEO_CAMERA = FixedCameraSpec(
    resolution=(1280, 720), position_m=(1.8, -2.5, 6.5), lookat_m=(1.8, 0.0, 0.0), fov_deg=50.0
)
# タスク管理の時間切れのあと、結果が届くのを待つシミュレーション時間 [s]。結果が届かないときの最後の歯止め
TASK_RESULT_GRACE_S = 5.0
# --trial-indices で全試行を表す値
ALL_TRIALS = "all"
# launch の引数をそのまま渡せるよう、真偽を文字列で受け取る
VIDEO_ON = "true"
VIDEO_OFF = "false"
EXIT_INVALID_INPUT = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_PEER_NODE_FAILED = 5
EXIT_GIT_UNAVAILABLE = 6
EXIT_VIDEO_RECORDING_FAILED = 7
EXIT_ROS_EXECUTOR_FAILED = 8

logger = logging.getLogger(__name__)


class TeamOutputExistsError(FileExistsError):
    """評価の保存先が既にあることを表す（上書きしない）。"""


@dataclass(frozen=True)
class _Inputs:
    settings: Ros2TeamSettings
    run: TeamRunSettings
    gait: GaitConfig
    commit_sha: str


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 のシミュレーションノード（通路の開通・1 台と 2 台）")
    parser.add_argument("--config", type=Path, required=True, help="ROS 2 の実行系の設定（configs/ros2_team*.yaml）")
    parser.add_argument(
        "--mode", choices=[*(mode.value for mode in CoordinationMode), LEARNED_MODE], required=True,
        help="協調のノードのモード（記録の条件として残す。協調のノードにも同じ値を渡す）",
    )  # fmt: skip
    parser.add_argument("--seed", type=int, required=True, help="評価用の乱数シード（開始位置と初期姿勢）")
    parser.add_argument("--num-trials", type=int, required=True, help="開始位置と初期姿勢を引く試行の数")
    parser.add_argument(
        "--trial-indices", required=True,
        help=f"実際に行う試行の番号（カンマ区切り）。{ALL_TRIALS} なら全試行。開始位置は --num-trials 試行分を引いてから選ぶので、"
        "同じシード・試行の数の評価の同じ番号の試行と同じ初期条件になる（評価動画を一部の試行だけ撮るときに使う）",
    )  # fmt: skip
    parser.add_argument("--device", choices=sorted(DEVICES), required=True, help="Genesis を動かすデバイス")
    parser.add_argument("--run-name", required=True, help=f"実行の名前。{OUTPUT_ROOT}/<run-name> に保存する")
    parser.add_argument("--video", choices=[VIDEO_ON, VIDEO_OFF], required=True, help="評価動画を撮るか（全試行を 1 本に）")
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error(f"--num-trials は 1 以上が必要: {args.num_trials}")
    if args.trial_indices == ALL_TRIALS:
        args.trial_indices = list(range(args.num_trials))
    else:
        indices = [int(index) for index in args.trial_indices.split(",")]
        if not indices or any(not 0 <= index < args.num_trials for index in indices):
            parser.error(f"--trial-indices は 0 以上 {args.num_trials} 未満の番号が必要: {args.trial_indices}")
        args.trial_indices = indices
    return args


def _load_inputs(config_path: Path) -> _Inputs:
    settings = load_ros2_team_settings(config_path)
    run = load_team_run_settings(Path(settings.team_config))
    gait = load_gait_settings(Path(run.settings.gait_policy.run_dir) / RUN_CONFIG_FILENAME).gait
    # 周期が歩行方策の制御周期の整数倍でなければここで止める
    run.settings.team.gait_steps_per_high_level_step(gait.env.control_dt_s)
    return _Inputs(settings, run, gait, current_commit_sha())


def _trial_setup(inputs: _Inputs, robot_names: tuple[str, ...]) -> TeamTrialSetup:
    scene = inputs.run.scene.scene
    team = inputs.run.settings.team
    size_x, size_y, _ = scene.box.size_m
    retreat_x, retreat_y = scene.retreat_area.center_xy_m
    control_dt_s = inputs.gait.env.control_dt_s
    return TeamTrialSetup(
        robot_names=robot_names,
        corridor=scene.corridor,
        box_half_size_xy_m=(size_x / 2, size_y / 2),
        retreat_xy_m=(retreat_x, retreat_y),
        retreat_radius_m=scene.retreat_area.radius_m,
        verify_hold_s=team.coordination.verify_hold_s,
        fall_tilt_deg=team.fall_tilt_deg,
        control_dt_ns=seconds_to_ns(control_dt_s),
        judge_steps=team.gait_steps_per_high_level_step(control_dt_s),
        max_trial_ns=seconds_to_ns(team.task_manager.timeout_s + TASK_RESULT_GRACE_S),
    )


def _run_trials(
    args: argparse.Namespace, inputs: _Inputs, node: Node, executor_thread: ExecutorThread, output_dir: Path
) -> tuple[list[TeamTrialRecord], RoundTripLatencyProbe]:
    gait_env = inputs.gait.env
    camera = VIDEO_CAMERA if args.video == VIDEO_ON else None
    simulation = GenesisCorridorSimulation.build(inputs.gait, inputs.run.scene, CorridorSceneRequest(1, camera))
    world = GenesisTeamWorld(simulation)
    robot_names = simulation.robot_names
    control_dt_ns = seconds_to_ns(gait_env.control_dt_s)
    timing = JointTargetTiming(
        control_dt_ns=control_dt_ns,
        latency_steps=1 if gait_env.simulate_action_latency else 0,
        expiry_ns=seconds_to_ns(inputs.settings.command_expiry.joint_target_s),
    )
    schedulers = tuple(JointTargetScheduler(timing, world.default_dof_pos) for _ in robot_names)
    latency_probe = RoundTripLatencyProbe()
    subscribe_team_joint_targets(node, robot_names, schedulers, latency_probe)
    runtime = inputs.settings.runtime
    peers = RosTeamTaskPeers(node, robot_names, runtime.reply_timeout_s)
    peers.wait_for_peers(runtime.startup_timeout_s)
    pacer = HealthCheckedPacer(RealTimePacer(gait_env.control_dt_s, runtime.max_real_time_factor), executor_thread)
    ports = TeamTrialPorts(
        world=world,
        publisher=RosTeamStatePublisher(node, robot_names, latency_probe),
        peers=peers,
        pacer=pacer,
        wall_clock=time.perf_counter,
    )
    runner = TeamTrialRunner(ports, schedulers, _trial_setup(inputs, robot_names))
    # 全試行の開始位置とリセット時の関節角を、並列環境での確認（run_team_push.py）と同じ順でまとめて引く
    generator = make_seeded_generator(args.seed)
    initial_xy = tuple(tuple(robot.initial_xy_m) for robot in inputs.run.scene.scene.robots)
    start_xy = sample_team_start_positions(initial_xy, inputs.run.settings.team.reset, args.num_trials, generator)
    noise_rad = gait_env.reset_joint_pos_noise_rad
    dof_pos = tuple(
        sample_reset_dof_pos(simulation.default_dof_pos, args.num_trials, noise_rad, generator) for _ in robot_names
    )
    recorder = None
    if camera is not None and simulation.scene.camera is not None:
        recorder = GenesisVideoRecorder(simulation.scene.camera, output_dir / VIDEO_FILENAME, gait_env.control_dt_s)
        recorder.start()
    records = []
    for trial_index in args.trial_indices:
        rows = slice(trial_index, trial_index + 1)
        record = runner.run_trial(
            trial_index, tuple(pos[rows].cpu() for pos in dof_pos), tuple(xy[rows].cpu() for xy in start_xy)
        )
        logger.info(
            "試行 %d: 真値 %s・タスク管理 %s（%s）、開通 %s s、%.1f s（実時間 %.1f s）", trial_index,
            "成功" if record.success else "失敗", "成功" if record.task_success else "失敗", record.task_reason,
            record.open_time_s, record.end_time_s, record.wall_time_s,
        )  # fmt: skip
        records.append(record)
    if recorder is not None:
        recorder.stop()
    return records, latency_probe


def _trial_row(record: TeamTrialRecord) -> dict[str, object]:
    starts = record.push_start_times_s
    started = [time_s for time_s in starts if time_s is not None]
    spread = round(max(started) - min(started), 3) if started and len(started) == len(starts) else None
    return {
        "trial": record.trial_index,
        "success": record.success,
        "task_success": record.task_success,
        "task_reason": record.task_reason,
        "corridor_opened": record.corridor_opened,
        "open_time_s": record.open_time_s,
        "success_time_s": record.success_time_s,
        "end_time_s": record.end_time_s,
        "push_start_times_s": "|".join("-" if time_s is None else f"{time_s:.2f}" for time_s in starts),
        "push_start_spread_s": spread,
        "box_dx_m": record.box_dx_m,
        "box_max_abs_yaw_deg": record.box_max_abs_yaw_deg,
        "falls": record.falls,
        "diverged": record.diverged,
        "wall_time_s": round(record.wall_time_s, 2),
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _summary(args: argparse.Namespace, inputs: _Inputs, rows: list[dict[str, object]]) -> dict[str, object]:
    num = len(rows)
    scene = inputs.run.scene.scene
    spreads = [float(row["push_start_spread_s"]) for row in rows if row["push_start_spread_s"] is not None]
    yaws = [float(row["box_max_abs_yaw_deg"]) for row in rows]
    return {
        "conditions": {
            "run_id": args.run_name,
            "commit_sha": inputs.commit_sha,
            "config": str(args.config),
            "team_config": inputs.settings.team_config,
            "mode": args.mode,
            "seed": args.seed,
            "num_trials_sampled": args.num_trials,
            "num_robots": len(scene.robots),
            "box_mass_kg": scene.box.mass_kg,
            "push_speed_mps": inputs.run.settings.team.push_control.push_speed_mps,
            "genesis_device": args.device,
        },
        "summary": {
            "num_trials": num,
            "success_rate": sum(bool(row["success"]) for row in rows) / num,
            "task_success_rate": sum(bool(row["task_success"]) for row in rows) / num,
            "task_agrees_with_ground_truth": sum(row["success"] == row["task_success"] for row in rows) / num,
            "corridor_open_rate": sum(bool(row["corridor_opened"]) for row in rows) / num,
            "mean_open_time_s": _mean([float(row["open_time_s"]) for row in rows if row["open_time_s"] is not None]),
            "mean_success_time_s": _mean(
                [float(row["success_time_s"]) for row in rows if row["success_time_s"] is not None]
            ),
            "mean_box_dx_m": _mean([float(row["box_dx_m"]) for row in rows]),
            "mean_box_max_abs_yaw_deg": _mean(yaws),
            "max_box_max_abs_yaw_deg": max(yaws),
            "mean_push_start_spread_s": _mean(spreads),
            "max_push_start_spread_s": max(spreads) if spreads else None,
            "falls": sum(int(row["falls"]) for row in rows),
            "task_reasons": dict(Counter(str(row["task_reason"]) for row in rows)),
        },
    }


def _metrics(args: argparse.Namespace, inputs: _Inputs, records: list[TeamTrialRecord],
             latency_probe: RoundTripLatencyProbe) -> dict[str, object]:  # fmt: skip
    selections: Counter[str] = Counter()
    delays: Counter[int] = Counter()
    for record in records:
        selections.update({selection.value: count for selection, count in record.target_selections.items()})
        delays.update(record.target_delay_steps)
    sim_time_s = sum(record.end_time_s for record in records)
    wall_time_s = sum(record.wall_time_s for record in records)
    return {
        "genesis_device": args.device,
        "max_real_time_factor": inputs.settings.runtime.max_real_time_factor,
        "num_trials": len(records),
        "physics_steps_by_target_selection": dict(selections),
        "physics_steps_by_target_delay_steps": {str(steps): count for steps, count in sorted(delays.items())},
        "sim_time_s": sim_time_s,
        "wall_time_s": wall_time_s,
        "real_time_factor": sim_time_s / wall_time_s if wall_time_s > 0 else math.nan,
        "joint_target_round_trip_s": latency_probe.round_trip.summary(),
    }


def _save(output_dir: Path, rows: list[dict[str, object]], summary: dict[str, object]) -> None:
    with (output_dir / TRIALS_FILENAME).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / SUMMARY_FILENAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """試行を続けて行い、記録を保存する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        inputs = _load_inputs(args.config)
    except (FileNotFoundError, ConfigValidationError, InvalidTeamTaskConfigError) as error:
        logger.error("設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    except GitCommitUnavailableError as error:
        logger.error("%s", error)
        sys.exit(EXIT_GIT_UNAVAILABLE)
    torch.set_num_threads(inputs.settings.runtime.torch_num_threads)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        logger.error("%s", TeamOutputExistsError(f"保存先が既にある（上書きしない）: {output_dir}"))
        sys.exit(EXIT_OUTPUT_EXISTS)
    output_dir.mkdir(parents=True)
    # 歩行方策・協調のノードが終了時に計測値を書く場所（launch が同じパスを渡す）
    (output_dir / NODE_METRICS_DIRNAME).mkdir()

    DEVICES[args.device](args.seed)
    rclpy.init(args=raw_argv)
    node = Node(NODE_NAME)
    executor_thread = ExecutorThread(node)
    left = f"（保存先 {output_dir} は途中のまま残っている。削除してから再実行する）"
    try:
        records, latency_probe = _run_trials(args, inputs, node, executor_thread, output_dir)
    except SimulationSetupError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    except (PeerNodeTimeoutError, PeerNodeRejectedError, NoTeamJointTargetsAppliedError, NoTaskResultError) as error:
        executor_error = executor_thread.find_error_or_none()
        if executor_error is not None:
            logger.error("ROS 2 の受信のスレッドが止まった: %r（%s）%s", executor_error, error, left)
            sys.exit(EXIT_ROS_EXECUTOR_FAILED)
        logger.error("方策・協調・タスク管理のノードとのやり取りに失敗した: %s%s", error, left)
        sys.exit(EXIT_PEER_NODE_FAILED)
    except ExecutorThreadFailedError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_ROS_EXECUTOR_FAILED)
    except VideoRecordingError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_VIDEO_RECORDING_FAILED)
    finally:
        executor_thread.executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

    rows = [_trial_row(record) for record in records]
    summary = _summary(args, inputs, rows)
    logger.info("評価指標: %s", summary["summary"])
    _save(output_dir, rows, summary)
    write_node_metrics(output_dir / METRICS_FILENAME, _metrics(args, inputs, records, latency_probe))
    logger.info("評価結果を保存した: %s", output_dir)


if __name__ == "__main__":
    main()
