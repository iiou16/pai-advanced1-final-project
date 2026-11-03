"""ROS 2 のシミュレーションノード。Genesis のシーンを進める唯一のプロセス（要件 5.2）。

/clock・Go2 の関節と胴体・箱・接触・TF を配信し、歩行方策のノードの関節目標を受けて PD 制御する。
試行ごとに歩行方策のノードをリセットし、高レベル方策のノードに目標を依頼する（P3 では依頼者を兼ねる）。
成功と終了はシミュレーションの真値で判定し、学習環境での評価（go2-evaluate-push）と同じ形式で記録する。

起動例（launch から起動する。ロボットと同じ Namespace にする）:
    go2-ros2-sim --config configs/ros2_push_single.yaml --seed 2001 --num-trials 10 --device cpu \\
        --run-name ros2-wide60-seed2001 --video false --ros-args -r __ns:=/go2_a -r __node:=genesis_sim -p use_sim_time:=true

結果は outputs/ros2_eval/<run-name>/ に trials.csv・summary.json（go2-evaluate-push と同じ形式）、
ros2_metrics.json（関節目標の遅れ・実時間比）、--video true なら video.mp4 を保存する。保存先が既にあれば止まる。
"""

import argparse
import logging
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import rclpy
from genesis.utils.geom import default_friction
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.utilities import remove_ros_args

from cooperative_go2.entities.command_expiry import seconds_to_ns
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.go2_state import sample_reset_dof_pos
from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.entities.push_layout import PushLayout, sample_push_layout
from cooperative_go2.entities.push_task_config import require_compatible_with_gait
from cooperative_go2.entities.push_trial import summarize_push_trials
from cooperative_go2.frameworks.evaluate_push import (
    DEVICES,
    SCENE_ID,
    VIDEO_CAMERA,
    VIDEO_FILENAME,
    VIDEO_LOOKAT_OFFSET_M,
)
from cooperative_go2.frameworks.genesis_runtime import make_seeded_generator
from cooperative_go2.frameworks.git_revision import GitCommitUnavailableError, current_commit_sha
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    PushSceneSettings,
    PushTrainingSettings,
    Ros2PushSettings,
    load_gait_settings,
    load_push_scene_settings,
    load_push_training_settings,
    load_ros2_push_settings,
)
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_push_scene import PushSceneRequest, PushVideoSpec
from cooperative_go2.interface_adapters.genesis_push_simulation import GenesisPushSimulation
from cooperative_go2.interface_adapters.genesis_push_world import GenesisPushWorld
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder, VideoRecordingError
from cooperative_go2.interface_adapters.push_evaluation_store import (
    FilePushEvaluationStore,
    PushEvaluationConditions,
    PushEvaluationOutputExistsError,
)
from cooperative_go2.interface_adapters.real_time_pacer import RealTimePacer
from cooperative_go2.interface_adapters.ros2_metrics import RoundTripLatencyProbe, write_node_metrics
from cooperative_go2.interface_adapters.ros2_sim_ports import (
    PeerNodeRejectedError,
    PeerNodeTimeoutError,
    RosPushTaskPeers,
    RosWorldStatePublisher,
    subscribe_joint_targets,
)
from cooperative_go2.use_cases.joint_target_scheduler import JointTargetScheduler, JointTargetTiming
from cooperative_go2.use_cases.ros_push_trial import PushTrialPorts, PushTrialRecord, PushTrialRunner, PushTrialTiming
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

NODE_NAME = "genesis_sim"
OUTPUT_ROOT = Path("outputs/ros2_eval")
METRICS_FILENAME = "ros2_metrics.json"
ALGORITHM = "policy_ros2"
# launch の引数をそのまま渡せるよう、真偽を文字列で受け取る
VIDEO_ON = "true"
VIDEO_OFF = "false"
EXIT_INVALID_INPUT = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_PEER_NODE_FAILED = 5
EXIT_GIT_UNAVAILABLE = 6
EXIT_VIDEO_RECORDING_FAILED = 7

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _Inputs:
    settings: Ros2PushSettings
    push: PushTrainingSettings
    scene: PushSceneSettings
    gait: GaitConfig
    commit_sha: str


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROS 2 のシミュレーションノード（1 台の箱押し）")
    parser.add_argument(
        "--config", type=Path, required=True, help="ROS 2 の実行系の設定（configs/ros2_push_single.yaml 形式）"
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="評価用の乱数シード（配置と初期姿勢）。学習のシードとは別にする"
    )
    parser.add_argument("--num-trials", type=int, required=True, help="続けて行う試行の数")
    parser.add_argument("--device", choices=sorted(DEVICES), required=True, help="Genesis を動かすデバイス")
    parser.add_argument("--run-name", required=True, help=f"実行の名前。{OUTPUT_ROOT}/<run-name> に保存する")
    parser.add_argument(
        "--video", choices=[VIDEO_ON, VIDEO_OFF], required=True, help="評価動画を撮るか（全試行を 1 本に続けて撮る）"
    )
    args = parser.parse_args(argv)
    if args.num_trials < 1:
        parser.error(f"--num-trials は 1 以上が必要: {args.num_trials}")
    return args


def _load_inputs(config_path: Path) -> _Inputs:
    settings = load_ros2_push_settings(config_path)
    push = load_push_training_settings(Path(settings.push_config))
    scene = load_push_scene_settings(Path(push.scene_config))
    gait = load_gait_settings(Path(push.gait_policy.run_dir) / RUN_CONFIG_FILENAME).gait
    require_compatible_with_gait(push.task, gait)
    return _Inputs(settings, push, scene, gait, current_commit_sha())


@dataclass(frozen=True)
class _TrialsOutcome:
    records: list[PushTrialRecord]
    latency_probe: RoundTripLatencyProbe


def _run_trials(args: argparse.Namespace, inputs: _Inputs, node: Node, output_dir: Path) -> _TrialsOutcome:
    task = inputs.push.task
    gait_env = inputs.gait.env
    video = None
    if args.video == VIDEO_ON:
        video = PushVideoSpec(VIDEO_CAMERA, VIDEO_LOOKAT_OFFSET_M, task.success.goal_radius_m)
    simulation = GenesisPushSimulation.build(inputs.gait, inputs.scene, PushSceneRequest(1, video))
    world = GenesisPushWorld(simulation)
    control_dt_ns = seconds_to_ns(gait_env.control_dt_s)
    timing = JointTargetTiming(
        control_dt_ns=control_dt_ns,
        latency_steps=1 if gait_env.simulate_action_latency else 0,
        expiry_ns=seconds_to_ns(inputs.settings.command_expiry.joint_target_s),
    )
    scheduler = JointTargetScheduler(timing, world.default_dof_pos)
    latency_probe = RoundTripLatencyProbe()
    subscribe_joint_targets(node, scheduler, latency_probe)
    runtime = inputs.settings.runtime
    peers = RosPushTaskPeers(node, runtime.reply_timeout_s)
    peers.wait_for_peers(runtime.startup_timeout_s)
    ports = PushTrialPorts(
        world=world,
        publisher=RosWorldStatePublisher(node, latency_probe),
        peers=peers,
        pacer=RealTimePacer(gait_env.control_dt_s, runtime.max_real_time_factor),
        wall_clock=time.perf_counter,
    )
    trial_timing = PushTrialTiming(control_dt_ns, task.gait_steps_per_high_level_step(gait_env.control_dt_s))
    runner = PushTrialRunner(ports, scheduler, task, trial_timing)
    generator = make_seeded_generator(args.seed)
    recorder = None
    if args.video == VIDEO_ON:
        camera = simulation.scene.ground_and_robot.require_camera()
        recorder = GenesisVideoRecorder(camera, output_dir / VIDEO_FILENAME, gait_env.control_dt_s)
        recorder.start()
    # 全試行の配置とリセット時の関節角を、学習環境の最初のリセット（Go2PushEnv）と同じ順でまとめて引く。
    # 同じシード・同じデバイスなら、go2-evaluate-push を --num-envs <試行の数> で評価したときの各環境の最初の試行と、
    # 試行ごとに同じ初期条件になり、ROS 2 を経由しない評価と 1 試行ずつ比べられる
    layouts = sample_push_layout(inputs.scene.box.initial_xy_m, task.reset, args.num_trials, generator)
    noise_rad = gait_env.reset_joint_pos_noise_rad
    reset_dof_pos = sample_reset_dof_pos(simulation.default_dof_pos, args.num_trials, noise_rad, generator)
    records = []
    for trial_index in range(args.num_trials):
        rows = slice(trial_index, trial_index + 1)
        layout = PushLayout(box_xy_m=layouts.box_xy_m[rows].cpu(), goal_xy_m=layouts.goal_xy_m[rows].cpu())
        record = runner.run_trial(trial_index, layout, reset_dof_pos[rows].cpu())
        logger.info(
            "試行 %d: %s（%.1f s、実時間 %.1f s）", trial_index, record.result.termination_reason.name,
            record.sim_time_s, record.result.wall_time_s,
        )  # fmt: skip
        records.append(record)
    if recorder is not None:
        recorder.stop()
    return _TrialsOutcome(records, latency_probe)


def _conditions(args: argparse.Namespace, inputs: _Inputs) -> PushEvaluationConditions:
    policy = inputs.settings.push_policy
    task = inputs.push.task
    return PushEvaluationConditions(
        run_id=args.run_name,
        commit_sha=inputs.commit_sha,
        scene_id=SCENE_ID,
        seed=args.seed,
        algorithm=ALGORITHM,
        checkpoint=f"{policy.run_dir}/model_{policy.checkpoint}.pt",
        mass_kg=inputs.scene.box.mass_kg,
        # 接触する 2 面の摩擦は大きい方が使われる。床は Genesis の既定値のまま
        friction=max(inputs.scene.box.friction, default_friction()),
        high_level_period_s=task.episode.high_level_period_s,
        gait_steps_per_high_level_step=task.gait_steps_per_high_level_step(inputs.gait.env.control_dt_s),
        # 並列環境は 1 つで、試行を続けて行う
        num_envs=1,
    )


def _metrics(args: argparse.Namespace, settings: Ros2PushSettings, outcome: _TrialsOutcome) -> dict[str, object]:
    records = outcome.records
    selections: Counter[str] = Counter()
    delays: Counter[int] = Counter()
    for record in records:
        selections.update({selection.value: count for selection, count in record.target_selections.items()})
        delays.update(record.target_delay_steps)
    sim_time_s = sum(record.sim_time_s for record in records)
    wall_time_s = sum(record.result.wall_time_s for record in records)
    return {
        "genesis_device": args.device,
        "max_real_time_factor": settings.runtime.max_real_time_factor,
        "num_trials": len(records),
        "physics_steps_by_target_selection": dict(selections),
        "physics_steps_by_target_delay_steps": {str(steps): count for steps, count in sorted(delays.items())},
        "sim_time_s": sim_time_s,
        "wall_time_s": wall_time_s,
        "real_time_factor": sim_time_s / wall_time_s,
        # 状態の配信から、その状態で作った関節目標が届くまで（DDS 2 回分 ＋ 歩行方策の推論）
        "joint_target_round_trip_s": outcome.latency_probe.round_trip.summary(),
    }


def _spin_in_background(node: Node) -> MultiThreadedExecutor:
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    return executor


def main(argv: list[str] | None = None) -> None:
    """試行を続けて行い、記録を保存する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    raw_argv = sys.argv if argv is None else argv
    args = _parse_args(remove_ros_args(raw_argv)[1:])
    try:
        inputs = _load_inputs(args.config)
    except (FileNotFoundError, ConfigValidationError, InvalidPushConfigError) as error:
        logger.error("設定が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    except GitCommitUnavailableError as error:
        logger.error("%s", error)
        sys.exit(EXIT_GIT_UNAVAILABLE)
    output_dir = OUTPUT_ROOT / args.run_name
    store = FilePushEvaluationStore(output_dir)
    try:
        store.initialize()
    except PushEvaluationOutputExistsError as error:
        logger.error("%s", error)
        sys.exit(EXIT_OUTPUT_EXISTS)

    DEVICES[args.device](args.seed)
    rclpy.init(args=raw_argv)
    node = Node(NODE_NAME)
    executor = _spin_in_background(node)
    left = f"（保存先 {output_dir} は途中のまま残っている。削除してから再実行する）"
    try:
        outcome = _run_trials(args, inputs, node, output_dir)
    except SimulationSetupError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    except (PeerNodeTimeoutError, PeerNodeRejectedError) as error:
        logger.error("方策のノードとのやり取りに失敗した: %s%s", error, left)
        sys.exit(EXIT_PEER_NODE_FAILED)
    except VideoRecordingError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_VIDEO_RECORDING_FAILED)
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()

    results = [record.result for record in outcome.records]
    summary = summarize_push_trials(results, inputs.push.task.episode.high_level_period_s)
    logger.info("評価指標: %s", summary)
    store.save(_conditions(args, inputs), results, summary)
    write_node_metrics(output_dir / METRICS_FILENAME, _metrics(args, inputs.settings, outcome))
    logger.info("評価結果を保存した: %s", output_dir)


if __name__ == "__main__":
    main()
