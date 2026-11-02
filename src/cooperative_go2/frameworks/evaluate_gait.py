"""学習済みの歩行方策に固定の速度指令を与え、追従誤差と転倒を測る CLI。

例:
    uv run go2-evaluate-gait --run-dir outputs/gait/gait-baseline --checkpoint 100 \\
        --vx 0.5 --vy 0.0 --yaw-rate 0.0 --num-envs 1 --duration-s 10 --seed 1001 \\
        --video outputs/gait/gait-baseline/eval_forward.mp4

結果は JSON で標準出力に書く。--video を付けると環境 0 を追従した評価動画を撮る。
現状の評価は決定的で、シードや環境数を変えても結果は変わらない（全環境が同じ初期状態から始まるため）。
"""

import argparse
import dataclasses
import json
import logging
import sys
from pathlib import Path

import genesis as gs

from cooperative_go2.entities.velocity_command import VelocityCommand
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import ConfigValidationError, load_gait_settings
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_flat_scene import CameraSpec, SimulationSetupError, build_flat_scene
from cooperative_go2.interface_adapters.genesis_gait_env import GenesisGo2GaitEnv
from cooperative_go2.interface_adapters.genesis_gait_rollout import GenesisGaitRolloutEnvironment
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder
from cooperative_go2.interface_adapters.rsl_rl_gait import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    load_rsl_rl_gait_policy,
)
from cooperative_go2.use_cases.evaluate_gait import (
    GaitEvaluationRequest,
    GaitEvaluator,
    InvalidEvaluationRequestError,
    SimulationDivergedError,
    plan_gait_evaluation,
)

# Go2 の全身と足元が入るよう、斜め後ろ上から撮る
VIDEO_CAMERA = CameraSpec(resolution=(640, 480), offset_m=(-1.5, -1.5, 0.8), fov_deg=40.0)
EXIT_INVALID_INPUT = 2
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_SIMULATION_DIVERGED = 5

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学習済みの Go2 歩行方策を固定の速度指令で評価する")
    parser.add_argument("--run-dir", type=Path, required=True, help="学習の出力先（outputs/gait/<run-name>）")
    parser.add_argument("--checkpoint", type=int, required=True, help="読み込む反復番号（model_<番号>.pt）")
    parser.add_argument("--vx", type=float, required=True, help="前後方向の速度指令 [m/s]。前が正")
    parser.add_argument("--vy", type=float, required=True, help="左右方向の速度指令 [m/s]。左が正")
    parser.add_argument("--yaw-rate", type=float, required=True, help="ヨー角速度の指令 [rad/s]。反時計回りが正")
    parser.add_argument("--num-envs", type=int, required=True, help="並列に走らせる環境の数")
    parser.add_argument("--duration-s", type=float, required=True, help="走らせる時間 [s]。エピソードの長さ以内")
    parser.add_argument("--seed", type=int, required=True, help="評価用の乱数シード。学習のシードとは別にする")
    parser.add_argument("--video", type=Path, help="評価動画（mp4）の保存先。省略すると撮らない")
    return parser.parse_args(argv)


def _request_from(args: argparse.Namespace) -> GaitEvaluationRequest:
    command = VelocityCommand(vx_mps=args.vx, vy_mps=args.vy, yaw_rate_radps=args.yaw_rate)
    return GaitEvaluationRequest(command=command, duration_s=args.duration_s, num_envs=args.num_envs, seed=args.seed)


def main(argv: list[str] | None = None) -> None:
    """チェックポイントを読み込んで評価し、結果を JSON で出力する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    # 設定の不在・不正、評価の条件の不正、チェックポイントの不在・破損は、いずれも利用者が入力を直すべき点で同じ
    try:
        settings = load_gait_settings(args.run_dir / RUN_CONFIG_FILENAME)
        plan = plan_gait_evaluation(settings.gait, _request_from(args))
    except (FileNotFoundError, ConfigValidationError, InvalidEvaluationRequestError) as error:
        logger.error("評価の条件が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)

    init_genesis_on_cuda(plan.seed)
    try:
        policy = load_rsl_rl_gait_policy(args.run_dir / f"model_{args.checkpoint}.pt", settings.rsl_rl, gs.device)
    except (CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("%s", error)
        sys.exit(EXIT_INVALID_INPUT)
    try:
        scene = build_flat_scene(plan.config, VIDEO_CAMERA if args.video is not None else None)
    except SimulationSetupError as error:
        logger.error("%s", error)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)

    env = GenesisGo2GaitEnv(plan.config, scene, make_seeded_generator(plan.seed))
    recorder = None
    if args.video is not None:
        recorder = GenesisVideoRecorder(scene.require_camera(), args.video, plan.config.env.control_dt_s)
    try:
        summary = GaitEvaluator(GenesisGaitRolloutEnvironment(env), policy, recorder).run(plan)
    except SimulationDivergedError as error:
        logger.error("%s", error)
        sys.exit(EXIT_SIMULATION_DIVERGED)
    print(json.dumps(dataclasses.asdict(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
