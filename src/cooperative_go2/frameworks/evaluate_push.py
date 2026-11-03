"""箱押しの方策（規則ベース・学習済み）を、並列環境ごとに 1 試行ずつ評価し、記録を保存する CLI。

例（リポジトリのルートで実行する）:
    uv run go2-evaluate-push --config configs/push_single.yaml --algorithm rule_based \\
        --num-envs 64 --seed 2001 --device cuda --run-name e1-mass2kg
    uv run go2-evaluate-push --config configs/push_single.yaml --algorithm policy \\
        --policy-run-dir outputs/push/push-stage1 --checkpoint 500 \\
        --num-envs 64 --seed 2001 --device cuda --run-name e2-stage1 --video

結果は outputs/push_eval/<run-name>/ に、trials.csv（試行ごとに 1 行、要件 9.2 の列）と summary.json
（条件と評価指標）として保存する。--video を付けると、環境 0 を追従した評価動画（video.mp4）も撮る。
保存先が既にあれば上書きせずに止まる。評価の条件（配置・成功判定など）は --config から取る。
--box-mass-kg を付けると、シーンの設定の箱の質量だけを置き換えて評価する（箱の質量を決める実験用）。
"""

import argparse
import dataclasses
import logging
import subprocess
import sys
import time
from pathlib import Path

import genesis as gs
from genesis.utils.geom import default_friction

from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.entities.push_trial import PushEpisodeResult, summarize_push_trials
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    PushSceneSettings,
    PushTrainingSettings,
    load_gait_settings,
    load_push_scene_settings,
    load_push_training_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_flat_scene import CameraSpec
from cooperative_go2.interface_adapters.genesis_push_scene import PushSceneRequest, PushVideoSpec
from cooperative_go2.interface_adapters.genesis_push_simulation import GenesisPushSimulation
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder, VideoRecordingError
from cooperative_go2.interface_adapters.push_env import Go2PushEnv, PushEnvSetup
from cooperative_go2.interface_adapters.push_evaluation_adapters import (
    Go2PushEvaluationEnvironment,
    PolicyCommandSource,
    RuleBasedCommandSource,
    load_push_policy,
)
from cooperative_go2.interface_adapters.push_evaluation_store import (
    FilePushEvaluationStore,
    PushEvaluationConditions,
    PushEvaluationOutputExistsError,
)
from cooperative_go2.interface_adapters.rsl_rl_gait import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.use_cases.evaluate_push import EvaluationDidNotFinishError, PushCommandSource, evaluate_push
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

OUTPUT_ROOT = Path("outputs/push_eval")
VIDEO_FILENAME = "video.mp4"
SCENE_ID = "S1_push"
ALGORITHM_RULE_BASED = "rule_based"
ALGORITHM_POLICY = "policy"
DEVICES = {"cuda": init_genesis_on_cuda, "cpu": init_genesis_on_cpu}
# Go2 と前方の箱・目標が入るよう、斜め後ろ上から撮る
VIDEO_CAMERA = CameraSpec(resolution=(960, 540), offset_m=(-1.2, -2.6, 1.6), fov_deg=50.0)
EXIT_INVALID_INPUT = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_EVALUATION_DID_NOT_FINISH = 5
EXIT_GIT_UNAVAILABLE = 6
EXIT_VIDEO_RECORDING_FAILED = 7

logger = logging.getLogger(__name__)


class GitCommitUnavailableError(RuntimeError):
    """評価の記録に残すコミットを git から取得できないことを表す。"""


class MissingPolicyArgumentsError(ValueError):
    """学習済みの方策を評価するのに、学習の出力先かチェックポイントが指定されていないことを表す。"""


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="箱押しの方策を並列環境ごとに 1 試行ずつ評価する")
    parser.add_argument("--config", type=Path, required=True, help="評価の条件（configs/push_single.yaml 形式）")
    parser.add_argument(
        "--algorithm", choices=[ALGORITHM_RULE_BASED, ALGORITHM_POLICY], required=True, help="速度指令を決める方式"
    )
    parser.add_argument("--policy-run-dir", type=Path, help="学習済みの方策の出力先（--algorithm policy のとき必須）")
    parser.add_argument("--checkpoint", type=int, help="学習済みの方策の反復番号（--algorithm policy のとき必須）")
    parser.add_argument("--num-envs", type=int, required=True, help="並列環境の数（= 試行の数）")
    parser.add_argument("--seed", type=int, required=True, help="評価用の乱数シード。学習のシードとは別にする")
    parser.add_argument("--device", choices=sorted(DEVICES), required=True, help="Genesis を動かすデバイス")
    parser.add_argument("--run-name", required=True, help=f"評価の実行の名前。{OUTPUT_ROOT}/<run-name> に保存する")
    parser.add_argument("--box-mass-kg", type=float, help="シーンの設定の箱の質量 [kg] を置き換える（実験用）")
    parser.add_argument("--video", action="store_true", help="環境 0 を追従した評価動画を撮る")
    return parser.parse_args(argv)


def _current_commit_sha() -> str:
    # 未コミットの変更がある状態で評価したことを記録から見分けられるよう、その場合は末尾に -dirty を付ける
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitCommitUnavailableError(f"コミットを取得できない: {error}") from error
    return f"{sha}-dirty" if status.strip() else sha


def _scene_settings(settings: PushTrainingSettings, box_mass_kg: float | None) -> PushSceneSettings:
    scene = load_push_scene_settings(Path(settings.scene_config))
    if box_mass_kg is None:
        return scene
    return dataclasses.replace(scene, box=dataclasses.replace(scene.box, mass_kg=box_mass_kg))


def _command_source(args: argparse.Namespace, settings: PushTrainingSettings) -> PushCommandSource:
    if args.algorithm == ALGORITHM_RULE_BASED:
        return RuleBasedCommandSource(settings.rule_based, settings.task.success.goal_radius_m)
    if args.policy_run_dir is None or args.checkpoint is None:
        raise MissingPolicyArgumentsError("--algorithm policy には --policy-run-dir と --checkpoint が必要")
    return PolicyCommandSource(load_push_policy(args.policy_run_dir, args.checkpoint, gs.device))


def _evaluate(
    args: argparse.Namespace, settings: PushTrainingSettings, scene: PushSceneSettings, output_dir: Path
) -> tuple[tuple[PushEpisodeResult, ...], Go2PushEnv]:
    run_dir = Path(settings.gait_policy.run_dir)
    gait = load_gait_settings(run_dir / RUN_CONFIG_FILENAME).gait
    task = dataclasses.replace(
        settings.task, episode=dataclasses.replace(settings.task.episode, num_envs=args.num_envs)
    )
    DEVICES[args.device](args.seed)
    source = _command_source(args, settings)
    generator = make_seeded_generator(args.seed)
    video = PushVideoSpec(VIDEO_CAMERA, task.success.goal_radius_m) if args.video else None
    simulation = GenesisPushSimulation.build(gait, scene, PushSceneRequest(args.num_envs, video))
    controller = load_gait_controller(run_dir, settings.gait_policy.checkpoint, args.num_envs, gs.device)
    env = Go2PushEnv(PushEnvSetup(task, gait, scene.box.initial_xy_m, generator), simulation, controller)
    environment = Go2PushEvaluationEnvironment(env)
    if not args.video:
        return evaluate_push(environment, source, time.perf_counter), env
    camera = simulation.scene.ground_and_robot.require_camera()
    recorder = GenesisVideoRecorder(camera, output_dir / VIDEO_FILENAME, gait.env.control_dt_s)
    recorder.start()
    results = evaluate_push(environment, source, time.perf_counter)
    recorder.stop()
    return results, env


def _conditions(
    args: argparse.Namespace, commit_sha: str, scene: PushSceneSettings, env: Go2PushEnv
) -> PushEvaluationConditions:
    checkpoint = None if args.algorithm == ALGORITHM_RULE_BASED else f"{args.policy_run_dir}/model_{args.checkpoint}.pt"
    return PushEvaluationConditions(
        run_id=args.run_name,
        commit_sha=commit_sha,
        scene_id=SCENE_ID,
        seed=args.seed,
        algorithm=args.algorithm,
        checkpoint=checkpoint,
        mass_kg=scene.box.mass_kg,
        # 接触する 2 面の摩擦は大きい方が使われる。床は Genesis の既定値のまま
        friction=max(scene.box.friction, default_friction()),
        high_level_period_s=env.cfg.episode.high_level_period_s,
        gait_steps_per_high_level_step=env.gait_steps_per_high_level_step,
        num_envs=args.num_envs,
    )


def main(argv: list[str] | None = None) -> None:
    """方策を評価して記録を保存する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        settings = load_push_training_settings(args.config)
        scene = _scene_settings(settings, args.box_mass_kg)
        commit_sha = _current_commit_sha()
    except (FileNotFoundError, ConfigValidationError) as error:
        logger.error("評価の条件が不正: %s", error)
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

    left = f"（保存先 {output_dir} は途中のまま残っている。削除してから再実行する）"
    try:
        results, env = _evaluate(args, settings, scene, output_dir)
    except (FileNotFoundError, ConfigValidationError, InvalidPushConfigError, MissingPolicyArgumentsError,
            CheckpointNotFoundError, CheckpointLoadError) as error:  # fmt: skip
        logger.error("評価の条件が不正: %s%s", error, left)
        sys.exit(EXIT_INVALID_INPUT)
    except SimulationSetupError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    except EvaluationDidNotFinishError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_EVALUATION_DID_NOT_FINISH)
    except VideoRecordingError as error:
        logger.error("%s%s", error, left)
        sys.exit(EXIT_VIDEO_RECORDING_FAILED)

    summary = summarize_push_trials(results, env.cfg.episode.high_level_period_s)
    logger.info("評価指標: %s", summary)
    store.save(_conditions(args, commit_sha, scene, env), results, summary)
    logger.info("評価結果を保存した: %s", output_dir)


if __name__ == "__main__":
    main()
