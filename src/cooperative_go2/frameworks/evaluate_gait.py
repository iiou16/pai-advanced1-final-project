"""学習済みの歩行方策に評価シナリオの速度指令を与え、区間ごとの追従誤差と転倒を測る CLI。

例（リポジトリのルートで実行する）:
    uv run go2-evaluate-gait --run-dir outputs/gait/gait-command-tracking --checkpoint 1500 \\
        --scenarios configs/gait_eval.yaml --num-envs 64 --seed 1001 --video

結果は <run-dir>/eval/model_<checkpoint>_seed_<seed>/ に、summary.json（条件と全結果）、segments.csv（区間ごと）、
シナリオのファイルの写しとして保存する。--video を付けると、シナリオごとに環境 0 を追従した評価動画を撮る。
保存先が既にあれば上書きせずに止まる。環境ごとにリセット時の関節角がシードに従ってばらつくため、
シードを変えると結果も変わる。
"""

import argparse
import logging
import sys
from pathlib import Path

import genesis as gs

from cooperative_go2.entities.gait_evaluation import ScenarioSummary
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_gait_evaluation_suite,
    load_gait_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import checkpoint_path
from cooperative_go2.interface_adapters.gait_env import Go2GaitEnv
from cooperative_go2.interface_adapters.gait_evaluation_store import (
    EvaluationOutputExistsError,
    FileGaitEvaluationStore,
    GaitEvaluationRecord,
    evaluation_output_dir,
)
from cooperative_go2.interface_adapters.gait_rollout import Go2GaitRolloutEnvironment
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_flat_scene import CameraSpec
from cooperative_go2.interface_adapters.genesis_gait_simulation import GenesisGaitSimulation
from cooperative_go2.interface_adapters.genesis_video_recorder import GenesisVideoRecorder, VideoRecordingError
from cooperative_go2.interface_adapters.rsl_rl_gait import (
    CheckpointLoadError,
    CheckpointNotFoundError,
    RslRlGaitPolicy,
    load_rsl_rl_gait_policy,
)
from cooperative_go2.use_cases.evaluate_gait import (
    GaitEvaluationPlan,
    GaitEvaluationRequest,
    GaitEvaluator,
    InvalidEvaluationRequestError,
    plan_gait_evaluation,
)
from cooperative_go2.use_cases.simulation_errors import SimulationDivergedError, SimulationSetupError

# Go2 の全身と足元が入るよう、斜め後ろ上から撮る
VIDEO_CAMERA = CameraSpec(resolution=(640, 480), offset_m=(-1.5, -1.5, 0.8), fov_deg=40.0)
EXIT_INVALID_INPUT = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_SIMULATION_DIVERGED = 5
EXIT_VIDEO_RECORDING_FAILED = 6

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学習済みの Go2 歩行方策を評価シナリオの速度指令で評価する")
    parser.add_argument("--run-dir", type=Path, required=True, help="学習の出力先（outputs/gait/<run-name>）")
    parser.add_argument("--checkpoint", type=int, required=True, help="読み込む反復番号（model_<番号>.pt）")
    parser.add_argument("--scenarios", type=Path, required=True, help="評価シナリオのファイル（例: configs/gait_eval.yaml）")
    parser.add_argument("--num-envs", type=int, required=True, help="並列に走らせる環境の数")
    parser.add_argument("--seed", type=int, required=True, help="評価用の乱数シード。学習のシードとは別にする")
    parser.add_argument("--video", action="store_true", help="シナリオごとに評価動画（mp4）を撮る")
    return parser.parse_args(argv)


def _run_in_simulation(
    plan: GaitEvaluationPlan, policy: RslRlGaitPolicy, video_store: FileGaitEvaluationStore | None
) -> tuple[ScenarioSummary, ...]:
    """シミュレーションを構築して評価シナリオを走らせる。video_store を渡すとシナリオごとの評価動画をそこに保存する。"""
    simulation = GenesisGaitSimulation.build(plan.config, VIDEO_CAMERA if video_store is not None else None)
    env = Go2GaitEnv(plan.config, simulation, make_seeded_generator(plan.seed))
    recorder_factory = None
    if video_store is not None:
        camera = simulation.scene.require_camera()
        control_dt_s = plan.config.env.control_dt_s

        def recorder_factory(scenario_name: str) -> GenesisVideoRecorder:
            return GenesisVideoRecorder(camera, video_store.video_path(scenario_name), control_dt_s)

    return GaitEvaluator(Go2GaitRolloutEnvironment(env), policy, recorder_factory).run(plan)


def _log_summaries(summaries: tuple[ScenarioSummary, ...]) -> None:
    for summary in summaries:
        errors = ", ".join(
            f"{_format_error(segment.mean_lin_vel_error_mps)} m/s / {_format_error(segment.mean_yaw_rate_error_radps)} rad/s"
            for segment in summary.segments
        )
        logger.info("%s: 転倒率 %.3f、区間ごとの追従誤差（線速度 / ヨー）[%s]", summary.name, summary.fall_rate, errors)


def _format_error(mean_error: float | None) -> str:
    # None は全環境が転倒し続けて追従誤差を集計できなかった区間
    return "集計なし" if mean_error is None else f"{mean_error:.3f}"


def _left_output_hint(output_dir: Path) -> str:
    return f"（保存先 {output_dir} は途中のまま残っている。削除してから再実行する）"


def main(argv: list[str] | None = None) -> None:
    """チェックポイントを読み込んで評価シナリオを走らせ、結果を保存する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    # 設定の不在・不正と評価の条件の不正は、いずれも利用者が引数か設定を直すべき点で同じなので同じ終了コードにする
    try:
        settings = load_gait_settings(args.run_dir / RUN_CONFIG_FILENAME)
        suite = load_gait_evaluation_suite(args.scenarios)
        request = GaitEvaluationRequest(suite=suite, num_envs=args.num_envs, seed=args.seed)
        plan = plan_gait_evaluation(settings.gait, request)
    except (FileNotFoundError, ConfigValidationError, InvalidEvaluationRequestError) as error:
        logger.error("評価の条件が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)

    init_genesis_on_cuda(plan.seed)
    # チェックポイントの不在も破損も、利用者が指定を直すべき入力の誤りとして扱う
    try:
        policy = load_rsl_rl_gait_policy(checkpoint_path(args.run_dir, args.checkpoint), settings.rsl_rl, gs.device)
    except (CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("%s", error)
        sys.exit(EXIT_INVALID_INPUT)

    output_dir = evaluation_output_dir(args.run_dir, args.checkpoint, plan.seed)
    store = FileGaitEvaluationStore(output_dir, args.scenarios)
    try:
        store.initialize()
    except EvaluationOutputExistsError as error:
        logger.error("%s", error)
        sys.exit(EXIT_OUTPUT_EXISTS)

    try:
        summaries = _run_in_simulation(plan, policy, store if args.video else None)
    except SimulationSetupError as error:
        logger.error("%s%s", error, _left_output_hint(output_dir))
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    except SimulationDivergedError as error:
        logger.error("%s%s", error, _left_output_hint(output_dir))
        sys.exit(EXIT_SIMULATION_DIVERGED)
    except VideoRecordingError as error:
        logger.error("%s%s", error, _left_output_hint(output_dir))
        sys.exit(EXIT_VIDEO_RECORDING_FAILED)
    record = GaitEvaluationRecord(
        run_dir=str(args.run_dir),
        checkpoint_iteration=args.checkpoint,
        seed=plan.seed,
        num_envs=plan.config.env.num_envs,
        settle_time_s=suite.settle_time_s,
        scenarios=summaries,
    )
    # 保存に失敗しても、長い評価の結果をログには残す
    _log_summaries(summaries)
    store.save(record)
    logger.info("評価結果を保存した: %s", output_dir)


if __name__ == "__main__":
    main()
