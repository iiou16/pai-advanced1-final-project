"""箱押しの高レベル方策を学習する CLI。

例（リポジトリのルートで実行する）:
    uv run go2-train-push --config configs/push_single.yaml --run-name push-stage1

出力: outputs/push/<run-name>/ に model_<反復>.pt、TensorBoard のログ、設定の写し（push_single.yaml と scene.yaml）。
"""

import argparse
import logging
import sys
from pathlib import Path

from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_push_scene_settings,
    load_push_training_settings,
)
from cooperative_go2.interface_adapters.gait_run_store import InvalidRunNameError, resolve_run_dir
from cooperative_go2.interface_adapters.rsl_rl_gait import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.rsl_rl_push import FilePushRunStore, RslRlPushTrainerFactory
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError
from cooperative_go2.use_cases.train_push import RunAlreadyExistsError, train_push

PUSH_RUNS_ROOT = Path("outputs/push")
EXIT_INVALID_INPUT = 2
EXIT_RUN_ALREADY_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go2 の箱押しの高レベル方策を PPO で学習する")
    parser.add_argument(
        "--config", type=Path, required=True, help="箱押しの設定ファイル（例: configs/push_single.yaml）"
    )
    parser.add_argument(
        "--run-name", required=True, help=f"出力先 {PUSH_RUNS_ROOT}/<run-name>/ の名前。既にあれば上書きせず止まる"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """設定ファイルを読み込み、高レベル方策を学習する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    try:
        settings = load_push_training_settings(args.config)
        scene_config = Path(settings.scene_config)
        scene = load_push_scene_settings(scene_config)
        run_dir = resolve_run_dir(PUSH_RUNS_ROOT, args.run_name)
    except (FileNotFoundError, ConfigValidationError, InvalidRunNameError) as error:
        logger.error("入力が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)

    seed = settings.training.seed
    init_genesis_on_cuda(seed)
    factory = RslRlPushTrainerFactory(run_dir, settings, scene, make_seeded_generator(seed))
    left_hint = f"（出力先 {run_dir} は残っている。削除するか別の --run-name で再実行する）"
    try:
        train_push(settings.training.max_iterations, FilePushRunStore(run_dir, args.config, scene_config), factory)
    except RunAlreadyExistsError as error:
        logger.error("%s", error)
        sys.exit(EXIT_RUN_ALREADY_EXISTS)
    except (InvalidPushConfigError, CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("入力が不正: %s%s", error, left_hint)
        sys.exit(EXIT_INVALID_INPUT)
    except SimulationSetupError as error:
        logger.error("%s%s", error, left_hint)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    logger.info("学習が終わった: %s", run_dir)


if __name__ == "__main__":
    main()
