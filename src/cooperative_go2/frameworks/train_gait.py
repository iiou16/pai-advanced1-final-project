"""歩行方策を学習する CLI。

例:
    uv run go2-train-gait --config configs/gait.yaml --run-name gait-baseline

出力: outputs/gait/<run-name>/ に model_<反復>.pt、TensorBoard のログ、設定ファイルの写し（gait.yaml）。
"""

import argparse
import logging
import sys
from pathlib import Path

from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.interface_adapters.config_loader import ConfigValidationError, load_gait_settings
from cooperative_go2.interface_adapters.gait_run_store import FileGaitRunStore
from cooperative_go2.interface_adapters.genesis_flat_scene import SimulationSetupError
from cooperative_go2.interface_adapters.rsl_rl_gait import RslRlGaitTrainerFactory
from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError, train_gait

GAIT_RUNS_ROOT = Path("outputs/gait")
EXIT_INVALID_CONFIG = 2
EXIT_RUN_ALREADY_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Go2 の歩行方策を PPO で学習する")
    parser.add_argument("--config", type=Path, required=True, help="歩行学習の設定ファイル（例: configs/gait.yaml）")
    parser.add_argument(
        "--run-name", required=True, help=f"出力先 {GAIT_RUNS_ROOT}/<run-name>/ の名前。既にあれば上書きせず止まる"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """設定ファイルを読み込み、歩行方策を学習する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    # 設定ファイルが無いことも内容の不正も、利用者が設定を直すべき点で同じなので同じ終了コードにする
    try:
        settings = load_gait_settings(args.config)
    except (FileNotFoundError, ConfigValidationError) as error:
        logger.error("設定ファイルを読み込めない: %s", error)
        sys.exit(EXIT_INVALID_CONFIG)

    seed = settings.gait.training.seed
    init_genesis_on_cuda(seed)
    run_dir = GAIT_RUNS_ROOT / args.run_name
    trainer_factory = RslRlGaitTrainerFactory(run_dir, settings.rsl_rl, make_seeded_generator(seed))
    try:
        train_gait(settings.gait, FileGaitRunStore(run_dir, args.config), trainer_factory)
    except RunAlreadyExistsError as error:
        logger.error("%s", error)
        sys.exit(EXIT_RUN_ALREADY_EXISTS)
    except SimulationSetupError as error:
        logger.error("%s", error)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    logger.info("学習が終わった: %s", run_dir)


if __name__ == "__main__":
    main()
