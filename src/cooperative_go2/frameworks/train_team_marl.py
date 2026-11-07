"""2 台の協調を IPPO・MAPPO で学習する CLI。

例（リポジトリのルートで、source scripts/setup/env.sh したシェルで実行する）:
    uv run go2-train-team-marl --config configs/push_multi.yaml --algorithm mappo --parameter-sharing shared \\
        --run-name mappo-shared-seed21

出力: outputs/team_marl/<run-name>/ に checkpoints/agent_<ステップ数>.pt、TensorBoard のログ、
設定の写し（push_multi.yaml と scene.yaml）、学習方式（variant.json）、コミット（git_commit.txt）。
"""

import argparse
import logging
import sys
from pathlib import Path

from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.frameworks.git_revision import GitCommitUnavailableError, current_commit_sha
from cooperative_go2.interface_adapters.config_loader import ConfigValidationError, load_team_marl_run_settings
from cooperative_go2.interface_adapters.gait_run_store import InvalidRunNameError, resolve_run_dir
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.skrl_marl_config import MarlAlgorithm, MarlVariant
from cooperative_go2.interface_adapters.skrl_team_training import (
    FileTeamMarlRunStore,
    SkrlTeamTrainerFactory,
    TeamMarlRunRecord,
)
from cooperative_go2.interface_adapters.team_marl_env import TeamSizeError
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError
from cooperative_go2.use_cases.train_push import RunAlreadyExistsError, train_push

TEAM_MARL_RUNS_ROOT = Path("outputs/team_marl")
PARAMETER_SHARING_CHOICES = {"shared": True, "separate": False}
EXIT_INVALID_INPUT = 2
EXIT_RUN_ALREADY_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2 台の Go2 の協調を IPPO・MAPPO で学習する")
    parser.add_argument(
        "--config", type=Path, required=True, help="協調学習の設定ファイル（例: configs/push_multi.yaml）"
    )
    parser.add_argument("--algorithm", choices=[algorithm.value for algorithm in MarlAlgorithm], required=True)
    parser.add_argument(
        "--parameter-sharing", choices=sorted(PARAMETER_SHARING_CHOICES), required=True,
        help="shared は 2 台で Actor と Critic を共有し、separate はロボットごとに別々に学習する",
    )  # fmt: skip
    parser.add_argument(
        "--run-name", required=True, help=f"出力先 {TEAM_MARL_RUNS_ROOT}/<run-name>/ の名前。既にあれば上書きせず止まる"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """設定ファイルを読み込み、協調を学習する。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    variant = MarlVariant(MarlAlgorithm(args.algorithm), PARAMETER_SHARING_CHOICES[args.parameter_sharing])
    try:
        run = load_team_marl_run_settings(args.config)
        run_dir = resolve_run_dir(TEAM_MARL_RUNS_ROOT, args.run_name)
        commit_sha = current_commit_sha()
    except (FileNotFoundError, ConfigValidationError, InvalidRunNameError, GitCommitUnavailableError) as error:
        logger.error("入力が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)

    seed = run.settings.training.seed
    init_genesis_on_cuda(seed)
    record = TeamMarlRunRecord(args.config, Path(run.settings.scene_config), variant, commit_sha)
    factory = SkrlTeamTrainerFactory(run_dir, run, variant, make_seeded_generator(seed))
    left_hint = f"（出力先 {run_dir} は残っている。削除するか別の --run-name で再実行する）"
    try:
        train_push(run.settings.training.max_iterations, FileTeamMarlRunStore(run_dir, record), factory)
    except RunAlreadyExistsError as error:
        logger.error("%s", error)
        sys.exit(EXIT_RUN_ALREADY_EXISTS)
    except (FileNotFoundError, ConfigValidationError, InvalidPushConfigError, TeamSizeError, CheckpointNotFoundError,
            CheckpointLoadError) as error:  # fmt: skip
        # FileNotFoundError・ConfigValidationError は、学習器を作るときに読む歩行方策の設定の写しが無い・壊れているとき
        logger.error("入力が不正: %s%s", error, left_hint)
        sys.exit(EXIT_INVALID_INPUT)
    except SimulationSetupError as error:
        logger.error("%s%s", error, left_hint)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    logger.info("学習が終わった: %s（%s）", run_dir, variant.name)


if __name__ == "__main__":
    main()
