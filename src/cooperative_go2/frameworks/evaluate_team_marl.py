"""学習した協調の方策（IPPO・MAPPO）を、ROS 2 を通さずに並列環境で評価する CLI。

規則ベースの協調の並列環境での確認（scripts/p4_coordination_check/run_team_push.py）と同じシード・同じ並列環境の数・
同じデバイスで動かせば、同じ開始位置と関節角から始まる。trials.csv の列のうち、押し始めの時刻・失敗の理由・時間切れの
時刻は規則ベースの確認と定義が違う（cooperative_go2.interface_adapters.team_marl_evaluation を参照）。

例（source scripts/setup/env.sh したシェルで実行する）:
    uv run go2-evaluate-team-marl --run-dir outputs/team_marl/mappo-shared-seed21 --iteration 1000 \\
        --num-envs 32 --seed 5001 --drop-probability 0.0 --device cuda --output-dir outputs/team_marl_eval/<name>
"""

import argparse
import csv
import dataclasses
import json
import logging
import sys
from pathlib import Path

import genesis as gs

from cooperative_go2.entities.config_validation import InvalidConfigError
from cooperative_go2.entities.push_config import InvalidPushConfigError
from cooperative_go2.frameworks.genesis_runtime import init_genesis_on_cpu, init_genesis_on_cuda, make_seeded_generator
from cooperative_go2.frameworks.git_revision import GitCommitUnavailableError, current_commit_sha
from cooperative_go2.interface_adapters.config_loader import (
    ConfigValidationError,
    load_corridor_scene_settings,
    load_gait_settings,
)
from cooperative_go2.interface_adapters.gait_checkpoint import load_gait_controller
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.genesis_corridor_scene import CorridorSceneRequest
from cooperative_go2.interface_adapters.genesis_corridor_simulation import GenesisCorridorSimulation
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.skrl_team_policy import (
    InvalidRunRecordError,
    load_team_policy,
    load_team_run_settings_copy,
    load_variant,
)
from cooperative_go2.interface_adapters.team_marl_env import Go2TeamMarlEnv, TeamMarlEnvSetup, TeamSizeError
from cooperative_go2.interface_adapters.team_marl_evaluation import TeamMarlTrialRow, evaluate_team_policies
from cooperative_go2.use_cases.evaluate_push import EvaluationDidNotFinishError
from cooperative_go2.use_cases.simulation_errors import SimulationSetupError

TRIALS_FILENAME = "trials.csv"
SUMMARY_FILENAME = "summary.json"
GENESIS_INITIALIZERS = {"cpu": init_genesis_on_cpu, "cuda": init_genesis_on_cuda}
EXIT_INVALID_INPUT = 2
EXIT_OUTPUT_EXISTS = 3
EXIT_SIMULATION_SETUP_FAILED = 4
EXIT_EVALUATION_DID_NOT_FINISH = 5

logger = logging.getLogger(__name__)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="学習した 2 台の協調の方策を並列環境で評価する")
    parser.add_argument("--run-dir", type=Path, required=True, help="協調学習の出力先（outputs/team_marl/<run-name>）")
    parser.add_argument("--iteration", type=int, required=True, help="読み込むチェックポイントの更新の回数")
    parser.add_argument("--num-envs", type=int, required=True, help="試行の数（並列環境の数）")
    parser.add_argument(
        "--seed", type=int, required=True, help="開始位置・関節角を引くシード（学習のシードと別にする）"
    )
    parser.add_argument(
        "--drop-probability", type=float, required=True,
        help="評価での通信の欠落の確率。遅れは学習時の設定のまま。規則ベースの協調（欠落なし）と比べるなら 0",
    )  # fmt: skip
    # 学習と同じ cuda を基本にする。cpu はコードの動作を少数の環境で確かめるために使う
    parser.add_argument("--device", choices=sorted(GENESIS_INITIALIZERS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="結果の出力先。既にあれば上書きせず止まる")
    return parser.parse_args(argv)


def _summary(rows: list[TeamMarlTrialRow]) -> dict[str, object]:
    num = len(rows)
    successes = [row for row in rows if row.success]
    open_times = [row.open_time_s for row in rows if row.open_time_s is not None]
    spreads = [row.first_box_contact_spread_s for row in rows if row.first_box_contact_spread_s is not None]
    yaws = [row.box_max_abs_yaw_deg for row in rows]
    reasons: dict[str, int] = {}
    for row in rows:
        reasons[row.termination_reason] = reasons.get(row.termination_reason, 0) + 1
    return {
        "trials": num,
        "successes": len(successes),
        "corridor_opened": len(open_times),
        "mean_success_time_s": sum(row.end_time_s for row in successes) / len(successes) if successes else None,
        "mean_open_time_s": sum(open_times) / len(open_times) if open_times else None,
        "mean_first_box_contact_spread_s": sum(spreads) / len(spreads) if spreads else None,
        "mean_box_max_abs_yaw_deg": sum(yaws) / num,
        "max_box_max_abs_yaw_deg": max(yaws),
        "mean_box_dx_m": sum(row.box_dx_m for row in rows) / num,
        "falls": sum(row.falls for row in rows),
        "termination_reasons": reasons,
    }


def main(argv: list[str] | None = None) -> None:
    """学習した方策を評価し、試行ごとの結果と集計を書く。失敗の種類ごとに非ゼロの終了コードで終わる。"""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    output_dir: Path = args.output_dir
    if output_dir.exists():
        logger.error("出力先が既にある（上書きしない）: %s", output_dir)
        sys.exit(EXIT_OUTPUT_EXISTS)
    try:
        settings = load_team_run_settings_copy(args.run_dir)
    except (FileNotFoundError, ConfigValidationError) as error:
        logger.error("入力が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    # 学習と同じシードで評価すると、学習中に見た開始位置の並びで評価してしまうため止める
    if args.seed == settings.training.seed:
        logger.error("入力が不正: 評価のシード %d が学習のシードと同じ", args.seed)
        sys.exit(EXIT_INVALID_INPUT)
    try:
        variant = load_variant(args.run_dir)
        scene = load_corridor_scene_settings(Path(settings.scene_config))
        gait_run_dir = Path(settings.gait_policy.run_dir)
        gait = load_gait_settings(gait_run_dir / RUN_CONFIG_FILENAME).gait
        communication = dataclasses.replace(settings.task.communication, drop_probability=args.drop_probability)
        episode = dataclasses.replace(settings.task.episode, num_envs=args.num_envs)
        task = dataclasses.replace(settings.task, episode=episode, communication=communication)
        commit_sha = current_commit_sha()
    except (FileNotFoundError, ConfigValidationError, InvalidRunRecordError, InvalidConfigError,
            GitCommitUnavailableError) as error:  # fmt: skip
        # InvalidConfigError は、--drop-probability を設定の値の範囲で検証したときの誤り
        logger.error("入力が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)

    GENESIS_INITIALIZERS[args.device](args.seed)
    try:
        simulation = GenesisCorridorSimulation.build(gait, scene, CorridorSceneRequest(args.num_envs, camera=None))
        controllers = tuple(
            load_gait_controller(gait_run_dir, settings.gait_policy.checkpoint, args.num_envs, gs.device)
            for _ in simulation.robot_names
        )
        policies = tuple(
            load_team_policy(args.run_dir, args.iteration, name, gs.device) for name in simulation.robot_names
        )
        setup = TeamMarlEnvSetup(task, gait, scene.scene, make_seeded_generator(args.seed))
        env = Go2TeamMarlEnv(setup, simulation, controllers)
        rows = evaluate_team_policies(env, policies)
    except (InvalidPushConfigError, TeamSizeError, CheckpointNotFoundError, CheckpointLoadError) as error:
        logger.error("入力が不正: %s", error)
        sys.exit(EXIT_INVALID_INPUT)
    except SimulationSetupError as error:
        logger.error("%s", error)
        sys.exit(EXIT_SIMULATION_SETUP_FAILED)
    except EvaluationDidNotFinishError as error:
        logger.error("%s", error)
        sys.exit(EXIT_EVALUATION_DID_NOT_FINISH)

    output_dir.mkdir(parents=True)
    with (output_dir / TRIALS_FILENAME).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[field.name for field in dataclasses.fields(TeamMarlTrialRow)])
        writer.writeheader()
        writer.writerows(dataclasses.asdict(row) for row in rows)
    summary = {
        "run_dir": str(args.run_dir), "iteration": args.iteration, "variant": variant.name, "seed": args.seed,
        "num_envs": args.num_envs, "communication_delay_s": communication.delay_s,
        "drop_probability": communication.drop_probability, "device": args.device, "commit": commit_sha,
    } | _summary(rows)  # fmt: skip
    (output_dir / SUMMARY_FILENAME).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("評価が終わった: %s", json.dumps(_summary(rows), ensure_ascii=False))


if __name__ == "__main__":
    main()
