"""1 台の箱押しの頑健性の評価（configs/push_robustness.yaml の全条件）を、方式とシードごとに go2-evaluate-push で流す。

条件 × 方式（規則ベース・学習した方策）× シードの組み合わせを 1 つずつ別のプロセスで評価する（シーンの箱の質量・
摩擦は組み合わせごとに作り直すため）。結果は outputs/push_eval/e8-<方式>-<条件>-seed<シード>/ に保存する。
summary.json がある組み合わせは評価済みとして飛ばすので、途中で止めても続きから流せる。保存先があるのに
summary.json が無い組み合わせ（途中で止まったもの）があれば、上書きせずに止まる。

実行例（リポジトリのルートで、source scripts/setup/env.sh の後に）:
    uv run --frozen python scripts/p6_robustness/run_push_robustness.py
    uv run --frozen python scripts/p6_robustness/run_push_robustness.py --condition nominal --seed 2001
"""

import argparse
import subprocess
import sys
from pathlib import Path

from cooperative_go2.entities.push_robustness import UnknownRobustnessConditionError
from cooperative_go2.interface_adapters.config_loader import load_push_robustness_suite

ROBUSTNESS_CONFIG = Path("configs/push_robustness.yaml")
# 学習した方策（第 2 段階）と、その評価の設定。学習時の条件の評価（E2）と同じ
EVALUATION_CONFIG = Path("configs/push_single_wide.yaml")
POLICY_RUN_DIR = Path("outputs/push/push-stage2-wide60")
POLICY_CHECKPOINT = 200
# 学習時の条件の評価（E1・E2）と同じ評価のシードと試行の数。nominal は E1・E2 の ±60° の評価と同じ初期条件になる
SEEDS = (2001, 2002, 2003)
NUM_ENVS = 64
DEVICE = "cuda"
ALGORITHMS = ("rule_based", "policy")
OUTPUT_ROOT = Path("outputs/push_eval")
SUMMARY_FILENAME = "summary.json"
RUN_NAME_PREFIX = "e8"


class UnfinishedEvaluationError(RuntimeError):
    """前に途中で止まった評価の保存先が残っていることを表す。"""


class EvaluationFailedError(RuntimeError):
    """go2-evaluate-push が非ゼロの終了コードで終わったことを表す。"""


def run_name(algorithm: str, condition: str, seed: int) -> str:
    """評価の実行の名前（保存先のディレクトリ名）を返す。集計（summarize_push_robustness.py）と共通。"""
    return f"{RUN_NAME_PREFIX}-{algorithm}-{condition}-seed{seed}"


def _command(algorithm: str, condition: str, seed: int) -> list[str]:
    command = [
        "uv", "run", "--frozen", "go2-evaluate-push", "--config", str(EVALUATION_CONFIG), "--algorithm", algorithm,
        "--robustness-config", str(ROBUSTNESS_CONFIG), "--condition", condition,
        "--num-envs", str(NUM_ENVS), "--seed", str(seed), "--device", DEVICE,
        "--run-name", run_name(algorithm, condition, seed),
    ]  # fmt: skip
    if algorithm == "policy":
        command += ["--policy-run-dir", str(POLICY_RUN_DIR), "--checkpoint", str(POLICY_CHECKPOINT)]
    return command


def _is_finished(name: str) -> bool:
    output_dir = OUTPUT_ROOT / name
    if (output_dir / SUMMARY_FILENAME).exists():
        return True
    if output_dir.exists():
        raise UnfinishedEvaluationError(f"途中で止まった評価の保存先がある。確かめて削除してから再実行する: {output_dir}")
    return False


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1 台の箱押しの頑健性の評価を全条件で流す")
    parser.add_argument("--condition", action="append", help="流す条件の名前（繰り返し指定できる）。省略すると全条件")
    parser.add_argument("--seed", type=int, action="append", choices=SEEDS, help="流すシード。省略すると全シード")
    return parser.parse_args(argv)


def run(argv: list[str] | None = None) -> None:
    """評価していない組み合わせを順に評価する。

    Raises:
        UnfinishedEvaluationError: 途中で止まった評価の保存先がある。
        EvaluationFailedError: 評価が失敗した。
        UnknownRobustnessConditionError: 指定した条件が設定に無い。
    """
    args = _parse_args(argv)
    suite = load_push_robustness_suite(ROBUSTNESS_CONFIG)
    names = args.condition or [condition.name for condition in suite.conditions]
    conditions = [suite.get_condition(name).name for name in names]
    seeds = args.seed or list(SEEDS)
    for condition in conditions:
        for algorithm in ALGORITHMS:
            for seed in seeds:
                name = run_name(algorithm, condition, seed)
                if _is_finished(name):
                    print(f"評価済み: {name}", flush=True)
                    continue
                print(f"評価する: {name}", flush=True)
                completed = subprocess.run(_command(algorithm, condition, seed), check=False)
                if completed.returncode != 0:
                    raise EvaluationFailedError(f"{name} の評価が終了コード {completed.returncode} で失敗した")


def main(argv: list[str] | None = None) -> None:
    """全条件の評価を流す。失敗したら終了コード 1 で終わる。"""
    try:
        run(argv)
    except (UnfinishedEvaluationError, EvaluationFailedError, UnknownRobustnessConditionError) as error:
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
