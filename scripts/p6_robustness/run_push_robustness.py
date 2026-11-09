"""1 台の箱押しの頑健性の評価（configs/push_robustness.yaml の全条件）を、方式とシードごとに go2-evaluate-push で流す。

条件 × 方式（規則ベース・学習した方策）× シードの組み合わせを 1 つずつ別のプロセスで評価する（Genesis は 1 つの
プロセスで 1 回だけ初期化するため）。結果は outputs/push_eval/e8-<方式>-<条件>-seed<シード>/ に保存する。
実行名の接頭辞 e8 は、この頑健性の評価の実行を、箱押しの評価の実行（README の e1-…・e2-…）と見分けるためのもの。
summary.json がある組み合わせは評価済みとして飛ばすので、途中で止めても続きから流せる。飛ばす前に、記録に残った
条件の値・シード・方式・チェックポイント・試行の数が今の設定と一致するかを確かめ、違えば止める。保存先があるのに
summary.json が無い組み合わせ（途中で止まったもの）があれば、上書きせずに止まる。

実行例（リポジトリのルートで、source scripts/setup/env.sh の後に）:
    uv run --frozen python scripts/p6_robustness/run_push_robustness.py
    uv run --frozen python scripts/p6_robustness/run_push_robustness.py --condition nominal --seed 2001
"""

import argparse
import dataclasses
import json
import subprocess
import sys
from pathlib import Path

from cooperative_go2.entities.push_robustness import PushRobustnessCondition, UnknownRobustnessConditionError
from cooperative_go2.frameworks.evaluate_push import ALGORITHM_POLICY, ALGORITHM_RULE_BASED, OUTPUT_ROOT
from cooperative_go2.interface_adapters.config_loader import load_push_robustness_suite
from cooperative_go2.interface_adapters.push_evaluation_store import SUMMARY_FILENAME

ROBUSTNESS_CONFIG = Path("configs/push_robustness.yaml")
# 評価する学習した方策（目標の方向を ±60° に広げて学習した第 2 段階）と、その学習時の配置の設定
EVALUATION_CONFIG = Path("configs/push_single_wide.yaml")
POLICY_RUN_DIR = Path("outputs/push/push-stage2-wide60")
POLICY_CHECKPOINT = 200
# 箱押しの評価（README の「箱押しの高レベル方策の学習と評価」）と同じシードと試行の数。
# 同じにすると、nominal はその評価と試行ごとに同じ初期条件になる
SEEDS = (2001, 2002, 2003)
NUM_ENVS = 64
DEVICE = "cuda"
ALGORITHMS = (ALGORITHM_RULE_BASED, ALGORITHM_POLICY)
RUN_NAME_PREFIX = "e8"


class StaleEvaluationError(RuntimeError):
    """評価済みの記録の条件が、今の設定と一致しないことを表す（古い設定で流した結果を使い回さないために止める）。"""


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
    if algorithm == ALGORITHM_POLICY:
        command += ["--policy-run-dir", str(POLICY_RUN_DIR), "--checkpoint", str(POLICY_CHECKPOINT)]
    return command


def _expected_checkpoint(algorithm: str) -> str | None:
    # 評価の記録（summary.json）の checkpoint と同じ書き方。規則ベースは学習していないので None
    if algorithm == ALGORITHM_RULE_BASED:
        return None
    return f"{POLICY_RUN_DIR}/model_{POLICY_CHECKPOINT}.pt"


def require_matching_record(algorithm: str, condition: PushRobustnessCondition, seed: int) -> None:
    """評価済みの記録の条件が、今の設定（条件の値・シード・方式・チェックポイント・試行の数）と一致することを確かめる。

    前提: 保存先に summary.json があること。集計（summarize_push_robustness.py）も、集計する前にこれで確かめる。

    Raises:
        StaleEvaluationError: 記録の条件が今の設定と一致しない。
    """
    name = run_name(algorithm, condition.name, seed)
    path = OUTPUT_ROOT / name / SUMMARY_FILENAME
    recorded = json.loads(path.read_text(encoding="utf-8"))["conditions"]
    expected = {
        "seed": seed,
        "algorithm": algorithm,
        "checkpoint": _expected_checkpoint(algorithm),
        "num_envs": NUM_ENVS,
        "robustness_condition": dataclasses.asdict(condition),
    }
    mismatched = sorted(key for key, value in expected.items() if recorded[key] != value)
    if mismatched:
        raise StaleEvaluationError(
            f"{path} の {mismatched} が今の設定と一致しない。古い設定で流した結果なので、確かめて削除してから再実行する"
        )


def _is_finished(algorithm: str, condition: PushRobustnessCondition, seed: int) -> bool:
    output_dir = OUTPUT_ROOT / run_name(algorithm, condition.name, seed)
    if (output_dir / SUMMARY_FILENAME).exists():
        require_matching_record(algorithm, condition, seed)
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
        StaleEvaluationError: 評価済みの記録の条件が今の設定と一致しない。
    """
    args = _parse_args(argv)
    suite = load_push_robustness_suite(ROBUSTNESS_CONFIG)
    names = args.condition or [condition.name for condition in suite.conditions]
    conditions = [suite.get_condition(name) for name in names]
    seeds = args.seed or list(SEEDS)
    for condition in conditions:
        for algorithm in ALGORITHMS:
            for seed in seeds:
                name = run_name(algorithm, condition.name, seed)
                if _is_finished(algorithm, condition, seed):
                    print(f"評価済み: {name}", flush=True)
                    continue
                print(f"評価する: {name}", flush=True)
                completed = subprocess.run(_command(algorithm, condition.name, seed), check=False)
                if completed.returncode != 0:
                    raise EvaluationFailedError(f"{name} の評価が終了コード {completed.returncode} で失敗した")


def main(argv: list[str] | None = None) -> None:
    """全条件の評価を流す。失敗したら終了コード 1 で終わる。"""
    try:
        run(argv)
    except (UnfinishedEvaluationError, EvaluationFailedError, UnknownRobustnessConditionError,
            StaleEvaluationError) as error:  # fmt: skip
        print(error, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
