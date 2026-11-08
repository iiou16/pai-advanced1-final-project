"""1 台の箱押しの頑健性の評価（run_push_robustness.py の出力）を、条件と方式ごとに 3 シードを合わせて表にする。

各評価の trials.csv（試行ごとの結果）を読み、条件 × 方式ごとに全シードの試行を合わせた指標と、シードごとの成功率の
範囲を出す。結果は Markdown の表（robustness.md）と CSV（robustness.csv）に書く。評価の抜けがあれば止まる。

実行例（リポジトリのルートで）:
    uv run --frozen python scripts/p6_robustness/summarize_push_robustness.py --output-dir outputs/push_eval/e8-summary
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from run_push_robustness import ALGORITHMS, OUTPUT_ROOT, ROBUSTNESS_CONFIG, SEEDS, SUMMARY_FILENAME, run_name

from cooperative_go2.interface_adapters.config_loader import load_push_robustness_suite

TRIALS_FILENAME = "trials.csv"
TRUE_TEXT = "True"
FALSE_TEXT = "False"
TIME_OUT_REASON = "TIME_OUT"
ALGORITHM_LABELS = {"rule_based": "規則ベース", "policy": "学習した方策"}
PERCENT = 100.0


class SummaryInputError(ValueError):
    """集計する評価が無い、または中身が想定と違うことを表す。"""


@dataclass(frozen=True)
class ConditionSummary:
    """1 つの条件 × 方式の、全シードの試行を合わせた指標。

    Attributes:
        condition: 条件の名前。
        algorithm: 方式（rule_based / policy）。
        trials: 試行の数（全シードの合計）。
        success_rate: 成功の割合（0〜1）。
        min_seed_success_rate: シードごとの成功率の最小。
        max_seed_success_rate: シードごとの成功率の最大。
        mean_success_time_s: 成功した試行の、成功までの時間の平均 [s]。成功が無ければ None。
        mean_final_goal_distance_m: 最終の目標距離の平均 [m]。
        mean_rotation_deg: 箱の回転の平均 [deg]。
        fall_rate: 転倒した試行の割合（0〜1）。
        time_out_rate: 時間切れで終わった試行の割合（0〜1）。
        commits: 評価したコミット（重複を除く）。
    """

    condition: str
    algorithm: str
    trials: int
    success_rate: float
    min_seed_success_rate: float
    max_seed_success_rate: float
    mean_success_time_s: float | None
    mean_final_goal_distance_m: float
    mean_rotation_deg: float
    fall_rate: float
    time_out_rate: float
    commits: tuple[str, ...]


def _parse_bool(text: str, path: Path) -> bool:
    if text == TRUE_TEXT:
        return True
    if text == FALSE_TEXT:
        return False
    raise SummaryInputError(f"{path}: success の値が {TRUE_TEXT} / {FALSE_TEXT} ではない: {text!r}")


def _read_trials(name: str) -> list[dict[str, str]]:
    output_dir = OUTPUT_ROOT / name
    if not (output_dir / SUMMARY_FILENAME).exists():
        raise SummaryInputError(f"評価が終わっていない（{SUMMARY_FILENAME} が無い）: {output_dir}")
    with (output_dir / TRIALS_FILENAME).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise SummaryInputError(f"試行が無い: {output_dir / TRIALS_FILENAME}")
    return rows


def _summarize(condition: str, algorithm: str) -> ConditionSummary:
    rows: list[dict[str, str]] = []
    seed_success_rates: list[float] = []
    for seed in SEEDS:
        name = run_name(algorithm, condition, seed)
        seed_rows = _read_trials(name)
        path = OUTPUT_ROOT / name / TRIALS_FILENAME
        successes = [_parse_bool(row["success"], path) for row in seed_rows]
        seed_success_rates.append(sum(successes) / len(successes))
        rows += seed_rows
    success_times = [float(row["elapsed_sim_s"]) for row in rows if row["success"] == TRUE_TEXT]
    trials = len(rows)
    return ConditionSummary(
        condition=condition,
        algorithm=algorithm,
        trials=trials,
        success_rate=len(success_times) / trials,
        min_seed_success_rate=min(seed_success_rates),
        max_seed_success_rate=max(seed_success_rates),
        mean_success_time_s=sum(success_times) / len(success_times) if success_times else None,
        mean_final_goal_distance_m=sum(float(row["final_goal_distance_m"]) for row in rows) / trials,
        mean_rotation_deg=sum(float(row["payload_rotation_deg"]) for row in rows) / trials,
        fall_rate=sum(int(row["falls"]) > 0 for row in rows) / trials,
        time_out_rate=sum(row["termination_reason"] == TIME_OUT_REASON for row in rows) / trials,
        commits=tuple(sorted({row["commit_sha"] for row in rows})),
    )


def _percent(rate: float) -> str:
    return f"{rate * PERCENT:.1f} %"


def _markdown(summaries: list[ConditionSummary]) -> str:
    header = (
        "| 条件 | 方式 | 成功率（全試行） | 成功率（シードの範囲） | 成功までの時間 [s] | 最終の目標距離 [m] "
        "| 箱の回転 [deg] | 転倒率 | 時間切れ |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = [
        f"| {summary.condition} | {ALGORITHM_LABELS[summary.algorithm]} | {_percent(summary.success_rate)} "
        f"| {_percent(summary.min_seed_success_rate)}〜{_percent(summary.max_seed_success_rate)} "
        f"| {'—' if summary.mean_success_time_s is None else f'{summary.mean_success_time_s:.2f}'} "
        f"| {summary.mean_final_goal_distance_m:.3f} | {summary.mean_rotation_deg:.1f} "
        f"| {_percent(summary.fall_rate)} | {_percent(summary.time_out_rate)} |"
        for summary in summaries
    ]
    return header + "\n".join(lines) + "\n"


def _write_csv(path: Path, summaries: list[ConditionSummary]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(summaries[0])))
        writer.writeheader()
        for summary in summaries:
            row = asdict(summary)
            row["commits"] = json.dumps(row["commits"])
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    """全条件 × 方式を集計して書き出す。評価の抜けがあれば終了コード 1 で終わる。"""
    parser = argparse.ArgumentParser(description="1 台の箱押しの頑健性の評価を表にする")
    parser.add_argument("--output-dir", type=Path, required=True, help="表の出力先。既にあれば上書きせず止まる")
    args = parser.parse_args(argv)
    suite = load_push_robustness_suite(ROBUSTNESS_CONFIG)
    try:
        summaries = [
            _summarize(condition.name, algorithm) for condition in suite.conditions for algorithm in ALGORITHMS
        ]
    except SummaryInputError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
    if args.output_dir.exists():
        print(f"出力先が既にある: {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "robustness.md").write_text(_markdown(summaries), encoding="utf-8")
    _write_csv(args.output_dir / "robustness.csv", summaries)
    print(_markdown(summaries))


if __name__ == "__main__":
    main()
