"""ROS 2 を経由した通路の開通の評価（outputs/ros2_team_eval/<run>/trials.csv）を、方式ごとに合わせて並べる。

同じシード・同じ試行の数で流した評価を、方式ごとに複数の実行（1 回目と流し直し）を合わせて集計し、Markdown の表と
JSON を書く。ROS 2 の実行系は物理を方策の応答を待たずに進めるので、同じ開始位置でも実行ごとに結果が変わる。
そのため試行ごとではなく、試行の割合で比べる。

実行例:
    uv run python scripts/p5_marl_check/compare_ros2_team_runs.py --output outputs/p5_marl_check/comparison_ros2.json \\
        --method "E5 規則ベースの協調=e5-seed6001,e5-seed6001-rerun" \\
        --method "MAPPO・別々=learned-mappo-separate-seed6001,learned-mappo-separate-seed6001-rerun"
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

EVAL_ROOT = Path("outputs/ros2_team_eval")
TRIALS_FILENAME = "trials.csv"
SUMMARY_FILENAME = "summary.json"
TRUE_TEXT = "True"
FALSE_TEXT = "False"


class ComparisonInputError(ValueError):
    """比べる評価の指定や中身が不正であることを表す。"""


@dataclass(frozen=True)
class MethodSummary:
    """1 つの方式の、複数の実行を合わせた集計。

    Attributes:
        method: 方式の名前。
        runs: 合わせた実行の名前。
        commits: 実行したコミット（実行ごと）。
        trials: 試行の数。
        successes: 成功（シミュレーションの真値の判定）の数。
        task_successes: タスク管理の結果が成功だった数。
        corridor_opened: 通路が開いた試行の数。
        mean_open_time_s: 通路が開いた試行の、開通までの平均 [s]。開かなければ None。
        mean_success_time_s: 成功した試行の、成功までの平均 [s]。成功が無ければ None。
        mean_box_dx_m: 箱の +x 方向の移動量の平均 [m]。
        mean_box_max_abs_yaw_deg: 試行ごとの箱の回転の最大の平均 [deg]。
        max_box_max_abs_yaw_deg: 全試行の中の箱の回転の最大 [deg]。
        mean_push_start_spread_s: 2 台が押し始めた時刻の差の平均 [s]。押し始めを記録しない方式（学習した方策）は None。
            1 台では差が無いので 0.0 になる。
        falls: 転倒したロボットの数の合計。
    """

    method: str
    runs: tuple[str, ...]
    commits: tuple[str, ...]
    trials: int
    successes: int
    task_successes: int
    corridor_opened: int
    mean_open_time_s: float | None
    mean_success_time_s: float | None
    mean_box_dx_m: float
    mean_box_max_abs_yaw_deg: float
    max_box_max_abs_yaw_deg: float
    mean_push_start_spread_s: float | None
    falls: int


def _mean_or_none(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _read_rows(run: str) -> list[dict[str, str]]:
    path = EVAL_ROOT / run / TRIALS_FILENAME
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ComparisonInputError(f"試行が無い: {path}")
    return rows


def _flag(row: dict[str, str], column: str) -> bool:
    # 真偽値の列は "True" か "False" のどちらか。それ以外を False と数えると、書式が変わったときに成功の数が黙って 0 になる
    value = row[column]
    if value not in (TRUE_TEXT, FALSE_TEXT):
        raise ComparisonInputError(f"{column} の値が True でも False でもない: {value!r}")
    return value == TRUE_TEXT


def _commit(run: str) -> str:
    summary = json.loads((EVAL_ROOT / run / SUMMARY_FILENAME).read_text(encoding="utf-8"))
    return str(summary["conditions"]["commit_sha"])


def summarize_method(method: str, runs: tuple[str, ...]) -> MethodSummary:
    """方式 method の実行 runs の trials.csv を合わせて集計する。

    Raises:
        FileNotFoundError: 実行の trials.csv か summary.json が無い。
        ComparisonInputError: 試行が無い、または真偽値の列の値が True・False 以外。
    """
    rows = [row for run in runs for row in _read_rows(run)]
    yaws = [float(row["box_max_abs_yaw_deg"]) for row in rows]
    return MethodSummary(
        method=method,
        runs=runs,
        commits=tuple(_commit(run) for run in runs),
        trials=len(rows),
        successes=sum(_flag(row, "success") for row in rows),
        task_successes=sum(_flag(row, "task_success") for row in rows),
        corridor_opened=sum(_flag(row, "corridor_opened") for row in rows),
        mean_open_time_s=_mean_or_none([float(row["open_time_s"]) for row in rows if row["open_time_s"]]),
        mean_success_time_s=_mean_or_none([float(row["success_time_s"]) for row in rows if row["success_time_s"]]),
        mean_box_dx_m=sum(float(row["box_dx_m"]) for row in rows) / len(rows),
        mean_box_max_abs_yaw_deg=sum(yaws) / len(yaws),
        max_box_max_abs_yaw_deg=max(yaws),
        mean_push_start_spread_s=_mean_or_none(
            [float(row["push_start_spread_s"]) for row in rows if row["push_start_spread_s"]]
        ),
        falls=sum(int(row["falls"]) for row in rows),
    )


def _format(value: float | None, digits: int) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def to_markdown(summaries: list[MethodSummary]) -> str:
    """方式ごとの集計を Markdown の表にする。"""
    header = (
        "| 方式 | 成功 | 開通 | 開通まで [s] | 成功まで [s] | 移動量 [m] | 回転の最大の平均・全体 [deg] "
        "| 押し始めの差 [s] | 転倒 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
    )
    lines = [
        f"| {s.method} | {s.successes}/{s.trials} | {s.corridor_opened}/{s.trials} | {_format(s.mean_open_time_s, 1)} "
        f"| {_format(s.mean_success_time_s, 1)} | {s.mean_box_dx_m:.2f} | {s.mean_box_max_abs_yaw_deg:.0f}・"
        f"{s.max_box_max_abs_yaw_deg:.0f} | {_format(s.mean_push_start_spread_s, 2)} | {s.falls} |"
        for s in summaries
    ]
    return "\n".join([header, *lines])


def _parse_method(text: str) -> tuple[str, tuple[str, ...]]:
    name, separator, runs = text.partition("=")
    run_names = tuple(runs.split(","))
    # 空の実行の名前（"a,,b" や末尾のカンマ）は打ち間違いなので、黙って捨てずに止める
    if not separator or not name or any(not run for run in run_names):
        raise ComparisonInputError(f"--method は 名前=実行,実行 の形で書く: {text!r}")
    return name, run_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--method", action="append", required=True, help="名前=実行の名前,実行の名前（複数指定）")
    parser.add_argument("--output", type=Path, required=True, help="集計の JSON の出力先。既にあれば上書きせず止まる")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"出力先が既にある（上書きしない）: {args.output}")
    summaries = [summarize_method(*_parse_method(text)) for text in args.method]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps([asdict(summary) for summary in summaries], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(to_markdown(summaries))


if __name__ == "__main__":
    main()
