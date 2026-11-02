"""歩行方策の評価結果（JSON・CSV・評価動画・シナリオの写し）を 1 つのディレクトリに保存する。"""

import csv
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

from cooperative_go2.entities.gait_evaluation import ScenarioSummary

SUMMARY_FILENAME = "summary.json"
SEGMENTS_FILENAME = "segments.csv"
# 評価に使ったシナリオのファイルを、結果と同じディレクトリに複製するときの名前
SCENARIOS_COPY_FILENAME = "gait_eval.yaml"
SEGMENT_CSV_COLUMNS = (
    "scenario", "segment_index", "vx_mps", "vy_mps", "yaw_rate_radps", "duration_s", "num_envs",
    "num_tracking_samples", "mean_lin_vel_error_mps", "mean_yaw_rate_error_radps", "num_falls", "fall_rate",
    "scenario_fall_rate",
)  # fmt: skip


class EvaluationOutputExistsError(RuntimeError):
    """評価結果の保存先が既にあり、上書きすると過去の結果を失うことを表す。"""


@dataclass(frozen=True)
class GaitEvaluationRecord:
    """保存する評価 1 回分の条件と結果。

    Attributes:
        run_dir: 評価した学習の出力先。
        checkpoint_iteration: 評価したチェックポイントの反復番号。
        scenarios_file: 評価に使ったシナリオのファイル。
        seed: 評価用の乱数シード。
        num_envs: 並列に走らせた環境の数。
        settle_time_s: 追従誤差の集計から除いた過渡の時間 [s]。各区間の始めと、転倒した環境の立ち直りに使った。
        scenarios: シナリオごとの結果。
    """

    run_dir: str
    checkpoint_iteration: int
    scenarios_file: str
    seed: int
    num_envs: int
    settle_time_s: float
    scenarios: tuple[ScenarioSummary, ...]


@dataclass(frozen=True)
class EvaluationRunKey:
    """評価結果を左右する条件の組。保存先の名前に使い、条件が違う評価の結果を別々に残す。

    Attributes:
        checkpoint_iteration: 評価するチェックポイントの反復番号。
        scenarios_name: シナリオのファイルの名前（拡張子を除く）。
        num_envs: 並列に走らせる環境の数。
        seed: 評価用の乱数シード。
    """

    checkpoint_iteration: int
    scenarios_name: str
    num_envs: int
    seed: int

    def output_dir(self, run_dir: Path) -> Path:
        """学習の出力先の下に、この条件の評価結果の保存先のパスを作る。ディレクトリはまだ作らない。"""
        name = f"model_{self.checkpoint_iteration}_{self.scenarios_name}_envs_{self.num_envs}_seed_{self.seed}"
        return run_dir / "eval" / name


class FileGaitEvaluationStore:
    """評価結果の保存先を作り、シナリオのファイルの写し・評価動画・集計結果を置く。"""

    def __init__(self, output_dir: Path, scenarios_source: Path) -> None:
        """保存先と、複製するシナリオのファイルを受け取る。ディレクトリは initialize() まで作らない。

        Args:
            output_dir: 保存先のディレクトリ。まだ存在してはならない。
            scenarios_source: 評価に使うシナリオのファイル。読み込み・検証済みであること。
        """
        self._output_dir = output_dir
        self._scenarios_source = scenarios_source

    def initialize(self) -> None:
        """保存先を作り、シナリオのファイルを複製する。

        Raises:
            EvaluationOutputExistsError: 保存先が既にある。過去の結果を上書きしないため止める。
        """
        # 保存先を作る前に読む。読めずに止まったとき、空の保存先が残って再実行を塞がないようにするため
        scenarios_content = self._scenarios_source.read_bytes()
        try:
            self._output_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise EvaluationOutputExistsError(f"評価結果の保存先が既にある（上書きしない）: {self._output_dir}") from error
        (self._output_dir / SCENARIOS_COPY_FILENAME).write_bytes(scenarios_content)

    def video_path(self, scenario_name: str) -> Path:
        """シナリオの評価動画（mp4）の保存先を返す。"""
        return self._output_dir / f"{scenario_name}.mp4"

    def save(self, record: GaitEvaluationRecord) -> None:
        """区間ごとの結果を segments.csv に、評価の条件と結果を summary.json に書き出す。

        summary.json を最後に書くので、summary.json があれば保存が最後まで終わっている。
        追従誤差を集計できなかった区間の平均は、JSON では null、CSV では空欄になる。
        """
        with (self._output_dir / SEGMENTS_FILENAME).open("w", encoding="utf-8", newline="") as segments_file:
            writer = csv.writer(segments_file)
            writer.writerow(SEGMENT_CSV_COLUMNS)
            writer.writerows(_segment_rows(record.scenarios))
        (self._output_dir / SUMMARY_FILENAME).write_text(
            json.dumps(dataclasses.asdict(record), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )


def _segment_rows(scenarios: tuple[ScenarioSummary, ...]) -> list[list[object]]:
    return [
        [
            scenario.name, index, *segment.command.as_tuple(), segment.duration_s, segment.num_envs,
            segment.num_tracking_samples, segment.mean_lin_vel_error_mps, segment.mean_yaw_rate_error_radps,
            segment.num_falls, segment.fall_rate, scenario.fall_rate,
        ]  # fmt: skip
        for scenario in scenarios
        for index, segment in enumerate(scenario.segments)
    ]
