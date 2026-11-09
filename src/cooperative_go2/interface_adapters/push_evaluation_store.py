"""箱押しの評価の記録（試行ごとに 1 行の trials.csv と、条件と集計の summary.json）を保存する。

trials.csv には、試行の条件（コミット・シーン・シード・方式・チェックポイント・箱の質量と摩擦・開始時と目標の位置）と、
結果（成功・終了理由・所要時間・箱の移動量と回転・衝突・転倒・物理ステップ数・実時間・目標方向の移動量・最終目標距離）を書く。
summary.json を最後に書くので、summary.json があれば保存は最後まで終わっている。
"""

import csv
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cooperative_go2.entities.push_robustness import PushRobustnessCondition
from cooperative_go2.entities.push_trial import PushEpisodeResult, PushTrialSummary

TRIALS_FILENAME = "trials.csv"
SUMMARY_FILENAME = "summary.json"
TRIAL_COLUMNS = (
    "run_id",
    "commit_sha",
    "scene_id",
    "seed",
    "algorithm",
    "checkpoint",
    "mass",
    "friction",
    "start_poses",
    "goal_pose",
    "success",
    "termination_reason",
    "elapsed_sim_s",
    "payload_displacement_m",
    "payload_rotation_deg",
    "collisions",
    "falls",
    "sim_steps",
    "wall_time_s",
    "payload_progress_m",
    "final_goal_distance_m",
)


class PushEvaluationOutputExistsError(RuntimeError):
    """評価の保存先が既にあることを表す。前の結果を上書きしないために止める。"""


@dataclass(frozen=True)
class PushEvaluationConditions:
    """評価の条件。すべての試行に共通で、trials.csv の各行と summary.json に書く。

    Attributes:
        run_id: 評価の実行の名前。保存先のディレクトリ名にも使う。
        commit_sha: 評価したときのリポジトリのコミット。
        scene_id: シーンの名前（例: S1_push）。
        seed: 評価に使った乱数のシード。学習のシードとは別にする。
        algorithm: 速度指令を決めた方式（例: rule_based）。
        checkpoint: 学習済みの方策のチェックポイント。規則ベースのように学習していない方式では None。
        mass_kg: 箱の質量 [kg]。
        friction: 箱と床の間の実際の摩擦係数（接触する 2 面の大きい方）。
        high_level_period_s: 高レベル方策の周期 [s]。
        gait_steps_per_high_level_step: 高レベル方策の 1 ステップで物理を進める回数。
        num_envs: 並列環境の数（= 試行の数）。
        robustness_condition: 頑健性の評価で変えた条件（質量・摩擦・配置・速度指令の遅れ）。評価の設定のままなら None。
            summary.json にだけ書く（試行ごとの配置と質量・摩擦は trials.csv の各行にもある）。
    """

    run_id: str
    commit_sha: str
    scene_id: str
    seed: int
    algorithm: str
    checkpoint: str | None
    mass_kg: float
    friction: float
    high_level_period_s: float
    gait_steps_per_high_level_step: int
    num_envs: int
    robustness_condition: PushRobustnessCondition | None


class FilePushEvaluationStore:
    """評価の記録を output_dir に保存する。"""

    def __init__(self, output_dir: Path) -> None:
        """保存先のディレクトリを受け取る。作るのは initialize() を呼んだとき。"""
        self._output_dir = output_dir

    def initialize(self) -> None:
        """保存先のディレクトリを作る。

        Raises:
            PushEvaluationOutputExistsError: 保存先が既にある。
        """
        if self._output_dir.exists():
            raise PushEvaluationOutputExistsError(f"評価の保存先が既にある: {self._output_dir}")
        self._output_dir.mkdir(parents=True)

    def save(
        self, conditions: PushEvaluationConditions, results: Sequence[PushEpisodeResult], summary: PushTrialSummary
    ) -> None:
        """試行ごとの結果を trials.csv に、条件と集計を summary.json に書く。前提: initialize() 済み。"""
        with (self._output_dir / TRIALS_FILENAME).open("w", encoding="utf-8", newline="") as trials_file:
            writer = csv.writer(trials_file)
            writer.writerow(TRIAL_COLUMNS)
            writer.writerows(_trial_row(conditions, result) for result in results)
        summary_json = {"conditions": dataclasses.asdict(conditions), "summary": dataclasses.asdict(summary)}
        (self._output_dir / SUMMARY_FILENAME).write_text(
            json.dumps(summary_json, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )


def _trial_row(conditions: PushEvaluationConditions, result: PushEpisodeResult) -> list[object]:
    start_poses = {"robot": dataclasses.asdict(result.robot_start), "box": dataclasses.asdict(result.box_start)}
    return [
        conditions.run_id,
        conditions.commit_sha,
        conditions.scene_id,
        conditions.seed,
        conditions.algorithm,
        "" if conditions.checkpoint is None else conditions.checkpoint,
        conditions.mass_kg,
        conditions.friction,
        json.dumps(start_poses, allow_nan=False),
        json.dumps({"x_m": result.goal_x_m, "y_m": result.goal_y_m}, allow_nan=False),
        result.success,
        result.termination_reason.name,
        result.high_level_steps * conditions.high_level_period_s,
        result.displacement_m(),
        result.rotation_deg(),
        result.collisions,
        result.falls,
        result.high_level_steps * conditions.gait_steps_per_high_level_step,
        result.wall_time_s,
        result.progress_toward_goal_m(),
        result.final_goal_distance_m(),
    ]
