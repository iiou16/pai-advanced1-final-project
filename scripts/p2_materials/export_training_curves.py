"""高レベル方策の学習ログ（TensorBoard）から、学習曲線を CSV と PNG に書き出す（レポート・提出動画の素材）。

全スカラーを 1 行 1 値の CSV（tag, step, value）にし、主な指標（平均報酬・エピソード長・終了理由の割合・
報酬の内訳）を 1 枚の図にまとめる。step は反復番号。ただしタグが /time で終わるものは学習開始からの経過秒。

実行: uv run python scripts/p2_materials/export_training_curves.py <学習の出力先> <書き出し先のディレクトリ>
例:   uv run python scripts/p2_materials/export_training_curves.py outputs/push/push-stage1 outputs/materials/training
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  描画先を画面なしにしてから読み込む
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# 学習ログのスカラーを、イベントファイルにあるだけすべて読む（0 は件数の上限なし）
ALL_SCALARS = {"scalars": 0}
# 図に描く指標と、各パネルの見出し
PANELS = (
    ("平均報酬", ("Train/mean_reward",)),
    ("平均エピソード長 [ステップ]", ("Train/mean_episode_length",)),
    (
        "終了理由の割合",
        (
            "Episode/termination_success",
            "Episode/termination_time_out",
            "Episode/termination_fall",
            "Episode/termination_out_of_bounds",
            "Episode/termination_sim_diverged",
        ),
    ),
    (
        "報酬の内訳（1 秒あたり）",
        (
            "Episode/rew_box_progress",
            "Episode/rew_success",
            "Episode/rew_intended_face_contact",
            "Episode/rew_box_rotation",
            "Episode/rew_control_effort",
        ),
    ),
)
FIGURE_SIZE_IN = (12, 8)
FIGURE_DPI = 150
# 図の日本語を描くフォント（Ubuntu の fonts-noto-cjk に含まれる）
JAPANESE_FONT_FAMILY = "Noto Sans CJK JP"
# RSL-RL が経過秒を step にして記録するタグの接尾辞
ELAPSED_TIME_TAG_SUFFIX = "/time"


class MissingTrainingLogError(FileNotFoundError):
    """学習の出力先に TensorBoard のイベントファイルが無いことを表す。"""


def load_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """学習の出力先のイベントファイルから、タグ → (反復, 値) の並びを読む。

    Raises:
        MissingTrainingLogError: イベントファイルが無い。
    """
    if not any(run_dir.glob("events.out.tfevents.*")):
        raise MissingTrainingLogError(f"TensorBoard のイベントファイルが無い: {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance=ALL_SCALARS)
    accumulator.Reload()
    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


def write_csv(scalars: dict[str, list[tuple[int, float]]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["tag", "step", "value"])
        for tag, points in sorted(scalars.items()):
            writer.writerows((tag, step, value) for step, value in points)


def write_figure(scalars: dict[str, list[tuple[int, float]]], path: Path, title: str) -> None:
    plt.rcParams["font.family"] = JAPANESE_FONT_FAMILY
    figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_IN)
    for axis, (panel_title, tags) in zip(axes.flat, PANELS, strict=True):
        for tag in tags:
            points = scalars[tag]
            axis.plot([step for step, _ in points], [value for _, value in points], label=tag.split("/")[-1])
        axis.set_title(panel_title)
        axis.set_xlabel("反復")
        axis.grid(alpha=0.3)
        if len(tags) > 1:
            axis.legend(fontsize="small")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(path, dpi=FIGURE_DPI)


def main() -> None:
    run_dir, output_dir = Path(sys.argv[1]), Path(sys.argv[2])
    scalars = load_scalars(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(scalars, output_dir / f"{run_dir.name}_scalars.csv")
    write_figure(scalars, output_dir / f"{run_dir.name}_curves.png", f"学習曲線: {run_dir.name}")
    last_iteration = max(
        step for tag, points in scalars.items() if not tag.endswith(ELAPSED_TIME_TAG_SUFFIX) for step, _ in points
    )
    print(f"{run_dir.name}: 反復 {last_iteration} まで書き出した → {output_dir}")


if __name__ == "__main__":
    main()
