"""ROS 2 の実行系の計測値（推論時間・遅延など）の集計と書き出し。"""

import json
import statistics
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

MILLISECONDS_PER_SECOND = 1000.0
PERCENTILE_COUNT = 100
P95_INDEX = 94


class LatencyStats:
    """時間の計測値 [s] を集め、件数・平均・中央値・95 パーセンタイル・最大 [ms] にまとめる。"""

    def __init__(self) -> None:
        """空の集計を作る。"""
        self._samples_s: list[float] = []

    def add(self, duration_s: float) -> None:
        """計測値 [s] を 1 つ加える。"""
        self._samples_s.append(duration_s)

    def summary(self) -> dict[str, float | int | None]:
        """集計を [ms] で返す。計測値が無ければ件数 0 で、ほかは None。"""
        samples_ms = [sample * MILLISECONDS_PER_SECOND for sample in self._samples_s]
        if not samples_ms:
            return {"count": 0, "mean_ms": None, "median_ms": None, "p95_ms": None, "max_ms": None}
        p95 = statistics.quantiles(samples_ms, n=PERCENTILE_COUNT)[P95_INDEX] if len(samples_ms) > 1 else samples_ms[0]
        return {
            "count": len(samples_ms),
            "mean_ms": statistics.fmean(samples_ms),
            "median_ms": statistics.median(samples_ms),
            "p95_ms": p95,
            "max_ms": max(samples_ms),
        }


def write_node_metrics(path: Path, metrics: Mapping[str, object]) -> None:
    """ノードの計測値を JSON で新しいファイルに書く。既にあるファイルは上書きしない。

    Args:
        path: 書き出すファイル。親ディレクトリが無ければ作るが、その親（実行の保存先）は既にあること。
            保存先はシミュレーションノードが作るので、保存先が無いのは実行を始められなかったことを表す。
        metrics: 計測値。JSON にできる値だけを含むこと（NaN は不可）。

    Raises:
        FileNotFoundError: 実行の保存先（path の親の親）が無い。
        FileExistsError: path が既にある。前の実行の計測値を上書きしないために止める。
    """
    path.parent.mkdir(exist_ok=True)
    text = json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as file:
        file.write(text)


class RoundTripLatencyProbe:
    """状態を配信してから、その状態から作った関節目標が届くまでの実時間（往復の遅延）を測る。

    往復には、DDS の配信 2 回分と歩行方策の推論が含まれる。配信は試行のスレッドから、受信は ROS 2 の受信のスレッドから呼ぶ。
    """

    def __init__(self, wall_clock: Callable[[], float] = time.perf_counter) -> None:
        """実時間 [s] を返す関数を受け取る。"""
        self._wall_clock = wall_clock
        self._lock = threading.Lock()
        self._published_at: dict[int, float] = {}
        self.round_trip = LatencyStats()

    def on_published(self, stamp_ns: int) -> None:
        """時刻 stamp_ns の状態を配信した。"""
        with self._lock:
            self._published_at[stamp_ns] = self._wall_clock()

    def on_target_received(self, stamp_ns: int) -> None:
        """時刻 stamp_ns の状態から作った目標が届いた。それより前の状態の配信の記録は捨てる。"""
        received_at = self._wall_clock()
        with self._lock:
            # 前の試行の状態から作った目標など、配信の記録が無い目標は測らない
            if stamp_ns in self._published_at:
                self.round_trip.add(received_at - self._published_at[stamp_ns])
            self._published_at = {stamp: at for stamp, at in self._published_at.items() if stamp > stamp_ns}
