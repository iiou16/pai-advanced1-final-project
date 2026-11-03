"""物理ステップを、実時間の何倍かを上限にした間隔で進める。"""

import time
from collections.abc import Callable


class RealTimePacer:
    """物理ステップの実時間の間隔を、シミュレーション時間の 1 制御周期 ÷ 実時間比の上限 以上にする。

    計算が間に合わないときは待たずに進める（遅れを取り戻すために速めることもしない）。
    """

    def __init__(
        self, control_dt_s: float, max_real_time_factor: float,
        wall_clock: Callable[[], float] = time.perf_counter, sleep: Callable[[float], None] = time.sleep,
    ) -> None:  # fmt: skip
        """制御周期 [s] と実時間比の上限を受け取る。wall_clock と sleep はテストで差し替える。"""
        self._min_interval_s = control_dt_s / max_real_time_factor
        self._wall_clock = wall_clock
        self._sleep = sleep
        self._last_step_s: float | None = None

    def wait_for_next_step(self) -> None:
        """前の物理ステップから最小の間隔が過ぎるまで待つ。最初の呼び出しは待たない。"""
        now = self._wall_clock()
        if self._last_step_s is not None:
            remaining = self._last_step_s + self._min_interval_s - now
            if remaining > 0:
                self._sleep(remaining)
                now = self._wall_clock()
        self._last_step_s = now
