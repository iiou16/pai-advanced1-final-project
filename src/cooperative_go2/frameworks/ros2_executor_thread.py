"""ROS 2 のシミュレーションノードが使う、受信の処理を回す別のスレッドと、物理ステップの前のその確認。

試行の流れ（物理ステップ）は 1 つのスレッドで進め、ROS 2 の受信・サービス・アクションの応答は別のスレッドで処理する。
"""

import threading

from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from cooperative_go2.interface_adapters.real_time_pacer import RealTimePacer


class ExecutorThreadFailedError(RuntimeError):
    """ROS 2 の受信を処理するスレッドが例外で止まったことを表す。関節目標や応答が届かなくなる。"""


class ExecutorThread:
    """ROS 2 の受信の処理を別のスレッドで回し、そのスレッドが例外で止まったことを試行のスレッドに伝える。"""

    def __init__(self, node: Node) -> None:
        self.executor = MultiThreadedExecutor()
        self.executor.add_node(node)
        self._error: Exception | None = None
        threading.Thread(target=self._spin, daemon=True).start()

    def _spin(self) -> None:
        try:
            self.executor.spin()
        except Exception as error:
            # このスレッドの最上位。例外を握りつぶさず保持し、試行のスレッドが raise_if_failed() で送出する
            self._error = error

    def find_error_or_none(self) -> Exception | None:
        """受信のスレッドを止めた例外を返す。止まっていなければ None。"""
        return self._error

    def raise_if_failed(self) -> None:
        """受信のスレッドが例外で止まっていれば ExecutorThreadFailedError を送出する。"""
        if self._error is not None:
            raise ExecutorThreadFailedError(f"ROS 2 の受信のスレッドが止まった: {self._error!r}") from self._error


class HealthCheckedPacer:
    """物理ステップの前に、ROS 2 の受信のスレッドが動いていることを確かめてから実時間の間隔を整える。"""

    def __init__(self, pacer: RealTimePacer, executor_thread: ExecutorThread) -> None:
        self._pacer = pacer
        self._executor_thread = executor_thread

    def wait_for_next_step(self) -> None:
        """受信のスレッドが止まっていれば ExecutorThreadFailedError を送出し、そうでなければ次のステップまで待つ。"""
        self._executor_thread.raise_if_failed()
        self._pacer.wait_for_next_step()
