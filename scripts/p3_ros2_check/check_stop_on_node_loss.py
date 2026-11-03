"""方策のノードを止めるとロボットが止まることを確かめる（要件 5.5、5.6、P3-05）。

launch で 1 試行を始め、Go2 が歩き出したら高レベル方策のノードを止める。速度指令が有効期限を過ぎると、
歩行方策は停止の指令で歩くので、Go2 はその場で立ち止まる。続けて歩行方策のノードを止める。関節目標が有効期限を
過ぎると、シミュレーションノードは既定姿勢を保つので、Go2 は立ったまま動かない。
それぞれの後で、胴体の水平の移動量と高さを /go2_a/odom から測り、止まらなければ終了コード 1 で終わる。
止まった後も、既定姿勢に移るときの揺れで速さは数秒 0 にならないので、速さではなく区間の正味の移動量で判定する。

前提: リポジトリのルートで、/opt/ros/jazzy/setup.bash・ros2_ws/install/setup.bash・scripts/setup/env.sh を source 済み。
    uv run --frozen python scripts/p3_ros2_check/check_stop_on_node_loss.py --run-name p3-stop-check --seed 2002 \
        --device cuda --video true

出力: outputs/ros2_eval/<run-name>/（シミュレーションノードの記録）、outputs/ros2_bags/<run-name>/（rosbag2）、
outputs/ros2_eval/<run-name>/stop_check.json（このスクリプトの計測値）。
"""

import argparse
import json
import math
import signal
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import rclpy
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

LAUNCH_FILE = "ros2_ws/launch/push_single.launch.py"
ODOM_TOPIC = "/go2_a/odom"
EVAL_ROOT = Path("outputs/ros2_eval")
COORD_PROCESS_PATTERN = "go2-ros2-coord"
GAIT_PROCESS_PATTERN = "go2-ros2-gait"
# 歩き出したとみなす胴体の水平の速さ [m/s]（直近の WALKING_WINDOW_S の平均）と、判定を始めるシミュレーション時刻 [s]。
# 試行の始めは着地の揺れで一瞬速さが出るので、平均を取り、始めの 1 s は判定しない
WALKING_SPEED_MPS = 0.15
WALKING_WINDOW_S = 0.5
WALKING_EARLIEST_S = 1.0
# 止まったとみなす、測る区間の胴体の水平の正味の移動量 [m]
STOPPED_DISPLACEMENT_M = 0.05
# 立っているとみなす胴体の高さ [m]（既定姿勢の高さは約 0.3 m）
STANDING_HEIGHT_M = 0.2
# ノードを止めてから測り始めるまでのシミュレーション時間 [s]。有効期限（最大 0.3 s）と減速の過渡を含める
SETTLE_S = 1.5
# 止まったかを測るシミュレーション時間 [s]
MEASURE_S = 1.5
# 動画を撮ると実時間比が 0.15 程度まで下がり、20 s の試行に 2 分以上かかるため長めに取る
WAIT_TIMEOUT_S = 600.0
POLL_S = 0.05


@dataclass(frozen=True)
class PhaseMeasurement:
    """ある区間の胴体の水平の速さの平均 [m/s]、水平の正味の移動量 [m]、胴体の高さの最小 [m]。"""

    mean_speed_mps: float
    displacement_m: float
    min_height_m: float


class OdomRecorder:
    """/go2_a/odom の時刻・水平の速さ・胴体の高さを記録する。"""

    def __init__(self, node: Node) -> None:
        self._lock = threading.Lock()
        # (時刻 [s], 水平の速さ [m/s], 高さ [m], x [m], y [m])
        self.samples: list[tuple[float, float, float, float, float]] = []
        node.create_subscription(Odometry, ODOM_TOPIC, self._on_odom, qos_profile_sensor_data)

    def _on_odom(self, message: Odometry) -> None:
        stamp = message.header.stamp.sec + message.header.stamp.nanosec * 1e-9
        linear = message.twist.twist.linear
        with self._lock:
            position = message.pose.pose.position
            self.samples.append((stamp, math.hypot(linear.x, linear.y), position.z, position.x, position.y))

    def latest(self) -> tuple[float, float, float, float, float] | None:
        with self._lock:
            return self.samples[-1] if self.samples else None

    def measure(self, start_s: float, end_s: float) -> PhaseMeasurement:
        with self._lock:
            window = [sample for sample in self.samples if start_s <= sample[0] <= end_s]
        if not window:
            raise RuntimeError(f"区間 [{start_s:.2f}, {end_s:.2f}] s の odom が無い")
        first, last = window[0], window[-1]
        return PhaseMeasurement(
            mean_speed_mps=sum(sample[1] for sample in window) / len(window),
            displacement_m=math.hypot(last[3] - first[3], last[4] - first[4]),
            min_height_m=min(sample[2] for sample in window),
        )


def _wait_until(predicate, what: str) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} を {WAIT_TIMEOUT_S} s 待ったが起きない")
        time.sleep(POLL_S)


def _sim_time_now(recorder: OdomRecorder) -> float:
    latest = recorder.latest()
    if latest is None:
        raise RuntimeError("odom をまだ受け取っていない")
    return latest[0]


def _wait_sim_time(recorder: OdomRecorder, target_s: float, what: str) -> None:
    _wait_until(lambda: (recorder.latest() or (0.0,))[0] >= target_s, what)


def _is_walking(recorder: OdomRecorder) -> bool:
    latest = recorder.latest()
    if latest is None or latest[0] < WALKING_EARLIEST_S:
        return False
    now = latest[0]
    return recorder.measure(now - WALKING_WINDOW_S, now).mean_speed_mps > WALKING_SPEED_MPS


def _stop_process(pattern: str) -> None:
    # launch が起動した方策のノードに SIGINT を送る（ros2 の通常の停止と同じ）
    pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=True).stdout.split()
    if not pids:
        raise RuntimeError(f"{pattern} のプロセスが見つからない")
    for pid in pids:
        subprocess.run(["kill", f"-{signal.SIGINT.value}", pid], check=True)


def _stop_and_measure(recorder: OdomRecorder, pattern: str) -> tuple[float, PhaseMeasurement]:
    stopped_at = _sim_time_now(recorder)
    _stop_process(pattern)
    start = stopped_at + SETTLE_S
    end = start + MEASURE_S
    _wait_sim_time(recorder, end, f"{pattern} を止めた後のシミュレーション時間 {end:.1f} s")
    return stopped_at, recorder.measure(start, end)


def main() -> None:
    parser = argparse.ArgumentParser(description="方策のノードを止めるとロボットが止まることを確かめる")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True, help="Genesis を動かすデバイス")
    parser.add_argument("--video", choices=["true", "false"], required=True, help="止まる様子の評価動画を撮るか")
    args = parser.parse_args()
    launch = subprocess.Popen(
        [
            "ros2", "launch", LAUNCH_FILE, f"seed:={args.seed}", "num_trials:=1", f"run_name:={args.run_name}",
            f"device:={args.device}", f"video:={args.video}",
        ],
    )  # fmt: skip
    rclpy.init()
    node = Node("stop_check")
    recorder = OdomRecorder(node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        _wait_until(lambda: _is_walking(recorder), "Go2 が歩き出す")
        now = _sim_time_now(recorder)
        walking = recorder.measure(now - WALKING_WINDOW_S, now)
        coord_stopped_at, after_coord = _stop_and_measure(recorder, COORD_PROCESS_PATTERN)
        gait_stopped_at, after_gait = _stop_and_measure(recorder, GAIT_PROCESS_PATTERN)
    finally:
        launch.wait(timeout=WAIT_TIMEOUT_S)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    result = {
        "walking_before_stop": asdict(walking),
        "coord_policy_stopped_at_sim_s": coord_stopped_at,
        "after_coord_policy_stopped": asdict(after_coord),
        "gait_policy_stopped_at_sim_s": gait_stopped_at,
        "after_gait_policy_stopped": asdict(after_gait),
        "settle_s": SETTLE_S,
        "measure_s": MEASURE_S,
    }
    (EVAL_ROOT / args.run_name / "stop_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    stopped = all(
        phase.displacement_m < STOPPED_DISPLACEMENT_M and phase.min_height_m > STANDING_HEIGHT_M
        for phase in (after_coord, after_gait)
    )
    if not stopped:
        print("方策のノードを止めても Go2 が止まらない、または立っていない", file=sys.stderr)
        sys.exit(1)
    print("方策のノードを止めると Go2 はその場で立ち止まった")


if __name__ == "__main__":
    main()
