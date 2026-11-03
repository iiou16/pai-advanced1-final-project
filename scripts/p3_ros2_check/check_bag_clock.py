"""rosbag2 の記録から、配信された時刻が /clock と整合していることを確かめる（T04、P3-03）。

確かめること:
- /clock が単調に増える（巻き戻らない）。
- 状態・指令・TF のヘッダの時刻が、どれも /clock で配信された時刻のいずれかである（独自の時計で打刻していない）。
- 指令（cmd_vel・joint_targets）の時刻が、記録した時点までに配信された最新の /clock より先に進んでいない。
  記録するプロセスがトピックごとに受け取る順は保証されないので、1 制御周期までの先行は許す。

前提: /opt/ros/jazzy/setup.bash と ros2_ws/install/setup.bash を source 済み。
    uv run --frozen python scripts/p3_ros2_check/check_bag_clock.py outputs/ros2_bags/<run-name> [...]
1 つでも満たさない記録があれば終了コード 1 で終わる。
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import rosbag2_py
from geometry_msgs.msg import TwistStamped
from go2_interfaces.msg import ContactState
from nav_msgs.msg import Odometry
from rclpy.serialization import deserialize_message
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from tf2_msgs.msg import TFMessage
from trajectory_msgs.msg import JointTrajectory

NANOSECONDS_PER_SECOND = 1_000_000_000
CONTROL_DT_NS = 20_000_000
CLOCK_TOPIC = "/clock"
COMMAND_TOPICS = ("/go2_a/cmd_vel", "/go2_a/joint_targets")
MESSAGE_TYPES = {
    CLOCK_TOPIC: Clock,
    "/go2_a/odom": Odometry,
    "/go2_a/joint_states": JointState,
    "/go2_a/box_contact": ContactState,
    "/team/object_state": Odometry,
    "/go2_a/cmd_vel": TwistStamped,
    "/go2_a/joint_targets": JointTrajectory,
    "/tf": TFMessage,
}


@dataclass
class TopicReport:
    """トピックごとの集計。"""

    messages: int = 0
    stamps_not_on_clock: int = 0
    max_lead_over_clock_ns: int = 0


@dataclass
class BagReport:
    """1 つの記録の集計。"""

    bag: str
    clock_messages: int = 0
    clock_rewinds: int = 0
    topics: dict[str, TopicReport] = field(default_factory=dict)

    def passed(self) -> bool:
        if self.clock_messages == 0 or self.clock_rewinds > 0:
            return False
        for name, topic in self.topics.items():
            if topic.messages == 0 or topic.stamps_not_on_clock > 0:
                return False
            if name in COMMAND_TOPICS and topic.max_lead_over_clock_ns > CONTROL_DT_NS:
                return False
        return True


def _stamp_ns(stamp: object) -> int:
    return stamp.sec * NANOSECONDS_PER_SECOND + stamp.nanosec  # type: ignore[attr-defined]


def _header_stamps(topic: str, message: object) -> list[int]:
    if topic == "/tf":
        return [_stamp_ns(transform.header.stamp) for transform in message.transforms]  # type: ignore[attr-defined]
    return [_stamp_ns(message.header.stamp)]  # type: ignore[attr-defined]


def _read(bag: Path) -> list[tuple[str, object]]:
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"), rosbag2_py.ConverterOptions("", ""))
    messages = []
    while reader.has_next():
        topic, data, _ = reader.read_next()
        if topic in MESSAGE_TYPES:
            messages.append((topic, deserialize_message(data, MESSAGE_TYPES[topic])))
    return messages


def check_bag(bag: Path) -> BagReport:
    """記録 bag を読み、時刻の整合を集計する。"""
    messages = _read(bag)
    clock_values = {_stamp_ns(message.clock) for topic, message in messages if topic == CLOCK_TOPIC}
    report = BagReport(bag=str(bag))
    latest_clock_ns = -1
    for topic, message in messages:
        if topic == CLOCK_TOPIC:
            value = _stamp_ns(message.clock)  # type: ignore[attr-defined]
            report.clock_messages += 1
            if value < latest_clock_ns:
                report.clock_rewinds += 1
            latest_clock_ns = max(latest_clock_ns, value)
            continue
        topic_report = report.topics.setdefault(topic, TopicReport())
        for stamp in _header_stamps(topic, message):
            topic_report.messages += 1
            if stamp not in clock_values:
                topic_report.stamps_not_on_clock += 1
            topic_report.max_lead_over_clock_ns = max(topic_report.max_lead_over_clock_ns, stamp - latest_clock_ns)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="rosbag2 の記録の時刻が /clock と整合しているかを確かめる")
    parser.add_argument("bags", type=Path, nargs="+", help="rosbag2 の記録のディレクトリ")
    args = parser.parse_args()
    reports = [check_bag(bag) for bag in args.bags]
    for report in reports:
        print(json.dumps({**asdict(report), "passed": report.passed()}, ensure_ascii=False, indent=2))
    if not all(report.passed() for report in reports):
        print("/clock と整合しない時刻がある", file=sys.stderr)
        sys.exit(1)
    print("すべての記録で、時刻は /clock と整合している")


if __name__ == "__main__":
    main()
