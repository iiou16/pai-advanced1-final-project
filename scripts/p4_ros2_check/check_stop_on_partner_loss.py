"""2 台で押している最中に相手の協調のノードを止めると、残ったロボットが押すのをやめて待機することを確かめる。

launch で協調（cooperative）の 1 試行を始め、2 台がともに PUSH を MIN_PUSHING_S 続けたら、go2_b の協調のノードを止める。
go2_a は go2_b の作業の状態が途絶えた（partner_stale_after_s より古くなった）と判断して PUSH から READY に移り、
停止の指令を出すはず。go2_b の最後の状態の時刻から、go2_a が PUSH を抜けた時刻までをシミュレーション時間で測り、
途絶えの判定時間と判断の周期の和（と通信の遅れの余裕）より遅ければ終了コード 1 で終わる。

前提: リポジトリのルートで、/opt/ros/jazzy/setup.bash・ros2_ws/install/setup.bash・scripts/setup/env.sh を source 済み。
    uv run --frozen python scripts/p4_ros2_check/check_stop_on_partner_loss.py --run-name p4-partner-loss --seed 6101

出力: outputs/ros2_team_eval/<run-name>/partner_loss_check.json（このスクリプトの計測値）。
シミュレーションノードは確認の後に止めるので、試行の記録（trials.csv）は残らない。
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import rclpy
from geometry_msgs.msg import TwistStamped
from go2_interfaces.msg import AgentStatus
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from cooperative_go2.entities.command_expiry import ns_to_seconds
from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.interface_adapters.config_loader import load_ros2_team_settings, load_team_run_settings
from cooperative_go2.interface_adapters.ros2_messages import time_to_ns
from cooperative_go2.interface_adapters.ros2_team_messages import read_agent_status
from cooperative_go2.interface_adapters.ros2_topics import AGENT_STATUS_TOPIC, COMMAND_QOS, TEAM_STATUS_QOS

LAUNCH_FILE = "ros2_ws/launch/team.launch.py"
CONFIG = Path("configs/ros2_team.yaml")
EVAL_ROOT = Path("outputs/ros2_team_eval")
KEPT_ROBOT = "go2_a"
STOPPED_ROBOT = "go2_b"
STOPPED_PROCESS_PATTERN = f"go2-ros2-team-coord.*__ns:=/{STOPPED_ROBOT}"
# 2 台が押している状態が落ち着いてから止める
MIN_PUSHING_S = 1.0
# 止めてから観測を続けるシミュレーション時間 [s]
OBSERVE_AFTER_S = 2.0
# 通信の遅れ（受信から判断まで）として許す余裕 [s]
LATENCY_MARGIN_S = 0.1
WAIT_TIMEOUT_S = 600.0
POLL_S = 0.05


class StatusRecorder:
    """作業の状態と、残すロボットの速度指令を受け取って記録する。"""

    def __init__(self, node: Node) -> None:
        self._lock = threading.Lock()
        self.statuses: dict[str, list[tuple[float, TeamPhase]]] = {KEPT_ROBOT: [], STOPPED_ROBOT: []}
        self.commands: list[tuple[float, float]] = []
        node.create_subscription(AgentStatus, AGENT_STATUS_TOPIC, self._on_status, TEAM_STATUS_QOS)
        node.create_subscription(TwistStamped, f"/{KEPT_ROBOT}/cmd_vel", self._on_command, COMMAND_QOS)

    def _on_status(self, message: AgentStatus) -> None:
        report = read_agent_status(message)
        with self._lock:
            if report.agent_id in self.statuses:
                self.statuses[report.agent_id].append((ns_to_seconds(report.stamp_ns), report.phase))

    def _on_command(self, message: TwistStamped) -> None:
        with self._lock:
            self.commands.append((ns_to_seconds(time_to_ns(message.header.stamp)), message.twist.linear.x))

    def snapshot(self) -> tuple[dict[str, list[tuple[float, TeamPhase]]], list[tuple[float, float]]]:
        with self._lock:
            return {name: list(items) for name, items in self.statuses.items()}, list(self.commands)


def _both_pushing_for(recorder: StatusRecorder, duration_s: float) -> bool:
    statuses, _ = recorder.snapshot()
    for items in statuses.values():
        if not items or items[-1][1] is not TeamPhase.PUSH:
            return False
        started = items[-1][0]
        for stamp_s, phase in reversed(items):
            if phase is not TeamPhase.PUSH:
                break
            started = stamp_s
        if items[-1][0] - started < duration_s:
            return False
    return True


def _wait_until(predicate, what: str) -> None:
    deadline = time.monotonic() + WAIT_TIMEOUT_S
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError(f"{what} を {WAIT_TIMEOUT_S} s 待ったが起きない")
        time.sleep(POLL_S)


def _stop_process(pattern: str) -> None:
    # pgrep は一致するプロセスが無いと終了コード 1 を返すので、終了コードではなく出力で判定する
    pids = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True, check=False).stdout.split()
    if not pids:
        raise RuntimeError(f"{pattern} のプロセスが見つからない")
    for pid in pids:
        subprocess.run(["kill", f"-{signal.SIGINT.value}", pid], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="相手の協調のノードを止めると、残ったロボットが押すのをやめることを確かめる")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    team = load_team_run_settings(Path(load_ros2_team_settings(CONFIG).team_config)).settings.team
    allowed_s = team.team_readiness.partner_stale_after_s + team.high_level_period_s + LATENCY_MARGIN_S
    launch = subprocess.Popen(
        ["ros2", "launch", LAUNCH_FILE, "mode:=cooperative", f"seed:={args.seed}", "num_trials:=1",
         f"run_name:={args.run_name}", "device:=cpu"],
    )  # fmt: skip
    rclpy.init()
    node = Node("partner_loss_check")
    recorder = StatusRecorder(node)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    try:
        _wait_until(lambda: _both_pushing_for(recorder, MIN_PUSHING_S), f"2 台が {MIN_PUSHING_S} s 押し続ける")
        _stop_process(STOPPED_PROCESS_PATTERN)
        statuses, _ = recorder.snapshot()
        stopped_at_s = statuses[KEPT_ROBOT][-1][0]
        _wait_until(
            lambda: recorder.snapshot()[0][KEPT_ROBOT][-1][0] >= stopped_at_s + OBSERVE_AFTER_S,
            f"{STOPPED_ROBOT} を止めてからシミュレーション時間 {OBSERVE_AFTER_S} s",
        )
    finally:
        launch.send_signal(signal.SIGINT)
        launch.wait(timeout=WAIT_TIMEOUT_S)
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()
    statuses, commands = recorder.snapshot()
    last_partner_s = statuses[STOPPED_ROBOT][-1][0]
    left_push = [(stamp_s, phase) for stamp_s, phase in statuses[KEPT_ROBOT]
                 if stamp_s > last_partner_s and phase is not TeamPhase.PUSH]  # fmt: skip
    if not left_push:
        print(f"{KEPT_ROBOT} は {STOPPED_ROBOT} が途絶えても PUSH のままだった", file=sys.stderr)
        sys.exit(1)
    left_at_s, left_phase = left_push[0]
    commands_after = [vx for stamp_s, vx in commands if stamp_s >= left_at_s]
    result = {
        "last_partner_status_sim_s": last_partner_s,
        "kept_robot_left_push_sim_s": left_at_s,
        "kept_robot_phase_after": left_phase.name,
        "delay_s": round(left_at_s - last_partner_s, 3),
        "allowed_delay_s": allowed_s,
        "max_forward_command_after_mps": max(commands_after) if commands_after else None,
    }
    output = EVAL_ROOT / args.run_name / "partner_loss_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    stopped_in_time = result["delay_s"] <= allowed_s and left_phase is TeamPhase.READY
    stopped_moving = commands_after and max(commands_after) == 0.0
    if not (stopped_in_time and stopped_moving):
        print("相手が途絶えても、残ったロボットが時間内に押すのをやめなかった", file=sys.stderr)
        sys.exit(1)
    print(f"{STOPPED_ROBOT} が途絶えてから {result['delay_s']} s で {KEPT_ROBOT} は押すのをやめて待機した")


if __name__ == "__main__":
    main()
