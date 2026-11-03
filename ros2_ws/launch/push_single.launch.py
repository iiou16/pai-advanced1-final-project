"""ROS 2 を経由した 1 台の箱押し（P3）を 1 コマンドで起動する。

シミュレーション・歩行方策・高レベル方策の 3 ノードを別プロセスで起動し、すべて use_sim_time=true にする。
ノードは学習用 venv の実行ファイル（go2-ros2-*）で起動する（torch と Genesis が venv にしか無いため）。
シミュレーションノードが全試行を終えて終了すると、launch 全体を終える。

前提: リポジトリのルートで、/opt/ros/jazzy/setup.bash・ros2_ws/install/setup.bash・scripts/setup/env.sh を source 済み。

例:
    ros2 launch ros2_ws/launch/push_single.launch.py seed:=2001 num_trials:=10 run_name:=ros2-wide60-seed2001
"""

from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = REPO_ROOT / ".venv" / "bin"
EVAL_ROOT = "outputs/ros2_eval"
BAG_ROOT = "outputs/ros2_bags"
# rosbag2 に残すトピック（相対名はロボットの Namespace の下）
RECORDED_ROBOT_TOPICS = ("odom", "joint_states", "box_contact", "cmd_vel", "joint_targets")
RECORDED_GLOBAL_TOPICS = ("/clock", "/team/object_state", "/tf")


def _node(executable: str, node_name: str, extra_args: list) -> ExecuteProcess:
    namespace = LaunchConfiguration("robot")
    return ExecuteProcess(
        cmd=[
            str(VENV_BIN / executable),
            "--config", LaunchConfiguration("config"),
            *extra_args,
            "--ros-args",
            "-r", ["__ns:=/", namespace],
            "-r", f"__node:={node_name}",
            "-p", "use_sim_time:=true",
        ],
        cwd=str(REPO_ROOT),
        output="screen",
        name=node_name,
    )  # fmt: skip


def generate_launch_description() -> LaunchDescription:
    """launch の構成を返す。"""
    run_name = LaunchConfiguration("run_name")
    robot = LaunchConfiguration("robot")
    metrics_dir = PathJoinSubstitution([EVAL_ROOT, run_name, "nodes"])
    sim = _node(
        "go2-ros2-sim",
        "genesis_sim",
        [
            "--seed", LaunchConfiguration("seed"),
            "--num-trials", LaunchConfiguration("num_trials"),
            "--device", LaunchConfiguration("device"),
            "--run-name", run_name,
            "--video", LaunchConfiguration("video"),
        ],
    )  # fmt: skip
    gait = _node("go2-ros2-gait", "gait_policy", ["--metrics-dir", metrics_dir])
    coord = _node("go2-ros2-coord", "coord_policy", ["--metrics-dir", metrics_dir])
    robot_topics = [["/", robot, "/", topic] for topic in RECORDED_ROBOT_TOPICS]
    bag = ExecuteProcess(
        cmd=[
            "ros2", "bag", "record", "--storage", "mcap",
            "--output", PathJoinSubstitution([BAG_ROOT, run_name]),
            *robot_topics, *RECORDED_GLOBAL_TOPICS,
        ],
        cwd=str(REPO_ROOT),
        output="screen",
        name="rosbag_record",
        condition=IfCondition(LaunchConfiguration("record_bag")),
    )  # fmt: skip
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="configs/ros2_push_single.yaml"),
            DeclareLaunchArgument("robot", default_value="go2_a", description="ロボットの Namespace（/ を除く）"),
            DeclareLaunchArgument("seed", description="評価用の乱数シード"),
            DeclareLaunchArgument("num_trials", description="続けて行う試行の数"),
            DeclareLaunchArgument("run_name", description=f"{EVAL_ROOT}/<run_name> に保存する"),
            DeclareLaunchArgument("device", default_value="cpu", description="Genesis を動かすデバイス（cpu / cuda）"),
            DeclareLaunchArgument("video", default_value="false", description="評価動画を撮るか（true / false）"),
            DeclareLaunchArgument(
                "record_bag", default_value="true", description=f"{BAG_ROOT}/<run_name> に rosbag2 を残すか"
            ),
            bag,
            gait,
            coord,
            sim,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=sim,
                    on_exit=[EmitEvent(event=Shutdown(reason="シミュレーションノードが終了した"))],
                )
            ),
        ]
    )
