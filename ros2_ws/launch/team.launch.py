"""ROS 2 を経由した通路の開通（1 台・2 台の規則ベース）を 1 コマンドで起動する。

シミュレーション・タスク管理と、ロボットごとの歩行方策・協調のノードを別プロセスで起動し、すべて use_sim_time=true にする。
ロボットごとのノードは、そのロボットの Namespace（例: /go2_a）で起動する。ノードは学習用 venv の実行ファイル（go2-ros2-*）で
起動する（torch と Genesis が venv にしか無いため）。シミュレーションノードが全試行を終えて終了すると、launch 全体を終える。

歩行方策のノードは 1 台の箱押し（push_single.launch.py）と同じものを、同じ設定（歩行方策のチェックポイント・指令の有効期限・
実時間の条件）で起動する。

前提: リポジトリのルートで、/opt/ros/jazzy/setup.bash・ros2_ws/install/setup.bash・scripts/setup/env.sh を source 済み。

例（E5: 2 台の協調）:
    ros2 launch ros2_ws/launch/team.launch.py mode:=cooperative seed:=6001 num_trials:=16 run_name:=e5-seed6001
例（E4: 2 台・通信なし）:
    ros2 launch ros2_ws/launch/team.launch.py mode:=independent seed:=6001 num_trials:=16 run_name:=e4-seed6001
例（E3: 1 台）:
    ros2 launch ros2_ws/launch/team.launch.py config:=configs/ros2_team_single.yaml robots:=go2_a \\
        mode:=independent seed:=6001 num_trials:=16 run_name:=e3-seed6001
"""

from pathlib import Path

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution

REPO_ROOT = Path(__file__).resolve().parents[2]
VENV_BIN = REPO_ROOT / ".venv" / "bin"
EVAL_ROOT = "outputs/ros2_team_eval"
BAG_ROOT = "outputs/ros2_team_bags"
# 歩行方策のノードの設定（1 台の箱押しと同じ歩行方策・指令の有効期限・実時間の条件）
GAIT_CONFIG = "configs/ros2_push_single.yaml"
# rosbag2 に残すトピック（相対名はロボットの Namespace の下）
RECORDED_ROBOT_TOPICS = ("odom", "joint_states", "cmd_vel", "joint_targets")
RECORDED_GLOBAL_TOPICS = ("/clock", "/team/object_state", "/team/agent_status", "/tf")


def _venv_process(executable: str, args: list, name: str, ros_args: list) -> ExecuteProcess:
    return ExecuteProcess(
        cmd=[str(VENV_BIN / executable), *args, "--ros-args", *ros_args, "-p", "use_sim_time:=true"],
        cwd=str(REPO_ROOT),
        output="screen",
        name=name,
    )


def _robot_nodes(context: LaunchContext) -> list[ExecuteProcess]:
    # ロボットの名前はシーンの設定と同じ順・同じ名前で渡す（協調のノードは Namespace の名前でシーンの中の自分を探す）
    robots = [name for name in LaunchConfiguration("robots").perform(context).split(",") if name]
    metrics_dir = PathJoinSubstitution([EVAL_ROOT, LaunchConfiguration("run_name"), "nodes"]).perform(context)
    config = LaunchConfiguration("config").perform(context)
    mode = LaunchConfiguration("mode").perform(context)
    nodes = []
    for robot in robots:
        namespace = ["-r", f"__ns:=/{robot}"]
        nodes.append(
            _venv_process(
                "go2-ros2-gait", ["--config", GAIT_CONFIG, "--metrics-dir", f"{metrics_dir}/{robot}"],
                f"{robot}_gait_policy", [*namespace, "-r", "__node:=gait_policy"],
            )
        )  # fmt: skip
        nodes.append(
            _venv_process(
                "go2-ros2-team-coord", ["--config", config, "--mode", mode, "--metrics-dir", metrics_dir],
                f"{robot}_coord_policy", [*namespace, "-r", "__node:=coord_policy"],
            )
        )  # fmt: skip
    return nodes


def _bag(context: LaunchContext) -> list[ExecuteProcess]:
    robots = [name for name in LaunchConfiguration("robots").perform(context).split(",") if name]
    robot_topics = [f"/{robot}/{topic}" for robot in robots for topic in RECORDED_ROBOT_TOPICS]
    output = PathJoinSubstitution([BAG_ROOT, LaunchConfiguration("run_name")]).perform(context)
    return [
        ExecuteProcess(
            cmd=["ros2", "bag", "record", "--storage", "mcap", "--output", output,
                 *robot_topics, *RECORDED_GLOBAL_TOPICS],
            cwd=str(REPO_ROOT),
            output="screen",
            name="rosbag_record",
            condition=IfCondition(LaunchConfiguration("record_bag")),
        )  # fmt: skip
    ]


def generate_launch_description() -> LaunchDescription:
    """launch の構成を返す。"""
    config = LaunchConfiguration("config")
    sim = _venv_process(
        "go2-ros2-team-sim",
        [
            "--config", config,
            "--mode", LaunchConfiguration("mode"),
            "--seed", LaunchConfiguration("seed"),
            "--num-trials", LaunchConfiguration("num_trials"),
            "--device", LaunchConfiguration("device"),
            "--run-name", LaunchConfiguration("run_name"),
            "--video", LaunchConfiguration("video"),
        ],
        "genesis_sim",
        ["-r", "__node:=genesis_sim"],
    )  # fmt: skip
    task_manager = _venv_process(
        "go2-ros2-task-manager", ["--config", config], "task_manager", ["-r", "__node:=task_manager"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value="configs/ros2_team.yaml"),
            DeclareLaunchArgument(
                "robots", default_value="go2_a,go2_b", description="ロボットの Namespace（シーンの設定の順、カンマ区切り）"
            ),
            DeclareLaunchArgument("mode", description="協調のモード（cooperative: E5 / independent: E3・E4）"),
            DeclareLaunchArgument("seed", description="評価用の乱数シード"),
            DeclareLaunchArgument("num_trials", description="続けて行う試行の数"),
            DeclareLaunchArgument("run_name", description=f"{EVAL_ROOT}/<run_name> に保存する"),
            DeclareLaunchArgument("device", default_value="cuda", description="Genesis を動かすデバイス（cpu / cuda）"),
            DeclareLaunchArgument("video", default_value="false", description="評価動画を撮るか（true / false）"),
            DeclareLaunchArgument(
                "record_bag", default_value="false", description=f"{BAG_ROOT}/<run_name> に rosbag2 を残すか"
            ),
            OpaqueFunction(function=_bag),
            OpaqueFunction(function=_robot_nodes),
            task_manager,
            sim,
            RegisterEventHandler(
                OnProcessExit(
                    target_action=sim,
                    on_exit=[EmitEvent(event=Shutdown(reason="シミュレーションノードが終了した"))],
                )
            ),
        ]
    )
