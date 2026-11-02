#!/usr/bin/env bash
# ROS 2 Jazzy の Topic / Action 疎通と、学習用 venv から rclpy と torch を同時に import できることを確認する。
# 前提: scripts/setup/install_ros2_jazzy.sh で ros-jazzy-desktop を導入済み。
# 実行: bash scripts/p0_env_check/ros2_smoke.sh （リポジトリのルートで）
set -eo pipefail

readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
readonly WAIT_TIMEOUT_S=20
readonly VENV_PYTHON="$(git rev-parse --show-toplevel)/.venv/bin/python"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ERROR: ${ROS_SETUP} が無い。ROS 2 Jazzy が未導入。" >&2
  exit 1
fi
# setup.bash は未定義変数を参照するため、source 後に nounset を有効にする
source "${ROS_SETUP}"
set -u

# 他の ROS 2 プロセスと混線しないよう、このテスト専用のドメインを使う
export ROS_DOMAIN_ID=42

background_pids=()
cleanup() {
  for pid in "${background_pids[@]}"; do
    kill "${pid}" 2>/dev/null || true  # 既に終了済みのプロセスは停止不要
  done
}
trap cleanup EXIT

echo "[1/3] Topic: demo_nodes_py talker -> topic echo"
ros2 run demo_nodes_py talker >/dev/null 2>&1 &
background_pids+=($!)
topic_output="$(timeout "${WAIT_TIMEOUT_S}" ros2 topic echo --once /chatter std_msgs/msg/String)"
echo "${topic_output}" | grep -q "Hello World" || { echo "ERROR: /chatter を受信できない" >&2; exit 1; }
echo "  PASS: ${topic_output//$'\n'/ }"

echo "[2/3] Action: fibonacci_action_server <- send_goal"
ros2 run action_tutorials_py fibonacci_action_server >/dev/null 2>&1 &
background_pids+=($!)
action_output="$(timeout "${WAIT_TIMEOUT_S}" ros2 action send_goal /fibonacci \
  action_tutorials_interfaces/action/Fibonacci "{order: 5}")"
echo "${action_output}" | grep -q "SUCCEEDED" || { echo "ERROR: Action が成功しない" >&2; echo "${action_output}" >&2; exit 1; }
echo "  PASS: goal SUCCEEDED"

echo "[3/3] 学習用 venv で rclpy のメッセージ変換と torch(CUDA) を同一プロセスで実行"
# apt の msg 型サポートはシステムの NumPy 1.x 向けにビルドされている。
# 固定長配列 (Odometry の covariance: float64[36]) の変換で venv の NumPy との互換性が問われるため、
# import だけでなく serialize / deserialize まで通す。
"${VENV_PYTHON}" -c "
import numpy, rclpy, torch
from rclpy.serialization import deserialize_message, serialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

odom = Odometry()
odom.pose.covariance[0] = 1.5
restored = deserialize_message(serialize_message(odom), Odometry)
assert restored.pose.covariance[0] == 1.5, 'Odometry の往復変換で値が一致しない'
joint_state = JointState(name=['j0'], position=[0.1])
assert deserialize_message(serialize_message(joint_state), JointState).position[0] == 0.1

rclpy.init(); node = rclpy.create_node('env_check_import'); node.destroy_node(); rclpy.shutdown()
assert torch.cuda.is_available(), 'torch から CUDA が見えない'
print('  PASS: rclpy msg 変換 + torch', torch.__version__, '/ numpy', numpy.__version__)
"
echo "RESULT: PASS"
