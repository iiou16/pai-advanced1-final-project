#!/usr/bin/env bash
# ROS 2 Jazzy の Topic / Action 疎通と、学習用 venv から rclpy と torch を同時に import できることを確認する。
# 前提: scripts/setup/install_ros2_jazzy.sh で ros-jazzy-desktop を導入済み、uv sync で .venv を作成済み。
# 実行: bash scripts/p0_env_check/ros2_smoke.sh
set -eo pipefail

readonly ROS_SETUP="/opt/ros/jazzy/setup.bash"
readonly WAIT_TIMEOUT_S=20
readonly STOP_TIMEOUT_S=10
# readonly と同時に代入すると git の失敗が set -e で止まらないため、先に代入する
repo_root="$(git rev-parse --show-toplevel)"
readonly VENV_PYTHON="${repo_root}/.venv/bin/python"

if [[ ! -f "${ROS_SETUP}" ]]; then
  echo "ERROR: ${ROS_SETUP} が無い。ROS 2 Jazzy が未導入。" >&2
  exit 1
fi
if [[ ! -x "${VENV_PYTHON}" ]]; then
  echo "ERROR: ${VENV_PYTHON} が無い。リポジトリのルートで uv sync を先に実行する。" >&2
  exit 1
fi
# setup.bash は未定義変数を参照するため、source 後に nounset を有効にする
source "${ROS_SETUP}"
set -u

# 他の ROS 2 プロセスや同じ LAN の他ホストと混線しないよう、専用のドメインで自ホスト内に限定する
export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# 前回の実行で残ったノードが応答すると、今回起動したノードが失敗しても PASS してしまうため先に止める
leftover_nodes="$(ros2 node list --no-daemon)" || { echo "ERROR: ros2 node list が失敗した" >&2; exit 1; }
if [[ -n "${leftover_nodes}" ]]; then
  echo "ERROR: ROS_DOMAIN_ID=${ROS_DOMAIN_ID} にノードが残っている。停止してから再実行する:" >&2
  echo "${leftover_nodes}" >&2
  exit 1
fi

# 起動したノードの出力は、失敗時に原因を示すため一時ファイルに残す
log_dir="$(mktemp -d)"

# ros2 run は子プロセスとしてノードを起動し、SIGTERM を子へ転送しない。
# setsid で別のプロセスグループにし、グループごと停止する
# （非対話 bash ではジョブ制御が無効で setsid が fork しないため、$! がそのままプロセスグループ ID になる）。
# rclpy ノードは SIGTERM から終了まで時間がかかるため、終了を待ってから抜ける（直後の再実行で残存と誤検知しない）
background_pgids=()
cleanup() {
  local pgid deadline
  for pgid in "${background_pgids[@]}"; do
    kill -TERM -- "-${pgid}" 2>/dev/null || true  # 既に終了済みのプロセスは停止不要
  done
  deadline=$((SECONDS + STOP_TIMEOUT_S))
  for pgid in "${background_pgids[@]}"; do
    while kill -0 -- "-${pgid}" 2>/dev/null; do
      if (( SECONDS >= deadline )); then
        echo "WARNING: プロセスグループ ${pgid} が ${STOP_TIMEOUT_S}s で終了しないため SIGKILL する" >&2
        kill -KILL -- "-${pgid}" 2>/dev/null || true  # 判定直後に終了した場合は停止不要
        break
      fi
      sleep 0.2
    done
  done
  rm -rf "${log_dir}"
}
trap cleanup EXIT

# 失敗理由と、起動したノードのログを表示して終了する
fail_with_node_log() {
  local message="$1" node_log="$2"
  echo "ERROR: ${message}" >&2
  echo "--- ${node_log##*/} ---" >&2
  cat "${node_log}" >&2
  exit 1
}

echo "[1/3] Topic: demo_nodes_py talker -> topic echo"
talker_log="${log_dir}/talker.log"
setsid ros2 run demo_nodes_py talker >"${talker_log}" 2>&1 &
background_pgids+=($!)
topic_output="$(timeout "${WAIT_TIMEOUT_S}" ros2 topic echo --once /chatter std_msgs/msg/String)" \
  || fail_with_node_log "${WAIT_TIMEOUT_S}s 以内に /chatter を受信できない" "${talker_log}"
grep -q "Hello World" <<<"${topic_output}" \
  || fail_with_node_log "/chatter の内容が想定と異なる: ${topic_output}" "${talker_log}"
echo "  PASS: ${topic_output//$'\n'/ }"

echo "[2/3] Action: fibonacci_action_server <- send_goal"
action_server_log="${log_dir}/fibonacci_action_server.log"
setsid ros2 run action_tutorials_py fibonacci_action_server >"${action_server_log}" 2>&1 &
background_pgids+=($!)
action_output="$(timeout "${WAIT_TIMEOUT_S}" ros2 action send_goal /fibonacci \
  action_tutorials_interfaces/action/Fibonacci "{order: 5}")" \
  || fail_with_node_log "${WAIT_TIMEOUT_S}s 以内に Action の結果を受け取れない" "${action_server_log}"
grep -q "SUCCEEDED" <<<"${action_output}" \
  || fail_with_node_log "Action が成功しない: ${action_output}" "${action_server_log}"
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
restored_position = deserialize_message(serialize_message(joint_state), JointState).position
assert restored_position[0] == 0.1, f'JointState の往復変換で値が一致しない: {restored_position}'

rclpy.init(); node = rclpy.create_node('env_check_import'); node.destroy_node(); rclpy.shutdown()
assert torch.cuda.is_available(), 'torch から CUDA が見えない'
print('  PASS: rclpy msg 変換 + torch', torch.__version__, '/ numpy', numpy.__version__)
"
echo "RESULT: PASS"
