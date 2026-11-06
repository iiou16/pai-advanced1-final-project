"""ROS 2 の実行系のトピック・サービス・アクションの名前と QoS。

ロボットごとの名前は相対名にし、ノードを起動するときの Namespace（例: /go2_a）で分ける。複数台で混線させないため。
シミュレーションノードはロボットと同じ Namespace で起動する。
"""

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

# シミュレーションノード → 方策のノード
CLOCK_TOPIC = "/clock"
ODOM_TOPIC = "odom"
JOINT_STATES_TOPIC = "joint_states"
BOX_CONTACT_TOPIC = "box_contact"
OBJECT_STATE_TOPIC = "/team/object_state"
# 方策のノードの間
CMD_VEL_TOPIC = "cmd_vel"
JOINT_TARGETS_TOPIC = "joint_targets"
# シミュレーションノード → 歩行方策のノード（ロボットを初期姿勢に戻したことを伝える）
GAIT_RESET_SERVICE = "gait_policy/reset"
# 依頼者 → 高レベル方策のノード（1 台の構成ではシミュレーションノードが依頼者を兼ねる。
# 2 台の構成ではタスク管理のノードが各ロボットの協調のノードへ依頼する）
CLEAR_CORRIDOR_ACTION = "clear_corridor"
# 協調のノードの間で、作業の状態と押す意図を共有する
AGENT_STATUS_TOPIC = "/team/agent_status"
# 依頼者 → タスク管理のノード（2 台の構成。シミュレーションノードが試行ごとに依頼する）
TASK_CLEAR_CORRIDOR_ACTION = "/task/clear_corridor"

# 座標系の名前。WORLD_FRAME はシミュレーションのワールド座標系（真値）
WORLD_FRAME = "map"
BOX_FRAME = "box"
BASE_LINK_FRAME = "base_link"

# 毎物理ステップ配信する状態。取りこぼしても次のステップの状態を使えばよいので、再送しない
SENSOR_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
)
# 指令。受け手が有効期限で古さを判定するので、届いたものは順に使う
COMMAND_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
# 作業の状態。受け手が時刻で途絶えを判定するので、届いたものは順に使う
TEAM_STATUS_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)
# /clock。rclpy の時刻の購読（best effort）とも、ros2 bag（reliable）とも互換にするため reliable で配信する
CLOCK_QOS = QoSProfile(
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
)


def robot_topic(robot_name: str, topic: str) -> str:
    """ロボットの名前（例: go2_a）の Namespace の下の、相対名 topic の絶対名（例: /go2_a/odom）を返す。

    複数のロボットのトピックやアクションを 1 つのノードから扱うノード（シミュレーションノード・タスク管理のノード）が使う。
    """
    return f"/{robot_name}/{topic}"


def robot_frame(namespace: str, frame: str) -> str:
    """ロボットの Namespace（例: /go2_a）を前に付けた座標系の名前（例: go2_a/base_link）を返す。"""
    return f"{namespace.strip('/')}/{frame}"
