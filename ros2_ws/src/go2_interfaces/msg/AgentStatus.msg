# 協調するロボットが互いに配信する、作業の状態と押す意図。
# header.stamp は配信した時点のシミュレーション時刻、header.frame_id は位置と方向の座標系。

uint8 PHASE_INIT=0
uint8 PHASE_APPROACH=1
uint8 PHASE_ALIGN=2
uint8 PHASE_READY=3
uint8 PHASE_PUSH=4
uint8 PHASE_VERIFY=5
uint8 PHASE_DONE=6
uint8 PHASE_FAILED=7

std_msgs/Header header
# 配信したロボットの名前（例: go2_a）
string agent_id
uint8 phase
# 箱を押すために立つ位置 [m]
geometry_msgs/Point push_position
# 押す方向の単位ベクトル
geometry_msgs/Vector3 push_direction
# 押し始める準備ができているか
bool ready
# 配信した時点（header.stamp の状態）の自分の胴体の位置 [m]（z は使わない）とヨー角 [rad]。
# 学習した協調の方策が、相手の位置と向きとして観測に入れる
geometry_msgs/Point position
float64 yaw
# 配信した時点に決めた速度指令（x: vx [m/s]、y: vy [m/s]、z: yaw_rate [rad/s]、本体座標系）。押す意図として相手に伝える
geometry_msgs/Vector3 command
