# ロボットが対象の物体（箱）に触れているか。シミュレーションの接触判定の真値。
# header.stamp は判定した時点のシミュレーション時刻。受け手は有効期限の判定に使う。
std_msgs/Header header
bool in_contact
