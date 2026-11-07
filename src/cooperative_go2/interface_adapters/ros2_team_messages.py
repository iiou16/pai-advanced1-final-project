"""2 台の協調で共有する作業の状態（AgentStatus）と、協調の判断が使う型との変換。

時刻はシミュレーション時刻の整数ナノ秒で扱う。位置と方向はワールド座標系（frame_id）の水平成分。
"""

from dataclasses import dataclass

from geometry_msgs.msg import Point, Vector3
from go2_interfaces.msg import AgentStatus

from cooperative_go2.entities.command_expiry import ns_to_seconds
from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.team_readiness import PartnerStatus
from cooperative_go2.interface_adapters.ros2_messages import ns_to_time, time_to_ns

# 押し始める準備ができているとみなす段階（AgentStatus の ready）
_READY_PHASES = frozenset({TeamPhase.READY, TeamPhase.PUSH})


class UnknownTeamPhaseError(ValueError):
    """AgentStatus の phase が、決めた段階（PHASE_INIT〜PHASE_FAILED）のどれでもないことを表す。"""


@dataclass(frozen=True)
class AgentStatusReport:
    """1 台のロボットが配信する、作業の状態と押す意図。

    Attributes:
        agent_id: ロボットの名前（例: go2_a）。
        stamp_ns: 状態を決めた時刻（シミュレーション時刻）[ns]。
        phase: 作業の段階。
        push_position_xy_m: 押すために立つ位置 (x, y) [m]。
        push_dir_xy: 押す方向の単位ベクトル (x, y)。
        position_xy_m: stamp_ns の状態の自分の胴体の位置 (x, y) [m]。
        yaw_rad: stamp_ns の状態の自分のヨー角 [rad]。
        command: stamp_ns の状態から決めた速度指令 (vx [m/s], vy [m/s], yaw_rate [rad/s])。
    """

    agent_id: str
    stamp_ns: int
    phase: TeamPhase
    push_position_xy_m: tuple[float, float]
    push_dir_xy: tuple[float, float]
    position_xy_m: tuple[float, float]
    yaw_rad: float
    command: tuple[float, float, float]

    @property
    def ready(self) -> bool:
        """押し始める準備ができているか（READY か PUSH）。"""
        return self.phase in _READY_PHASES

    def to_partner_status(self) -> PartnerStatus:
        """相手の状態として、チームの準備の判定に渡す形にする。"""
        return PartnerStatus(phase=self.phase, stamp_s=ns_to_seconds(self.stamp_ns))


def make_agent_status(report: AgentStatusReport, frame_id: str) -> AgentStatus:
    """配信する AgentStatus を作る。"""
    message = AgentStatus()
    message.header.stamp = ns_to_time(report.stamp_ns)
    message.header.frame_id = frame_id
    message.agent_id = report.agent_id
    message.phase = int(report.phase)
    message.push_position = Point(x=report.push_position_xy_m[0], y=report.push_position_xy_m[1], z=0.0)
    message.push_direction = Vector3(x=report.push_dir_xy[0], y=report.push_dir_xy[1], z=0.0)
    message.ready = report.ready
    message.position = Point(x=report.position_xy_m[0], y=report.position_xy_m[1], z=0.0)
    message.yaw = report.yaw_rad
    message.command = Vector3(x=report.command[0], y=report.command[1], z=report.command[2])
    return message


def read_agent_status(message: AgentStatus) -> AgentStatusReport:
    """受け取った AgentStatus を読む。

    Raises:
        UnknownTeamPhaseError: phase が決めた段階のどれでもない。
    """
    try:
        phase = TeamPhase(message.phase)
    except ValueError as error:
        raise UnknownTeamPhaseError(f"{message.agent_id} の作業の段階 {message.phase} は決めた段階にない") from error
    return AgentStatusReport(
        agent_id=message.agent_id,
        stamp_ns=time_to_ns(message.header.stamp),
        phase=phase,
        push_position_xy_m=(message.push_position.x, message.push_position.y),
        push_dir_xy=(message.push_direction.x, message.push_direction.y),
        position_xy_m=(message.position.x, message.position.y),
        yaw_rad=message.yaw,
        command=(message.command.x, message.command.y, message.command.z),
    )
