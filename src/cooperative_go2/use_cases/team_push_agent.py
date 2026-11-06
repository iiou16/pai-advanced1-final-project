"""1 台のロボットの、規則ベースで箱を押す判断（状態機械・押す位置の判定・速度指令）。

並列環境のシミュレーションで調整するときも、ROS 2 の協調のノードで動かすときも、同じこの判断を使う。
相手の状態を使うか（協調）・使わないか（1 台、通信なしの独立制御）は CoordinationMode で切り替える。
相手の状態を使わないときは、相手の準備を待たずに押し始め、相手が押すのをやめても押し続ける。
"""

from dataclasses import dataclass
from enum import Enum

import torch

from cooperative_go2.entities.coordination_state_machine import (
    CoordinationObservation,
    CoordinationState,
    advance_coordination,
)
from cooperative_go2.entities.push_success import is_in_goal_region
from cooperative_go2.entities.push_trial import PlanarPose
from cooperative_go2.entities.team_push_control import (
    TeamPushInputs,
    compute_push_commands,
    compute_push_geometry,
    compute_push_predicates,
)
from cooperative_go2.entities.team_readiness import PartnerStatus, TeamReadinessInputs, is_team_ready_with_partner
from cooperative_go2.entities.team_task_config import TeamRuleBasedConfig


class CoordinationMode(Enum):
    """相手の状態を判断に使うか。

    - COOPERATIVE: 2 台の協調（E5）。相手の準備を待って押し始め、相手が押すのをやめたら止まる。
    - INDEPENDENT: 1 台（E3）、または通信なしの独立制御（E4）。自分の準備だけで押し始め、相手の状態を使わない。
    """

    COOPERATIVE = "cooperative"
    INDEPENDENT = "independent"


@dataclass(frozen=True)
class TeamPushAgentInputs:
    """1 回の判断の材料。

    Attributes:
        now_s: 判断する時刻（状態の時刻）[s]。前回の判断の時刻以降。
        robot: 自分の胴体の位置と向き（ワールド座標系）。
        box: 箱の位置と向き（ワールド座標系）。
        partner: 最後に受け取った相手の状態。まだ受け取っていない、または相手がいなければ None。
            INDEPENDENT では使わない。
        task_assigned: 作業の依頼を受けている。
        fallen: 自分が転倒した。
    """

    now_s: float
    robot: PlanarPose
    box: PlanarPose
    partner: PartnerStatus | None
    task_assigned: bool
    fallen: bool


@dataclass(frozen=True)
class TeamPushDecision:
    """1 回の判断の結果。

    Attributes:
        commands: 速度指令 (1, 3)。列は vx [m/s]、vy [m/s]（常に 0）、yaw_rate [rad/s]（本体座標系）。
        state: 判断のあとの状態機械の状態。
        push_position_xy_m: 自分の押す位置 (x, y) [m]（相手に知らせる意図）。
        push_dir_xy: 押す方向の単位ベクトル (x, y)（相手に知らせる意図）。
        box_in_retreat_area: 箱の中心が退避領域に入っている。
    """

    commands: torch.Tensor
    state: CoordinationState
    push_position_xy_m: tuple[float, float]
    push_dir_xy: tuple[float, float]
    box_in_retreat_area: bool


@dataclass(frozen=True)
class RetreatArea:
    """箱を押し出す先（ClearCorridor の目標）。

    Attributes:
        center_xy_m: 退避領域の中心 (x, y) [m]。
        radius_m: 退避領域の半径 [m]。
    """

    center_xy_m: tuple[float, float]
    radius_m: float


class TeamPushAgent:
    """1 台のロボットの判断。状態機械の状態を持ち、判断のたびに進める。"""

    def __init__(self, config: TeamRuleBasedConfig, robot_index: int, mode: CoordinationMode) -> None:
        """設定・シーンの robots の中での自分の番号（押す側の割り当てに使う）・相手の状態を使うかを受け取る。

        Raises:
            IndexError: robot_index に対応する押す側の割り当てが設定に無い。
        """
        self._config = config
        self._lateral_offset_m = config.push_lateral_offsets_m[robot_index]
        self._mode = mode
        self._state = CoordinationState.initial(now_s=0.0)
        self._retreat: RetreatArea | None = None

    @property
    def state(self) -> CoordinationState:
        """現在の状態機械の状態。"""
        return self._state

    def start(self, retreat: RetreatArea, now_s: float) -> None:
        """新しい作業（箱を retreat へ移す）を始める。状態機械を INIT に戻す。依頼を受けたことは次の判断で伝える。"""
        self._retreat = retreat
        self._state = CoordinationState.initial(now_s=now_s)

    def decide(self, inputs: TeamPushAgentInputs) -> TeamPushDecision:
        """状態機械を 1 回進め、その段階の速度指令を返す。

        Raises:
            AgentNotStartedError: start() で作業を始める前に呼んだ。
        """
        retreat = self._retreat
        if retreat is None:
            raise AgentNotStartedError("start() で作業を始めてから判断する")
        push_inputs = TeamPushInputs(
            robot_xy_m=torch.tensor([[inputs.robot.x_m, inputs.robot.y_m]]),
            robot_yaw_rad=torch.tensor([inputs.robot.yaw_rad]),
            box_xy_m=torch.tensor([[inputs.box.x_m, inputs.box.y_m]]),
            box_yaw_rad=torch.tensor([inputs.box.yaw_rad]),
            retreat_xy_m=torch.tensor([retreat.center_xy_m]),
            lateral_offset_m=self._lateral_offset_m,
        )
        geometry = compute_push_geometry(push_inputs, self._config.push_control)
        predicates = compute_push_predicates(geometry, self._state.phase, self._config.push_control)
        box_in_retreat_area = bool(
            is_in_goal_region(push_inputs.box_xy_m, push_inputs.retreat_xy_m, retreat.radius_m)[0]
        )
        observation = CoordinationObservation(
            task_assigned=inputs.task_assigned,
            at_push_position=bool(predicates.at_push_position[0]),
            aligned=bool(predicates.aligned[0]),
            team_ready=self._team_ready(inputs),
            box_in_retreat_area=box_in_retreat_area,
            fallen=inputs.fallen,
        )
        self._state = advance_coordination(self._state, observation, inputs.now_s, self._config.coordination)
        commands = compute_push_commands(push_inputs, geometry, self._state.phase, self._config.push_control)
        push_position = geometry.push_position_xy_m[0]
        push_dir = geometry.push_dir[0]
        return TeamPushDecision(
            commands=commands,
            state=self._state,
            push_position_xy_m=(float(push_position[0]), float(push_position[1])),
            push_dir_xy=(float(push_dir[0]), float(push_dir[1])),
            box_in_retreat_area=box_in_retreat_area,
        )

    def _team_ready(self, inputs: TeamPushAgentInputs) -> bool:
        if self._mode is CoordinationMode.INDEPENDENT:
            return True
        readiness = TeamReadinessInputs(own=self._state, partner=inputs.partner, now_s=inputs.now_s)
        return is_team_ready_with_partner(readiness, self._config.team_readiness)


class AgentNotStartedError(RuntimeError):
    """作業を始める前に判断しようとしたことを表す。"""
