"""規則ベースで箱を押すときの、1 台分の押す位置・判定・速度指令。

各ロボットは、箱の後ろに割り当てられた押す位置（押す方向の線から横へ lateral_offset_m ずらした位置）へ向かい、
押す方向へ向きを合わせてから、自分の線に沿って押す方向へ直進する。押す方向は箱の中心から退避領域の中心への向き。
2 台で押すとき、箱が回り始めたら、2 つの方法で回転を戻せる。
- 押す線をずらす: 2 台の押す線をそろって遅れている側へずらし、押す力の中心を箱の中央から遅れている側へ移す。
  2 台とも押す速さを落とさないので、重い箱でも押す力が減らない。
- 速さを落とす: 前に出ている側のロボットが押す速さを落とす。速度指令の上限が決まっているので、遅れている側を
  速めるのではなく前に出ている側を遅くする。押す力の合計が減るため、2 台でやっと動く重い箱は止まりやすい。

判定と指令は、協調の状態機械（coordination_state_machine）が使う条件と、段階ごとの速度指令を返す。
自分の状態と箱の状態だけで計算し、相手との通信は使わない。テンソルは並列環境のバッチ（1 台分）。
位置はワールド座標系の水平成分。速度指令は本体座標系の [vx, vy, yaw_rate] で、横移動（vy）は使わない。
"""

import math
from dataclasses import dataclass

import torch

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError
from cooperative_go2.entities.coordination_state_machine import TeamPhase
from cooperative_go2.entities.planar_frame import wrap_angle_rad
from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE

X_AXIS, Y_AXIS = 0, 1
YAW_RATE_COLUMN = 2
# 押す位置に着いたあと（向きを合わせている間・待っている間）は、近づくときの許容値の何倍まで離れたら外れたとみなすか
_HOLD_TOLERANCE_FACTOR = 2.0
_HOLDING_PHASES = frozenset({TeamPhase.ALIGN, TeamPhase.READY})


class InvalidTeamPushConfigError(InvalidConfigError):
    """規則ベースで箱を押す制御の設定値が不変条件（正の速さ・許容値など）を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidTeamPushConfigError)


@dataclass(frozen=True)
class TeamPushConfig:
    """規則ベースで箱を押す制御のパラメータ。速さと旋回は、歩行方策が学習した速度指令の範囲に収める。

    Attributes:
        approach_speed_mps: 押す位置へ向かうときの前進速度 [m/s]。正の値。
        push_speed_mps: 押すときの前進速度 [m/s]。正の値。
        yaw_gain_per_s: 向きの誤差 [rad] からヨー角速度の指令 [rad/s] を作る比例ゲイン [1/s]。正の値。
        max_yaw_rate_radps: ヨー角速度の指令の絶対値の上限 [rad/s]。正の値。
        standoff_m: 押す位置の、箱の中心から押す方向の逆への距離 [m]。箱の奥行きの半分と Go2 の胴体の前端までの
            距離の和より少し大きくし、押す位置で箱に触れないようにする。正の値。
        position_tolerance_m: 押す位置に着いたとみなす距離 [m]。正の値。押す位置を押す方向へ越えていても、
            自分の線から横にこの距離の中にいれば着いたとみなす（行き過ぎて後ろ向きに戻らないため）。
            向きを合わせている間と待っている間は、この 2 倍まで離れるまで外れたとみなさない（判定が揺れないため）。
        approach_slowdown_m: 押す位置からこの距離 [m] の中では、距離に比例して近づく速さを落とす。正の値。
        heading_tolerance_rad: 押し始める前に、向きが合ったとみなす押す方向とのずれ [rad]。正の値。
        push_lateral_tolerance_m: 押している間に、自分の線から外れたとみなす横のずれ [m]。押し始める前より緩くし、
            押している間の小さな揺れで止まらないようにする。正の値。
        push_heading_tolerance_rad: 押している間に、向きが外れたとみなす押す方向とのずれ [rad]。正の値。
        lookahead_m: 押している間に向かう目標点を、自分の線の上で前へ置く距離 [m]。小さいほど線へ強く戻る。正の値。
        yaw_balance_gain_per_rad: 箱の回転 [rad] あたりに、前に出ている側の押す速さを落とす割合 [1/rad]。0 以上。
            0 なら速さでは回転を抑えない。
        yaw_shift_gain_m_per_rad: 箱の回転 [rad] あたりに、押す線を遅れている側へずらす距離 [m/rad]。0 以上。
            0 なら押す線をずらさない。
        max_yaw_shift_m: 押す線をずらす距離の上限 [m]。0 以上。ロボットが箱の幅から外れない範囲にする。
    """

    approach_speed_mps: float
    push_speed_mps: float
    yaw_gain_per_s: float
    max_yaw_rate_radps: float
    standoff_m: float
    position_tolerance_m: float
    approach_slowdown_m: float
    heading_tolerance_rad: float
    push_lateral_tolerance_m: float
    push_heading_tolerance_rad: float
    lookahead_m: float
    yaw_balance_gain_per_rad: float
    yaw_shift_gain_m_per_rad: float
    max_yaw_shift_m: float

    def __post_init__(self) -> None:
        for name in (
            "approach_speed_mps",
            "push_speed_mps",
            "yaw_gain_per_s",
            "max_yaw_rate_radps",
            "standoff_m",
            "position_tolerance_m",
            "approach_slowdown_m",
            "heading_tolerance_rad",
            "push_lateral_tolerance_m",
            "push_heading_tolerance_rad",
            "lookahead_m",
        ):
            _check.positive(f"push_control.{name}", getattr(self, name))
        _check.non_negative("push_control.yaw_balance_gain_per_rad", self.yaw_balance_gain_per_rad)
        _check.non_negative("push_control.yaw_shift_gain_m_per_rad", self.yaw_shift_gain_m_per_rad)
        _check.non_negative("push_control.max_yaw_shift_m", self.max_yaw_shift_m)


@dataclass(frozen=True)
class TeamPushInputs:
    """1 台分の判定と指令の材料。テンソルは並列環境のバッチ。

    Attributes:
        robot_xy_m: ロボットの胴体の位置 (num_envs, 2) [m]。
        robot_yaw_rad: ロボットのヨー角 (num_envs,) [rad]。
        box_xy_m: 箱の中心 (num_envs, 2) [m]。
        box_yaw_rad: 箱のヨー角 (num_envs,) [rad]。箱の x 軸（奥行きの辺）の向き。
        retreat_xy_m: 退避領域の中心 (num_envs, 2) [m]。箱の中心と重ならないこと（押す方向が決まらない）。
        lateral_offset_m: このロボットの押す位置の、押す方向の線からの横のずれ [m]。押す方向の左が正。
            1 台で押すときは 0。
    """

    robot_xy_m: torch.Tensor
    robot_yaw_rad: torch.Tensor
    box_xy_m: torch.Tensor
    box_yaw_rad: torch.Tensor
    retreat_xy_m: torch.Tensor
    lateral_offset_m: float


@dataclass(frozen=True)
class PushGeometry:
    """押す方向と、ロボットの押す位置からのずれ。テンソルは並列環境のバッチ。

    Attributes:
        push_dir: 押す方向の単位ベクトル (num_envs, 2)。箱の中心から退避領域の中心への向き。
        push_position_xy_m: このロボットの押す位置 (num_envs, 2) [m]。
        distance_to_push_position_m: 押す位置までの距離 (num_envs,) [m]。
        beyond_push_position_m: 押す位置から押す方向へ越えている距離 (num_envs,) [m]。手前にいれば負。
        lateral_error_m: 自分の線（押す方向の線から lateral_offset_m ずらし、さらに箱の回転に応じてずらした線）からの
            横のずれ (num_envs,) [m]。
        heading_error_rad: 押す方向から見たロボットの向きのずれ (num_envs,) [rad]。正なら左へ回ると合う。
        box_yaw_error_rad: 押す方向から見た箱の回転 (num_envs,) [rad]。箱は前後が同じ形なので [-π/2, π/2) に折り返す。
    """

    push_dir: torch.Tensor
    push_position_xy_m: torch.Tensor
    distance_to_push_position_m: torch.Tensor
    beyond_push_position_m: torch.Tensor
    lateral_error_m: torch.Tensor
    heading_error_rad: torch.Tensor
    box_yaw_error_rad: torch.Tensor


@dataclass(frozen=True)
class PushPredicates:
    """協調の状態機械に渡す、自分の状態から判定した条件 (num_envs,) の bool マスク。

    Attributes:
        at_push_position: 押す位置の近くにいる。
        aligned: 押し始める前の判定では、押す位置で押す方向へ向きが合っている。押している間の判定では、
            自分の線から大きく外れず、向きも大きく外れていない。
    """

    at_push_position: torch.Tensor
    aligned: torch.Tensor


def compute_push_geometry(inputs: TeamPushInputs, config: TeamPushConfig) -> PushGeometry:
    """押す方向・押す位置と、ロボットのずれを返す。

    箱が回っていれば、押す線を遅れている側へずらす（箱が左回りに回ると左側の端が遅れるので、左へずらす）。
    """
    to_retreat = inputs.retreat_xy_m - inputs.box_xy_m
    push_dir = to_retreat / torch.linalg.vector_norm(to_retreat, dim=1, keepdim=True)
    lateral_dir = torch.stack((-push_dir[:, Y_AXIS], push_dir[:, X_AXIS]), dim=1)
    push_angle = torch.atan2(push_dir[:, Y_AXIS], push_dir[:, X_AXIS])
    box_yaw_error = wrap_angle_rad(2.0 * (inputs.box_yaw_rad - push_angle)) / 2.0
    max_shift = config.max_yaw_shift_m
    shift = torch.clamp(config.yaw_shift_gain_m_per_rad * box_yaw_error, -max_shift, max_shift)
    lateral_offset = inputs.lateral_offset_m + shift
    push_position = inputs.box_xy_m - push_dir * config.standoff_m + lateral_dir * lateral_offset.unsqueeze(1)
    box_to_robot = inputs.robot_xy_m - inputs.box_xy_m
    return PushGeometry(
        push_dir=push_dir,
        push_position_xy_m=push_position,
        distance_to_push_position_m=torch.linalg.vector_norm(inputs.robot_xy_m - push_position, dim=1),
        beyond_push_position_m=torch.sum((inputs.robot_xy_m - push_position) * push_dir, dim=1),
        lateral_error_m=torch.sum(box_to_robot * lateral_dir, dim=1) - lateral_offset,
        heading_error_rad=wrap_angle_rad(push_angle - inputs.robot_yaw_rad),
        box_yaw_error_rad=box_yaw_error,
    )


def compute_push_predicates(geometry: PushGeometry, phase: TeamPhase, config: TeamPushConfig) -> PushPredicates:
    """phase の段階で使う判定を返す。PUSH の間は、押し始める前より緩い許容値で向きと線を判定する。"""
    tolerance = config.position_tolerance_m
    near_tolerance = tolerance * _HOLD_TOLERANCE_FACTOR if phase in _HOLDING_PHASES else tolerance
    passed_on_own_line = (geometry.beyond_push_position_m >= 0) & (geometry.lateral_error_m.abs() <= tolerance)
    at_push_position = (geometry.distance_to_push_position_m <= near_tolerance) | passed_on_own_line
    if phase is TeamPhase.PUSH:
        aligned = (geometry.lateral_error_m.abs() <= config.push_lateral_tolerance_m) & (
            geometry.heading_error_rad.abs() <= config.push_heading_tolerance_rad
        )
        # 押している間は箱と一緒に押す位置も前へ動くので、押す位置に着いているかは問わない
        return PushPredicates(at_push_position=torch.ones_like(aligned), aligned=aligned)
    aligned = at_push_position & (geometry.heading_error_rad.abs() <= config.heading_tolerance_rad)
    return PushPredicates(at_push_position=at_push_position, aligned=aligned)


def compute_push_commands(
    inputs: TeamPushInputs, geometry: PushGeometry, phase: TeamPhase, config: TeamPushConfig
) -> torch.Tensor:
    """phase の段階の速度指令 (num_envs, 3) を返す。

    - APPROACH: 押す位置へ向かって進む（向きが合うほど速く、後ろ向きならその場で向きを変える。
      押す位置の approach_slowdown_m 手前から、距離に比例して遅くする）。
    - ALIGN: その場で押す方向へ向きを変える。
    - PUSH: 自分の線の上で lookahead_m 前の点へ向かって押す。2 台で押すときは、箱の回転で前に出ている側が速さを落とす。
    - それ以外（INIT・READY・VERIFY・DONE・FAILED）: 止まる。
    """
    commands = torch.zeros(inputs.robot_xy_m.shape[0], VELOCITY_COMMAND_SIZE, device=inputs.robot_xy_m.device)
    if phase is TeamPhase.APPROACH:
        slowdown = torch.clamp(geometry.distance_to_push_position_m / config.approach_slowdown_m, max=1.0)
        return _steer_toward(inputs, geometry.push_position_xy_m, config.approach_speed_mps * slowdown, config)
    if phase is TeamPhase.ALIGN:
        commands[:, YAW_RATE_COLUMN] = _yaw_rate(geometry.heading_error_rad, config)
        return commands
    if phase is TeamPhase.PUSH:
        lateral_dir = torch.stack((-geometry.push_dir[:, Y_AXIS], geometry.push_dir[:, X_AXIS]), dim=1)
        on_own_line = inputs.robot_xy_m - lateral_dir * geometry.lateral_error_m.unsqueeze(1)
        target = on_own_line + geometry.push_dir * config.lookahead_m
        speed = config.push_speed_mps * yaw_balance_speed_scale(
            geometry.box_yaw_error_rad, inputs.lateral_offset_m, config.yaw_balance_gain_per_rad
        )
        return _steer_toward(inputs, target, speed, config)
    return commands


def yaw_balance_speed_scale(
    box_yaw_error_rad: torch.Tensor, lateral_offset_m: float, gain_per_rad: float
) -> torch.Tensor:
    """箱の回転に応じた、押す速さの倍率 (num_envs,) を [0, 1] で返す。

    箱が左回り（正）に回ると、押す方向の左側の端が遅れ、右側の端が前に出る。前に出ている側（右、lateral_offset_m < 0）の
    ロボットが速さを落とし、遅れている側が押す力で箱を戻す。押す方向の線の上（lateral_offset_m = 0）で押す 1 台は落とさない。

    Args:
        box_yaw_error_rad: 押す方向から見た箱の回転 (num_envs,) [rad]。
        lateral_offset_m: このロボットの押す位置の横のずれ [m]。押す方向の左が正。
        gain_per_rad: 回転 1 rad あたりに落とす割合 [1/rad]。0 以上。
    """
    side = math.copysign(1.0, lateral_offset_m) if lateral_offset_m != 0 else 0.0
    return torch.clamp(1.0 + gain_per_rad * side * box_yaw_error_rad, 0.0, 1.0)


def _yaw_rate(heading_error_rad: torch.Tensor, config: TeamPushConfig) -> torch.Tensor:
    return torch.clamp(config.yaw_gain_per_s * heading_error_rad, -config.max_yaw_rate_radps, config.max_yaw_rate_radps)


def _steer_toward(
    inputs: TeamPushInputs, target_xy_m: torch.Tensor, speed_mps: torch.Tensor, config: TeamPushConfig
) -> torch.Tensor:
    # 目標点の方向とのずれに比例して旋回し、向きが合うほど速く前進する。後ろ向きのときはその場で向きを変える
    to_target = target_xy_m - inputs.robot_xy_m
    heading_error = wrap_angle_rad(torch.atan2(to_target[:, Y_AXIS], to_target[:, X_AXIS]) - inputs.robot_yaw_rad)
    forward = speed_mps * torch.clamp(torch.cos(heading_error), min=0.0)
    return torch.stack((forward, torch.zeros_like(forward), _yaw_rate(heading_error, config)), dim=1)
