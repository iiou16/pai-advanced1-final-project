"""Unitree Go2 の関節構成。"""

from dataclasses import astuple, dataclass, fields


@dataclass(frozen=True)
class Go2JointAngles:
    """Go2 の 12 関節の角度 [rad]。フィールド名は URDF の関節名と同じ。

    フィールドの並び（脚は FR → FL → RR → RL、各脚は hip → thigh → calf）が、歩行方策の観測・行動で使う
    関節の順序の唯一の定義になる。学習済みの方策はこの順序に依存するため、並びを変えるとチェックポイントの互換性が失われる。
    """

    FR_hip_joint: float
    FR_thigh_joint: float
    FR_calf_joint: float
    FL_hip_joint: float
    FL_thigh_joint: float
    FL_calf_joint: float
    RR_hip_joint: float
    RR_thigh_joint: float
    RR_calf_joint: float
    RL_hip_joint: float
    RL_thigh_joint: float
    RL_calf_joint: float

    def in_order(self) -> tuple[float, ...]:
        """関節角を GO2_JOINT_NAMES の順に並べて返す。"""
        return astuple(self)


GO2_JOINT_NAMES: tuple[str, ...] = tuple(field.name for field in fields(Go2JointAngles))

NUM_GO2_JOINTS = len(GO2_JOINT_NAMES)
