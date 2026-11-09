"""頑健性の評価の条件（箱の質量・摩擦・配置）を、1 台の箱押しの評価の設定に当てはめる。

速度指令の遅れは設定ではなく環境の構築（PushEnvSetup）で渡すので、ここでは扱わない。
"""

import dataclasses
from dataclasses import dataclass

from cooperative_go2.entities.push_robustness import PushRobustnessCondition
from cooperative_go2.interface_adapters.config_loader import PushSceneSettings, PushTrainingSettings


class IneffectiveBoxFrictionError(ValueError):
    """条件の箱の摩擦が床の摩擦より小さく、実際の摩擦が変わらないことを表す。"""


@dataclass(frozen=True)
class PushEvaluationSettings:
    """条件を当てはめた評価の設定。

    Attributes:
        settings: 配置（task.reset）を条件の値に置き換えた箱押しの設定。
        scene: 箱の質量・摩擦を条件の値に置き換えたシーンの設定。
    """

    settings: PushTrainingSettings
    scene: PushSceneSettings


def apply_robustness_condition(
    settings: PushTrainingSettings, scene: PushSceneSettings, condition: PushRobustnessCondition, floor_friction: float
) -> PushEvaluationSettings:
    """箱の質量・摩擦と配置を condition の値に置き換えた評価の設定を返す。

    接触する 2 面の摩擦は大きい方が使われるため、箱の摩擦が床より小さい条件は実際の摩擦が変わらない。
    評価しても意味の無い条件なので止める。床と同じ摩擦は受け付ける。

    Args:
        settings: 評価の設定（configs/push_single_wide.yaml 形式）。
        scene: settings.scene_config から読んだシーンの設定。
        condition: 当てはめる条件。
        floor_friction: 床の摩擦係数（シミュレータの既定値）。

    Raises:
        IneffectiveBoxFrictionError: condition の箱の摩擦が floor_friction より小さい。
        InvalidConfigError: 置き換えた箱の質量・摩擦が、シーンの設定の値域を外れた。
    """
    if condition.box_friction < floor_friction:
        raise IneffectiveBoxFrictionError(
            f"条件 {condition.name} の箱の摩擦 {condition.box_friction} は床の摩擦 {floor_friction} より小さく、"
            "接触する 2 面の大きい方が使われるため実際の摩擦は変わらない"
        )
    box = dataclasses.replace(scene.box, mass_kg=condition.box_mass_kg, friction=condition.box_friction)
    task = dataclasses.replace(settings.task, reset=condition.reset)
    return PushEvaluationSettings(
        settings=dataclasses.replace(settings, task=task), scene=dataclasses.replace(scene, box=box)
    )
