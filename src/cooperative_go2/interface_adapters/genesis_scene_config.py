"""Genesis 固有のシーン設定（configs/scenes/*.yaml の genesis 節）と、Genesis が受け付ける値の範囲。

シミュレータ固有の語彙はこの層に閉じ込め、entities には渡さない。
このモジュールは genesis を import しない（設定の読み込みだけでシミュレータを読み込まないようにするため）。
"""

from dataclasses import dataclass

from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError

# Genesis 1.4.5 の剛体の材質（gs.materials.Rigid）が受け付ける摩擦係数の範囲。範囲外はシーンの構築時に例外になる
GENESIS_MIN_FRICTION = 0.01
GENESIS_MAX_FRICTION = 5.0
# Genesis の摩擦円錐のモデル（gs.friction_cone の名前）。pyramidal は角錐による近似、elliptic は厳密な円錐
FRICTION_CONE_MODELS = ("pyramidal", "elliptic")


class InvalidGenesisSceneConfigError(InvalidConfigError):
    """Genesis 固有のシーン設定が、Genesis の受け付ける値や値域を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidGenesisSceneConfigError)


@dataclass(frozen=True)
class GenesisSceneConfig:
    """Genesis の物理エンジンの接触計算に関する設定。

    Attributes:
        max_collision_pairs: 物理エンジンが扱う衝突候補の上限。超えた場合はシーンの物理ステップで例外にする。正の整数。
        friction_cone: 摩擦円錐のモデル。FRICTION_CONE_MODELS のいずれか。
            pyramidal（Genesis の既定）では、滑っている箱に床から押し上げる誤った力が生じ、箱が浮く・跳ねる。
            elliptic ではこれが起きない。シーン全体に効くため、Go2 の足と床の摩擦の計算も変わる。
    """

    max_collision_pairs: int
    friction_cone: str

    def __post_init__(self) -> None:
        _check.positive("genesis.max_collision_pairs", self.max_collision_pairs)
        if self.friction_cone not in FRICTION_CONE_MODELS:
            raise InvalidGenesisSceneConfigError(
                f"genesis.friction_cone は {FRICTION_CONE_MODELS} のいずれかが必要: {self.friction_cone!r}"
            )


def require_genesis_friction(name: str, friction: float) -> None:
    """摩擦係数が Genesis の受け付ける範囲になければ InvalidGenesisSceneConfigError を送出する。

    Args:
        name: エラーメッセージに出す設定のキー。
        friction: 摩擦係数。
    """
    if not GENESIS_MIN_FRICTION <= friction <= GENESIS_MAX_FRICTION:
        raise InvalidGenesisSceneConfigError(
            f"{name} は Genesis が受け付ける [{GENESIS_MIN_FRICTION}, {GENESIS_MAX_FRICTION}] が必要: {friction}"
        )
