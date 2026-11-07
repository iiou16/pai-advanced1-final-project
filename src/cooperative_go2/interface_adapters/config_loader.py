"""YAML の設定ファイルを読み込み、型付きの設定オブジェクトに変換して検証する。

キーの過不足と値の型を厳密に確かめる。既定値で補わないため、設定ファイルにはすべてのキーを書く。
数値は有限の値だけを受け付ける（YAML の .nan / .inf は不正とする）。
"""

import math
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

from cooperative_go2.entities.command_expiry import CommandExpiryConfig
from cooperative_go2.entities.config_validation import ConfigValueChecker, InvalidConfigError
from cooperative_go2.entities.corridor_scene_config import CorridorSceneConfig
from cooperative_go2.entities.gait_config import GaitConfig
from cooperative_go2.entities.gait_evaluation import GaitEvaluationSuite, InvalidEvaluationScenarioError
from cooperative_go2.entities.push_config import RuleBasedPushConfig
from cooperative_go2.entities.push_scene_config import BoxConfig
from cooperative_go2.entities.push_task_config import (
    PushTaskConfig,
    PushTrainingRunConfig,
    require_rule_based_within_commands,
)
from cooperative_go2.entities.team_marl_config import TeamMarlTaskConfig
from cooperative_go2.entities.team_task_config import TeamRuleBasedConfig
from cooperative_go2.interface_adapters.genesis_scene_config import GenesisSceneConfig, require_genesis_friction
from cooperative_go2.interface_adapters.rsl_rl_config import RslRlPpoConfig
from cooperative_go2.interface_adapters.skrl_marl_config import SkrlMarlPpoConfig

_ROOT_KEY_PATH = "<root>"
# 歩行の設定ファイルのうち、RSL-RL 固有の設定を書く節の名前
RSL_RL_SECTION = "rsl_rl"
# シーンの設定ファイルのうち、Genesis 固有の設定を書く節の名前
GENESIS_SECTION = "genesis"


class ConfigValidationError(ValueError):
    """設定ファイルの内容が不正（キーの過不足、型の不一致、値の不変条件の違反）であることを表す。

    メッセージには不正な項目のキーのパス（例: command.lin_vel_x_mps.low）を含める。
    """


@dataclass(frozen=True)
class GaitSettings:
    """歩行の設定ファイル 1 つ分。gait は内側の層に渡し、rsl_rl は RSL-RL との接続にだけ使う。"""

    gait: GaitConfig
    rsl_rl: RslRlPpoConfig


@dataclass(frozen=True)
class PushSceneSettings:
    """箱押しのシーンの設定ファイル 1 つ分。box は内側の層に渡し、genesis はシーンの構築にだけ使う。"""

    box: BoxConfig
    genesis: GenesisSceneConfig

    def __post_init__(self) -> None:
        # 箱の摩擦はドメインとしては正の値なら意味を持つが、Genesis が受け付ける範囲は狭いため、ここで早めに弾く
        require_genesis_friction("box.friction", self.box.friction)


@dataclass(frozen=True)
class CorridorSceneSettings:
    """通路のシーンの設定ファイル 1 つ分。scene は内側の層に渡し、genesis はシーンの構築にだけ使う。"""

    scene: CorridorSceneConfig
    genesis: GenesisSceneConfig

    def __post_init__(self) -> None:
        # 箱の摩擦はドメインとしては正の値なら意味を持つが、Genesis が受け付ける範囲は狭いため、ここで早めに弾く
        require_genesis_friction("box.friction", self.scene.box.friction)


@dataclass(frozen=True)
class GaitPolicySource:
    """固定して使う歩行方策の出所。

    Attributes:
        run_dir: 歩行学習の出力先（設定の写し gait.yaml とチェックポイントがあるディレクトリ）。
            相対パスは現在の作業ディレクトリ起点。
        checkpoint: 読み込むチェックポイントの反復番号（model_<番号>.pt）。
    """

    run_dir: str
    checkpoint: int


@dataclass(frozen=True)
class TeamRuleBasedSettings:
    """規則ベースで箱を押す作業の設定ファイル（configs/team_rule_based.yaml 形式）1 つ分。

    Attributes:
        scene_config: 通路のシーンの設定ファイル（configs/scenes/*.yaml）。相対パスは現在の作業ディレクトリ起点。
        gait_policy: 固定して使う歩行方策。
        team: 押す側の割り当て・制御・状態機械・開始位置の設定。
    """

    scene_config: str
    gait_policy: GaitPolicySource
    team: TeamRuleBasedConfig


@dataclass(frozen=True)
class TeamRunSettings:
    """規則ベースで箱を押す作業の設定と、それが指すシーンの設定。ロボットの数がそろっていることを確かめてある。"""

    settings: TeamRuleBasedSettings
    scene: CorridorSceneSettings


@dataclass(frozen=True)
class TeamMarlTrainingSettings:
    """2 台の協調学習の設定ファイル（configs/push_multi.yaml）1 つ分。

    Attributes:
        scene_config: 通路のシーンの設定ファイル（configs/scenes/s3_team.yaml 形式）。相対パスは現在の作業ディレクトリ起点。
        gait_policy: 固定して使う歩行方策の出所。
        task: 協調学習の環境の設定。内側の層に渡す。
        skrl: IPPO・MAPPO とネットワークの設定（skrl との接続にだけ使う）。
        training: 学習の実行条件（シードと、更新の回数 max_iterations）。
    """

    scene_config: str
    gait_policy: GaitPolicySource
    task: TeamMarlTaskConfig
    skrl: SkrlMarlPpoConfig
    training: PushTrainingRunConfig


@dataclass(frozen=True)
class TeamMarlRunSettings:
    """2 台の協調学習の設定と、それが指すシーンの設定。ロボットの数がそろっていることを確かめてある。"""

    settings: TeamMarlTrainingSettings
    scene: CorridorSceneSettings


@dataclass(frozen=True)
class PushTrainingSettings:
    """1 台の箱押しの設定ファイル（configs/push_single.yaml）1 つ分。

    Attributes:
        scene_config: シーンの設定ファイル（configs/scenes/s1_push.yaml 形式）。相対パスは現在の作業ディレクトリ起点。
        gait_policy: 固定して使う歩行方策の出所。
        task: 学習環境の設定。内側の層に渡す。
        rule_based: 学習した方策と比べる、規則ベースの制御の設定。
        rsl_rl: 高レベル方策の PPO とネットワークの設定（RSL-RL との接続にだけ使う）。
        training: 学習の実行条件。
    """

    scene_config: str
    gait_policy: GaitPolicySource
    task: PushTaskConfig
    rule_based: RuleBasedPushConfig
    rsl_rl: RslRlPpoConfig
    training: PushTrainingRunConfig

    def __post_init__(self) -> None:
        require_rule_based_within_commands(self.rule_based, self.task.commands)


class InvalidRos2RuntimeConfigError(InvalidConfigError):
    """ROS 2 の実行系の設定が不正であることを表す。"""


@dataclass(frozen=True)
class PushPolicySource:
    """ROS 2 の実行系で使う、学習済みの高レベル方策の出所。

    Attributes:
        run_dir: 箱押しの学習の出力先（設定の写し push_single.yaml とチェックポイントがあるディレクトリ）。
            相対パスは現在の作業ディレクトリ起点。
        checkpoint: 読み込むチェックポイントの反復番号（model_<番号>.pt）。
    """

    run_dir: str
    checkpoint: int


@dataclass(frozen=True)
class Ros2RuntimeConfig:
    """ROS 2 の実行系の、実時間の条件。

    Attributes:
        max_real_time_factor: シミュレーションを実時間の何倍まで速く進めるか。方策のノードの応答を待つ余裕を残すため、
            計算が速くてもこれ以上は速くしない。
        startup_timeout_s: 方策のノードが立ち上がるのを待つ実時間の上限 [s]。
        reply_timeout_s: 方策のノードがリセットと目標の依頼に応えるのを待つ実時間の上限 [s]。
        torch_num_threads: 各ノードで PyTorch が CPU の計算に使うスレッドの数。1 台分の小さなネットワークの推論では、
            複数のスレッドを調整する手間の方が大きく、他のプロセスと CPU を取り合うと推論が制御周期を超えて遅れるため。
    """

    max_real_time_factor: float
    startup_timeout_s: float
    reply_timeout_s: float
    torch_num_threads: int

    def __post_init__(self) -> None:
        check = ConfigValueChecker(InvalidRos2RuntimeConfigError)
        check.positive("max_real_time_factor", self.max_real_time_factor)
        check.positive("startup_timeout_s", self.startup_timeout_s)
        check.positive("reply_timeout_s", self.reply_timeout_s)
        check.positive("torch_num_threads", self.torch_num_threads)


@dataclass(frozen=True)
class Ros2PushSettings:
    """ROS 2 を経由した 1 台の箱押しの設定ファイル（configs/ros2_push_single.yaml）1 つ分。

    Attributes:
        push_config: 箱押しの設定ファイル（configs/push_single.yaml 形式）。配置・成功判定・歩行方策・シーンをここから取る。
            相対パスは現在の作業ディレクトリ起点。
        push_policy: 学習済みの高レベル方策。
        command_expiry: 速度指令と関節目標の有効期限。
        runtime: 実時間の条件。
    """

    push_config: str
    push_policy: PushPolicySource
    command_expiry: CommandExpiryConfig
    runtime: Ros2RuntimeConfig


@dataclass(frozen=True)
class Ros2TeamSettings:
    """ROS 2 を経由した通路の開通（1 台・2 台）の設定ファイル（configs/ros2_team.yaml 形式）1 つ分。

    Attributes:
        team_config: 作業の設定ファイル（configs/team_rule_based*.yaml 形式）。相対パスは現在の作業ディレクトリ起点。
        command_expiry: 速度指令と関節目標の有効期限。
        runtime: 実時間の条件。
    """

    team_config: str
    command_expiry: CommandExpiryConfig
    runtime: Ros2RuntimeConfig


def load_ros2_push_settings(path: Path) -> Ros2PushSettings:
    """ROS 2 を経由した 1 台の箱押しの設定ファイルを読み込み、検証済みの設定を返す。

    push_config が指す箱押しの設定ファイルは読まない（load_push_training_settings で別に読む）。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return _build_dataclass(Ros2PushSettings, _read_yaml(path), _ROOT_KEY_PATH)


def load_gait_settings(path: Path) -> GaitSettings:
    """歩行の設定ファイル（configs/gait.yaml 形式）を読み込み、検証済みの設定を返す。

    Args:
        path: 設定ファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return parse_gait_settings(_read_yaml(path))


def load_gait_evaluation_suite(path: Path) -> GaitEvaluationSuite:
    """歩行方策の評価シナリオのファイル（configs/gait_eval.yaml 形式）を読み込み、検証済みのシナリオ一式を返す。

    Args:
        path: シナリオのファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return _build_dataclass(GaitEvaluationSuite, _read_yaml(path), _ROOT_KEY_PATH)


def load_push_scene_settings(path: Path) -> PushSceneSettings:
    """箱押しのシーンの設定ファイル（configs/scenes/s1_push.yaml 形式）を読み込み、検証済みの設定を返す。

    Args:
        path: 設定ファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return parse_push_scene_settings(_read_yaml(path))


def load_corridor_scene_settings(path: Path) -> CorridorSceneSettings:
    """通路のシーンの設定ファイル（configs/scenes/s2_corridor.yaml 形式）を読み込み、検証済みの設定を返す。

    genesis 節だけを Genesis の設定として分け、残りの節を通路のシーンの配置として読む。

    Args:
        path: 設定ファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    mapping = _require_type(_read_yaml(path), dict, _ROOT_KEY_PATH)
    if GENESIS_SECTION not in mapping:
        raise ConfigValidationError(f"{_ROOT_KEY_PATH}: キー {GENESIS_SECTION!r} が無い")
    scene_data = {key: value for key, value in mapping.items() if key != GENESIS_SECTION}
    genesis = _build_dataclass(GenesisSceneConfig, mapping[GENESIS_SECTION], GENESIS_SECTION)
    scene = _build_dataclass(CorridorSceneConfig, scene_data, _ROOT_KEY_PATH)
    try:
        return CorridorSceneSettings(scene=scene, genesis=genesis)
    except InvalidConfigError as error:
        raise ConfigValidationError(f"{path}: {error}") from error


def load_ros2_team_settings(path: Path) -> Ros2TeamSettings:
    """ROS 2 を経由した通路の開通の設定ファイルを読み込み、検証済みの設定を返す。

    Args:
        path: 設定ファイル（configs/ros2_team.yaml 形式）。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return _build_dataclass(Ros2TeamSettings, _read_yaml(path), _ROOT_KEY_PATH)


def load_team_run_settings(path: Path) -> TeamRunSettings:
    """規則ベースで箱を押す作業の設定ファイルと、それが指すシーンの設定ファイルを読み込み、検証済みの設定を返す。

    Args:
        path: 作業の設定ファイル（configs/team_rule_based.yaml 形式）。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: 作業またはシーンの設定ファイルが無い。
        ConfigValidationError: YAML として読めない、キーの過不足・型・値が不正、または押す側の割り当ての数が
            シーンのロボットの数と違う。
    """
    settings = _build_dataclass(TeamRuleBasedSettings, _read_yaml(path), _ROOT_KEY_PATH)
    scene = load_corridor_scene_settings(Path(settings.scene_config))
    num_offsets = len(settings.team.push_lateral_offsets_m)
    num_robots = len(scene.scene.robots)
    if num_offsets != num_robots:
        raise ConfigValidationError(
            f"{path}: team.push_lateral_offsets_m の数 {num_offsets} がシーン {settings.scene_config} の"
            f"ロボットの数 {num_robots} と違う"
        )
    _require_push_positions_fit_box(path, settings.team, scene.scene.box.size_m)
    return TeamRunSettings(settings=settings, scene=scene)


def load_team_marl_run_settings(path: Path) -> TeamMarlRunSettings:
    """2 台の協調学習の設定ファイルと、それが指すシーンの設定ファイルを読み込み、検証済みの設定を返す。

    歩行方策の学習時の設定との突き合わせは、歩行方策を読み込んだ後に環境の構築時に行う。

    Args:
        path: 協調学習の設定ファイル（configs/push_multi.yaml 形式）。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: 協調学習またはシーンの設定ファイルが無い。
        ConfigValidationError: YAML として読めない、キーの過不足・型・値が不正、または task.agent_side_signs の数が
            シーンのロボットの数と違う。
    """
    settings = _build_dataclass(TeamMarlTrainingSettings, _read_yaml(path), _ROOT_KEY_PATH)
    scene = load_corridor_scene_settings(Path(settings.scene_config))
    num_robots = len(scene.scene.robots)
    if settings.task.num_agents != num_robots:
        raise ConfigValidationError(
            f"{path}: task.agent_side_signs の数 {settings.task.num_agents} がシーン {settings.scene_config} の"
            f"ロボットの数 {num_robots} と違う"
        )
    return TeamMarlRunSettings(settings=settings, scene=scene)


def _require_push_positions_fit_box(path: Path, team: TeamRuleBasedConfig, box_size_m: tuple[float, ...]) -> None:
    # 押す位置が箱の中にあると押す位置へ近づけず、押す線が箱の幅の外にあると箱の端を外して押すため
    size_x, size_y, _ = box_size_m
    control = team.push_control
    if control.standoff_m <= size_x / 2:
        raise ConfigValidationError(
            f"{path}: team.push_control.standoff_m ({control.standoff_m}) は箱の奥行きの半分 ({size_x / 2}) より大きくする"
        )
    for index, offset in enumerate(team.push_lateral_offsets_m):
        reach_m = abs(offset) + control.max_yaw_shift_m
        if reach_m > size_y / 2:
            raise ConfigValidationError(
                f"{path}: team.push_lateral_offsets_m[{index}] の絶対値と max_yaw_shift_m の和 ({reach_m}) が"
                f"箱の幅の半分 ({size_y / 2}) を超える"
            )


def load_push_training_settings(path: Path) -> PushTrainingSettings:
    """1 台の箱押しの設定ファイル（configs/push_single.yaml 形式）を読み込み、検証済みの設定を返す。

    歩行方策の学習時の設定との突き合わせは、歩行方策を読み込んだ後に require_compatible_with_gait() で行う。

    Args:
        path: 設定ファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    return parse_push_training_settings(_read_yaml(path))


def parse_push_training_settings(data: object) -> PushTrainingSettings:
    """YAML を読み込んだ値（dict）から、検証済みの箱押しの設定を作る。

    Raises:
        ConfigValidationError: キーの過不足・型・値が不正。
    """
    return _build_dataclass(PushTrainingSettings, data, _ROOT_KEY_PATH)


def parse_push_scene_settings(data: object) -> PushSceneSettings:
    """YAML を読み込んだ値（dict）から、検証済みの箱押しのシーンの設定を作る。

    Raises:
        ConfigValidationError: キーの過不足・型・値が不正。
    """
    return _build_dataclass(PushSceneSettings, data, _ROOT_KEY_PATH)


def parse_gait_settings(data: object) -> GaitSettings:
    """YAML を読み込んだ値（dict）から検証済みの設定を作る。

    Raises:
        ConfigValidationError: キーの過不足・型・値が不正。
    """
    mapping = _require_type(data, dict, _ROOT_KEY_PATH)
    if RSL_RL_SECTION not in mapping:
        raise ConfigValidationError(f"{_ROOT_KEY_PATH}: キー {RSL_RL_SECTION!r} が無い")
    gait_data = {key: value for key, value in mapping.items() if key != RSL_RL_SECTION}
    return GaitSettings(
        gait=_build_dataclass(GaitConfig, gait_data, _ROOT_KEY_PATH),
        rsl_rl=_build_dataclass(RslRlPpoConfig, mapping[RSL_RL_SECTION], RSL_RL_SECTION),
    )


def _read_yaml(path: Path) -> object:
    text = path.read_text(encoding="utf-8")
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigValidationError(f"{path}: YAML として読めない: {error}") from error


def _join_key(parent: str, child: str) -> str:
    return child if parent == _ROOT_KEY_PATH else f"{parent}.{child}"


def _build_dataclass[T](cls: type[T], data: object, key_path: str) -> T:
    mapping = _require_type(data, dict, key_path)
    expected_keys = {field.name for field in fields(cls)}  # type: ignore[arg-type]
    missing = expected_keys - mapping.keys()
    unknown = mapping.keys() - expected_keys
    if missing or unknown:
        raise ConfigValidationError(f"{key_path}: キーの過不足 不足={sorted(missing)} 未知={sorted(unknown)}")
    type_hints = get_type_hints(cls)
    kwargs = {name: _convert(type_hints[name], mapping[name], _join_key(key_path, name)) for name in expected_keys}
    # 不変条件の違反は、どの設定でも「設定ファイルを直すべき」点で同じなので同じ例外に変換する
    try:
        return cls(**kwargs)
    except (InvalidConfigError, InvalidEvaluationScenarioError) as error:
        raise ConfigValidationError(f"{key_path}: {error}") from error


def _convert(expected_type: object, value: object, key_path: str) -> object:
    if is_dataclass(expected_type) and isinstance(expected_type, type):
        return _build_dataclass(expected_type, value, key_path)
    if expected_type is bool:
        return _require_type(value, bool, key_path)
    if expected_type is str:
        return _require_type(value, str, key_path)
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigValidationError(f"{key_path}: 整数が必要だが {value!r} だった")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
            raise ConfigValidationError(f"{key_path}: 有限の数値が必要だが {value!r} だった")
        return float(value)
    if get_origin(expected_type) is tuple:
        element_type = get_args(expected_type)[0]
        items = _require_type(value, list, key_path)
        return tuple(_convert(element_type, item, f"{key_path}[{index}]") for index, item in enumerate(items))
    raise TypeError(f"{key_path}: 設定の読み込みが対応していない型 {expected_type!r}")


def _require_type[V](value: object, expected: type[V], key_path: str) -> V:
    if not isinstance(value, expected):
        raise ConfigValidationError(f"{key_path}: {expected.__name__} が必要だが {value!r} だった")
    return value
