"""YAML の設定ファイルを読み込み、型付きの設定オブジェクトに変換して検証する。

キーの過不足と値の型を厳密に確かめる。既定値で補わないため、設定ファイルにはすべてのキーを書く。
"""

from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints

import yaml

from cooperative_go2.entities.gait_config import GaitConfig, InvalidGaitConfigError
from cooperative_go2.interface_adapters.rsl_rl_config import InvalidRslRlConfigError, RslRlPpoConfig

_ROOT_KEY_PATH = "<root>"
# 歩行の設定ファイルのうち、RSL-RL 固有の設定を書く節の名前
RSL_RL_SECTION = "rsl_rl"


class ConfigValidationError(ValueError):
    """設定ファイルの内容が不正（キーの過不足、型の不一致、値の不変条件の違反）であることを表す。

    メッセージには不正な項目のキーのパス（例: command.lin_vel_x_mps.low）を含める。
    """


@dataclass(frozen=True)
class GaitSettings:
    """歩行の設定ファイル 1 つ分。gait は内側の層に渡し、rsl_rl は RSL-RL との接続にだけ使う。"""

    gait: GaitConfig
    rsl_rl: RslRlPpoConfig


def load_gait_settings(path: Path) -> GaitSettings:
    """歩行の設定ファイル（configs/gait.yaml 形式）を読み込み、検証済みの設定を返す。

    Args:
        path: 設定ファイル。相対パスは現在の作業ディレクトリ起点。

    Raises:
        FileNotFoundError: ファイルが無い。
        ConfigValidationError: YAML として読めない、またはキーの過不足・型・値が不正。
    """
    text = path.read_text(encoding="utf-8")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ConfigValidationError(f"{path}: YAML として読めない: {error}") from error
    return parse_gait_settings(data)


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
    # 不変条件の違反は、どちらの設定でも「設定ファイルを直すべき」点で同じなので同じ例外に変換する
    try:
        return cls(**kwargs)
    except (InvalidGaitConfigError, InvalidRslRlConfigError) as error:
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
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigValidationError(f"{key_path}: 数値が必要だが {value!r} だった")
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
