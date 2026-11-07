"""skrl で学習した協調の Actor を、推論用に読み込む。シミュレータ（Genesis）を読み込まない。

評価と ROS 2 の協調のノードが、学習時と同じネットワーク構成で Actor を組み立てられるよう、
学習の出力先に残した設定の写し（push_multi.yaml）と学習方式の記録（variant.json）から読む。
"""

import json
import pickle
from pathlib import Path

import torch

from cooperative_go2.interface_adapters.config_loader import TeamMarlRunSettings, load_team_marl_run_settings_with_scene
from cooperative_go2.interface_adapters.rsl_rl_policy import CheckpointLoadError, CheckpointNotFoundError
from cooperative_go2.interface_adapters.skrl_marl_config import MarlAlgorithm, MarlVariant
from cooperative_go2.interface_adapters.skrl_team import POLICY_ROLE, TeamPolicy

TEAM_RUN_CONFIG_FILENAME = "push_multi.yaml"
TEAM_RUN_SCENE_CONFIG_FILENAME = "scene.yaml"
TEAM_RUN_VARIANT_FILENAME = "variant.json"
TEAM_RUN_COMMIT_FILENAME = "git_commit.txt"
CHECKPOINTS_DIRNAME = "checkpoints"


class InvalidRunRecordError(ValueError):
    """学習の出力先に残した記録（学習方式など）が、読める形式でないことを表す。"""


def write_variant(run_dir: Path, variant: MarlVariant) -> None:
    """学習方式の記録を run_dir に書く。"""
    record = {"algorithm": variant.algorithm.value, "shared_parameters": variant.shared_parameters}
    (run_dir / TEAM_RUN_VARIANT_FILENAME).write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def load_variant(run_dir: Path) -> MarlVariant:
    """run_dir に書いた学習方式の記録を読む。

    Raises:
        FileNotFoundError: 記録が無い。
        InvalidRunRecordError: 記録が JSON として読めない、キーが足りない、または値が決めた形でない。
    """
    path = run_dir / TEAM_RUN_VARIANT_FILENAME
    text = path.read_text(encoding="utf-8")
    try:
        record = json.loads(text)
        algorithm = MarlAlgorithm(record["algorithm"])
        shared_parameters = record["shared_parameters"]
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        # JSON として読めない・キーが無い・辞書でない・学習方式の名前が無いものは、いずれも記録の形式の誤り
        raise InvalidRunRecordError(f"学習方式の記録を読めない: {path}: {error}") from error
    if not isinstance(shared_parameters, bool):
        raise InvalidRunRecordError(f"{path}: shared_parameters は true か false: {shared_parameters!r}")
    return MarlVariant(algorithm=algorithm, shared_parameters=shared_parameters)


def load_team_run_settings_copy(run_dir: Path) -> TeamMarlRunSettings:
    """学習の出力先に残した設定とシーンの写しを読む。

    作業ツリーのシーンの設定（写しの scene_config が指すファイル）は読まない。学習後にシーンの設定が変わっても、
    学習時のシーンで方策を評価・実行するため。

    Raises:
        FileNotFoundError: 設定かシーンの写しが無い。
        ConfigValidationError: 写しが現在の設定の形式に合わない。
    """
    return load_team_marl_run_settings_with_scene(
        run_dir / TEAM_RUN_CONFIG_FILENAME, run_dir / TEAM_RUN_SCENE_CONFIG_FILENAME
    )


def team_checkpoint_path(run_dir: Path, iteration: int, rollouts: int) -> Path:
    """更新 iteration 回目のあとに保存したチェックポイントのパス。skrl は集めたステップ数で名前を付ける。"""
    return run_dir / CHECKPOINTS_DIRNAME / f"agent_{iteration * rollouts}.pt"


def load_team_policy(run_dir: Path, iteration: int, agent_name: str, device: torch.device | str) -> TeamPolicy:
    """学習済みの Actor を推論用（評価モード）に読み込む。

    Args:
        run_dir: 協調学習の出力先。
        iteration: 読み込むチェックポイントの更新の回数（skrl.checkpoint_interval_updates の倍数）。
        agent_name: ロボットの名前（シーンの robots の名前）。パラメータを共有した学習では、どの名前でも同じ Actor。
        device: Actor を置くデバイス。

    Raises:
        FileNotFoundError: 設定かシーンの写しが無い。
        ConfigValidationError: 写しが現在の設定の形式に合わない。
        CheckpointNotFoundError: チェックポイントが無い。
        CheckpointLoadError: チェックポイントを読み込めない、そのロボットの Actor が無い、またはネットワークの構成が
            合わない。
    """
    settings = load_team_run_settings_copy(run_dir).settings
    path = team_checkpoint_path(run_dir, iteration, settings.skrl.rollouts)
    if not path.is_file():
        raise CheckpointNotFoundError(f"チェックポイントが無い: {path}")
    # skrl のモデルは作った時点ではパラメータを device に置かない（学習ではエージェントが移す）ので、ここで移す
    policy = TeamPolicy(settings.skrl, device).to(device)
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        policy.load_state_dict(checkpoint[agent_name][POLICY_ROLE])
    except (torch.OutOfMemoryError, torch.AcceleratorError):
        # メモリ不足と GPU の実行時エラーは RuntimeError の一種だが、チェックポイントの不良ではないのでそのまま伝える
        raise
    except (pickle.UnpicklingError, EOFError, KeyError, TypeError, RuntimeError) as error:
        # 中身が壊れている（UnpicklingError）、空や途中で切れている（EOFError）、そのロボットの Actor が無いか別の構造
        # （KeyError・TypeError）、ネットワークの構成が合わない（RuntimeError）は、いずれもこのチェックポイントを
        # 使えない点で同じ扱いにする
        raise CheckpointLoadError(f"{path} から {agent_name} の Actor を読み込めない: {error}") from error
    policy.eval()
    return policy
