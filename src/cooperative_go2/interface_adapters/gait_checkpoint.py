"""学習の出力先（outputs/gait/<run-name>/）から、学習済みの歩行方策を推論器として読み込む。"""

from pathlib import Path

import torch

from cooperative_go2.interface_adapters.config_loader import load_gait_settings
from cooperative_go2.interface_adapters.gait_run_store import RUN_CONFIG_FILENAME
from cooperative_go2.interface_adapters.rsl_rl_gait import load_rsl_rl_gait_policy
from cooperative_go2.use_cases.gait_controller import GaitController


def checkpoint_path(run_dir: Path, iteration: int) -> Path:
    """学習の出力先に保存された、反復番号 iteration のチェックポイント（model_<反復>.pt）のパスを返す。"""
    return run_dir / f"model_{iteration}.pt"


def load_gait_controller(
    run_dir: Path, iteration: int, num_envs: int, device: torch.device | str
) -> GaitController:
    """学習時の設定の写しとチェックポイントを読み込み、推論器を作る。シミュレータは使わない。

    Args:
        run_dir: 学習の出力先。設定の写し（gait.yaml）とチェックポイントがあること。
        iteration: 読み込むチェックポイントの反復番号。
        num_envs: 同時に推論するロボットの数。
        device: 方策と推論器の状態を置くデバイス。

    Raises:
        FileNotFoundError: 設定の写しが無い。
        ConfigValidationError: 設定の写しが現在の設定の形式に合わない。
        CheckpointNotFoundError: チェックポイントが無い。
        CheckpointLoadError: チェックポイントを読み込めない、またはネットワークの構成が設定と合わない。
    """
    settings = load_gait_settings(run_dir / RUN_CONFIG_FILENAME)
    policy = load_rsl_rl_gait_policy(checkpoint_path(run_dir, iteration), settings.rsl_rl, device)
    return GaitController(policy, settings.gait, num_envs, device)
