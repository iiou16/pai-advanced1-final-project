"""歩行学習 1 回分の出力先をファイルシステム上に用意する。"""

import shutil
from pathlib import Path

from cooperative_go2.use_cases.train_gait import RunAlreadyExistsError

# 出力先に複製する設定ファイルの名前。評価時はこのファイルを読み、学習時と同じ設定で環境を作る
RUN_CONFIG_FILENAME = "gait.yaml"


class InvalidRunNameError(ValueError):
    """学習の出力先の名前が、出力先のルートの直下の 1 つのディレクトリ名として使えないことを表す。"""


def resolve_run_dir(runs_root: Path, run_name: str) -> Path:
    """出力先のルートの直下に、run_name という名前の出力先のパスを作る。ディレクトリはまだ作らない。

    Raises:
        InvalidRunNameError: run_name が空、`.` / `..`、またはパスの区切りを含む（ルートの外を指しうる）。
    """
    if run_name in ("", ".", "..") or Path(run_name).name != run_name or "\\" in run_name:
        raise InvalidRunNameError(f"出力先の名前は 1 つのディレクトリ名にする: {run_name!r}")
    return runs_root / run_name


class FileGaitRunStore:
    """出力先のディレクトリを作り、学習に使った設定ファイルを複製する。"""

    def __init__(self, run_dir: Path, config_source: Path) -> None:
        """出力先と、複製する設定ファイルを受け取る。ディレクトリは initialize() まで作らない。

        Args:
            run_dir: 出力先のディレクトリ。まだ存在してはならない。
            config_source: 学習に使う設定ファイル。読み込み・検証済みであること。
        """
        self._run_dir = run_dir
        self._config_source = config_source

    def initialize(self) -> None:
        """出力先を作り、設定ファイルを複製する。

        Raises:
            RunAlreadyExistsError: 出力先が既にある。過去の結果を上書きしないため止める。
        """
        try:
            self._run_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise RunAlreadyExistsError(f"出力先が既にある（上書きしない）: {self._run_dir}") from error
        shutil.copyfile(self._config_source, self._run_dir / RUN_CONFIG_FILENAME)
