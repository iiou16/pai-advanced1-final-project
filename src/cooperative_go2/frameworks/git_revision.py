"""評価の記録に残す、リポジトリのコミットの取得。"""

import subprocess


class GitCommitUnavailableError(RuntimeError):
    """評価の記録に残すコミットを git から取得できないことを表す。"""


def current_commit_sha() -> str:
    """現在のコミットの SHA を返す。未コミットの変更があれば末尾に -dirty を付ける。

    Raises:
        GitCommitUnavailableError: git を実行できない、またはリポジトリの中ではない。
    """
    # 未コミットの変更がある状態で評価したことを記録から見分けられるよう、その場合は末尾に -dirty を付ける
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise GitCommitUnavailableError(f"コミットを取得できない: {error}") from error
    return f"{sha}-dirty" if status.strip() else sha
