#!/usr/bin/env bash
# ros2_ws の ament パッケージ（独自の msg / action）をビルドする。リポジトリのルートで実行する。
#   bash scripts/setup/build_ros2_ws.sh
# 学習用 venv の python が PATH の先頭にあると、rosidl のコード生成がシステムの python3 の
# モジュール（empy など）を見つけられずに失敗するため、venv を PATH から外してからビルドする。
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
system_path="$(printf '%s' "${PATH}" | tr ':' '\n' | grep -v '/\.venv/' | paste -sd:)"

cd "${repo_root}/ros2_ws"
env -u VIRTUAL_ENV PATH="${system_path}" bash -c 'source /opt/ros/jazzy/setup.bash && colcon build "$@"' _ "$@"
