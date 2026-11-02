#!/usr/bin/env bash
# ROS 2 Jazzy を Ubuntu 24.04 (WSL2) に apt で導入する。
# 手順は ros2_documentation (jazzy ブランチ) の Ubuntu-Install-Debs に従う。
# sudo のパスワード入力が必要なため、利用者自身のターミナルで実行する:
#   bash scripts/setup/install_ros2_jazzy.sh
set -euo pipefail

readonly ROS_DISTRO_NAME="jazzy"
readonly EXPECTED_CODENAME="noble"

codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"
if [[ "${codename}" != "${EXPECTED_CODENAME}" ]]; then
  echo "ERROR: ROS 2 ${ROS_DISTRO_NAME} の deb は Ubuntu ${EXPECTED_CODENAME} 向け。検出: ${codename}" >&2
  exit 1
fi

if [[ -f "/opt/ros/${ROS_DISTRO_NAME}/setup.bash" ]]; then
  echo "ROS 2 ${ROS_DISTRO_NAME} は導入済み: /opt/ros/${ROS_DISTRO_NAME}"
  exit 0
fi

# ROS 2 は UTF-8 ロケールを前提とする
if ! locale | grep -qi 'utf-8'; then
  echo "ERROR: UTF-8 ロケールではない。公式手順の locale-gen を先に実施すること。" >&2
  exit 1
fi

sudo apt install -y software-properties-common
sudo add-apt-repository -y universe

sudo apt update
sudo apt install -y curl

# set -e のもとでパイプラインの途中が失敗すると診断を出さずに終了するため、取得と解析を分けて失敗を個別に報告する
readonly ROS_APT_SOURCE_RELEASE_API="https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest"
release_json="$(curl -fsS "${ROS_APT_SOURCE_RELEASE_API}")" || {
  echo "ERROR: ros-apt-source の最新リリース情報を取得できない: ${ROS_APT_SOURCE_RELEASE_API}" >&2
  exit 1
}
ros_apt_source_version="$(grep -F '"tag_name"' <<<"${release_json}" | awk -F'"' '{print $4}')" || {
  echo "ERROR: リリース情報に tag_name が無い (GitHub API のレート制限の可能性):" >&2
  echo "${release_json}" >&2
  exit 1
}

download_dir="$(mktemp -d)"
trap 'rm -rf "${download_dir}"' EXIT
apt_source_deb="${download_dir}/ros2-apt-source.deb"
curl -fL -o "${apt_source_deb}" \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ros_apt_source_version}/ros2-apt-source_${ros_apt_source_version}.${codename}_all.deb"
sudo dpkg -i "${apt_source_deb}"

sudo apt update
# desktop は RViz と demo_nodes を含み、scripts/p0_env_check/ros2_smoke.sh の Topic / Action 疎通確認に使う
sudo apt install -y "ros-${ROS_DISTRO_NAME}-desktop" ros-dev-tools

echo "完了。確認: source /opt/ros/${ROS_DISTRO_NAME}/setup.bash && ros2 doctor --report | head"
