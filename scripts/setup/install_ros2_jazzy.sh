#!/usr/bin/env bash
# ROS 2 Jazzy を Ubuntu 24.04 (WSL2) に apt で導入する。
# 手順は ros2_documentation (jazzy ブランチ) の Ubuntu-Install-Debs に従う。
# sudo のパスワード入力が必要なため、利用者自身のターミナルで実行する:
#   bash scripts/setup/install_ros2_jazzy.sh
set -euo pipefail

readonly ROS_DISTRO_NAME="jazzy"
readonly EXPECTED_CODENAME="noble"
readonly REQUIRED_PACKAGES=("ros-${ROS_DISTRO_NAME}-desktop" ros-dev-tools)
# ROS の apt リポジトリと署名鍵を登録する deb。root で導入するため、動作確認した版に固定し、
# GitHub リリースが公開している SHA-256 と照合してから導入する
readonly ROS_APT_SOURCE_VERSION="1.3.0"
readonly ROS_APT_SOURCE_SHA256="f31d84adf5054c7d60ded0e82c0f776a77ea33af53d08f40b9ad7c94cca55296"

codename="$(. /etc/os-release && echo "${UBUNTU_CODENAME:-${VERSION_CODENAME}}")"
if [[ "${codename}" != "${EXPECTED_CODENAME}" ]]; then
  echo "ERROR: ROS 2 ${ROS_DISTRO_NAME} の deb は Ubuntu ${EXPECTED_CODENAME} 向け。検出: ${codename}" >&2
  exit 1
fi

# setup.bash は依存パッケージから先に配置されるため、途中で失敗した導入を完了と誤認しないようパッケージで判定する
is_installed() {
  [[ "$(dpkg-query -W -f='${Status}' "$1" 2>/dev/null)" == "install ok installed" ]]
}
all_installed=true
for package in "${REQUIRED_PACKAGES[@]}"; do
  is_installed "${package}" || all_installed=false
done
if [[ "${all_installed}" == true ]]; then
  echo "ROS 2 ${ROS_DISTRO_NAME} は導入済み: ${REQUIRED_PACKAGES[*]}"
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

download_dir="$(mktemp -d)"
trap 'rm -rf "${download_dir}"' EXIT
apt_source_deb="${download_dir}/ros2-apt-source.deb"
curl -fL -o "${apt_source_deb}" \
  "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.${codename}_all.deb"
sha256sum --check --quiet <<<"${ROS_APT_SOURCE_SHA256}  ${apt_source_deb}" || {
  echo "ERROR: ros2-apt-source ${ROS_APT_SOURCE_VERSION} の SHA-256 が一致しない。導入を中止する。" >&2
  exit 1
}
sudo dpkg -i "${apt_source_deb}"

sudo apt update
# desktop は RViz と demo_nodes を含み、scripts/p0_env_check/ros2_smoke.sh の Topic / Action 疎通確認に使う
sudo apt install -y "${REQUIRED_PACKAGES[@]}"

echo "完了。確認: source /opt/ros/${ROS_DISTRO_NAME}/setup.bash && ros2 doctor --report | head"
