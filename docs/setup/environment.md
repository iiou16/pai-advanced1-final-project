# 開発環境のセットアップ

本リポジトリの動作確認環境と、同じ環境を再現する手順。

## 1. 動作確認環境

| 項目 | 値 | 備考 |
|---|---|---|
| OS | Ubuntu 24.04.2 LTS on WSL2（kernel 6.18.54.1-microsoft-standard-WSL2） | |
| GPU | NVIDIA GeForce RTX 2080 Ti（11 GB、Compute Capability 7.5 / sm_75） | ドライバ 630.20（CUDA 13.4 対応）。WSL 内に Linux ドライバは入れない |
| CPU | 8 コア（Intel Core i7-9700K） | |
| RAM | ホスト 48 GB。WSL 既定では 23 GB しか見えないため、Windows 側の `%USERPROFILE%\.wslconfig` で `memory=47GB` を設定 | 反映には `wsl --shutdown` 後の再起動が必要 |
| WSLg | 1.0.79 | 対話 Viewer の表示に使う |
| ffmpeg | あり | 動画の変換・確認用 |
| nvcc | なし | PyTorch / Genesis は wheel 同梱の CUDA ランタイムを使うため不要 |

## 2. ディレクトリ配置

Windows ドライブ（`/mnt/<drive>/...`、9p/drvfs）はファイル I/O が遅いため、すべて WSL の Linux ファイルシステム上に置く。

| 対象 | 場所 |
|---|---|
| リポジトリ | `~/workspace/phisicalAI-advanced1` |
| 学習用 venv | リポジトリ直下の `.venv`（約 8 GB、uv の既定） |
| Genesis 公式ソース（examples 参照用） | `~/src/genesis-world` |
| Genesis のカーネルキャッシュ | `~/.cache/genesis` |

## 3. Python 環境（uv）

- Python: **システムの 3.12.3**（`/usr/bin/python3.12`）。ROS 2 Jazzy の `rclpy` が apt のシステム Python 向けにビルドされているため、uv 管理の Python は使わない（`pyproject.toml` の `python-preference = "only-system"`）。
- 依存は `pyproject.toml` で宣言し、`uv.lock` で固定する。
- torch は PyTorch 公式 index の CUDA 13.2 ビルドを使う。RTX 2080 Ti の `sm_75` を含むビルドであるため。

| パッケージ | バージョン |
|---|---|
| torch | 2.15.0+cu132 |
| genesis-world | 1.4.5 |
| quadrants（Genesis の計算バックエンド） | 1.3.3 |
| rsl-rl-lib | 5.5.2 |
| tensorboard | 2.21.0 |
| tensordict | 0.15.4 |
| pyyaml | 6.0.3 |
| numpy | 2.5.4 |
| pytest（開発用、`dev` グループ） | 9.2.0 |

Genesis 公式ソースは pip 版と同じタグ **`v1.4.5`（commit `1b6dfa927f3836d49cf80df7a024116499e68d15`）** を使う。
`examples/locomotion/go2_train.py` は `rsl-rl-lib>=5.0.0` を要求しており、上記と整合する。

### セットアップ

```bash
# 依存の導入（リポジトリのルートで）
source scripts/setup/env.sh
uv sync

# Genesis 公式 examples の取得（タグを pip 版に合わせる）
git clone --depth 1 --branch v1.4.5 https://github.com/Genesis-Embodied-AI/genesis-world.git ~/src/genesis-world
```

`uv sync` は開発用の `dev` グループ（pytest）も入れ、本リポジトリの `src/cooperative_go2` を editable で導入する。

`scripts/setup/env.sh` は `LD_LIBRARY_PATH` に `/usr/lib/wsl/lib` を追加する。**Genesis を GPU で動かすシェルでは毎回 source が必要。**

- 理由: WSL のバージョン無し `libcuda.so` は ldconfig のキャッシュに無い。そのため、これが無いと Genesis（quadrants）が `libcuda.so lib not found` となり、`RuntimeError: Arch=[Arch.cuda] is not supported` で停止する。PyTorch は `libcuda.so.1` を読むため影響を受けない。

## 4. ROS 2 Jazzy

- 導入: `bash scripts/setup/install_ros2_jazzy.sh`（sudo が必要。ros2_documentation の Jazzy / Ubuntu deb 手順に準拠。`ros-jazzy-desktop` と `ros-dev-tools` を入れる）。
- 前提: ロケールが UTF-8 であること、apt の Suites に `noble-updates` と `noble-backports` が設定されていること。
- 公式手順にある「導入前の `apt upgrade`」はスクリプトに含めていない。`apt install ros-jazzy-desktop` が依存関係の衝突で失敗した場合は、まず `sudo apt update && sudo apt upgrade` を実行する。
- 学習用 venv の NumPy は 2.x、apt 版の msg 型はシステムの NumPy 1.26 向けにビルドされている。上記の版の組み合わせでは、venv から固定長配列を含む msg 型を変換できることを `scripts/p0_env_check/ros2_smoke.sh` で確認している。版を更新して NumPy の ABI 不一致で失敗するようになった場合は、`pyproject.toml` に `numpy<2` を追加する。

| パッケージ | バージョン |
|---|---|
| ros-jazzy-desktop | 0.11.0 |
| ros-jazzy-rclpy | 7.1.12 |
| ros-dev-tools | 1.0.3 |
| ros-jazzy-rosbag2-storage-mcap | 0.26.11 |
| python3-colcon-common-extensions | 0.3.0 |
| python3-empy | 3.3.4 |

上の後 3 つは `ros-jazzy-desktop` と `ros-dev-tools` の依存として入る（rosbag2 の mcap 形式の記録、`ros2_ws` のビルド、独自の msg / action のコード生成に使う）。

### 独自の msg / action（ros2_ws）のビルド

ROS 2 を経由した実行（README の「ROS 2 を経由した箱押し」）の前に、`ros2_ws/src/go2_interfaces` をビルドする。

```bash
bash scripts/setup/build_ros2_ws.sh
```

- 学習用 venv を有効にしたシェルでそのまま `colcon build` すると、venv の python3 が使われ、rosidl のコード生成がシステムの python3 のモジュール（empy）を見つけられずに `ModuleNotFoundError: No module named 'em'` で止まる。スクリプトは venv を PATH から外してビルドする。
- ビルド後は `/opt/ros/jazzy/setup.bash` に続けて `ros2_ws/install/setup.bash` を source すると、venv から `go2_interfaces` の型を import できる。
- ROS 2 を source したシェルで pytest を動かすと、apt の `launch_testing` が pytest のプラグインとして読み込まれ、venv に無い依存（`lark`）で止まる。`pyproject.toml` の `addopts` で読み込まないようにしている。

## 5. 動作確認

セットアップ後、`scripts/p0_env_check/` のスクリプトで環境を確認する（`p0` は、学習や ROS 2 連携の実装に入る前の最初の段階に行う確認であることを示す）。
いずれもリポジトリのルートで実行し、検証に通れば最後に `RESULT: PASS` を出力する。失敗時は非ゼロで終了する。
Viewer の確認だけは、ウィンドウを生成して描画できたことまでを判定する。表示内容は目視で確かめる。

| 確認内容 | コマンド |
|---|---|
| PyTorch の CUDA 演算 | `uv run --frozen python scripts/p0_env_check/torch_cuda_smoke.py` |
| Genesis の剛体シミュレーション（GPU） | `uv run --frozen python scripts/p0_env_check/genesis_rigid_smoke.py` |
| オフスクリーン録画（`outputs/p0_env_check/offscreen_record.mp4` を出力し、フレーム数・解像度と映像の変化を検証） | `uv run --frozen python scripts/p0_env_check/genesis_offscreen_record.py` |
| 対話 Viewer（WSLg。ウィンドウに箱の落下が表示されることを目視する） | `uv run --frozen python scripts/p0_env_check/genesis_viewer_check.py` |
| ROS 2 の Topic / Action と、venv からの rclpy・torch の同時利用 | `bash scripts/p0_env_check/ros2_smoke.sh` |

Genesis を使うスクリプトは、事前に `source scripts/setup/env.sh` を実行したシェルで動かす。
初回はカーネルのコンパイルに数分かかる。
