# PAI Final Project

Physical AI 応用編1 知能ロボティクス講座（2026 Summer）最終課題のリポジトリです。

課題テーマ: **AI×ロボティクスによる創造的タスクの実現**

> 作業中です。タスク内容・実行方法は実装に合わせて追記します。

## セットアップ

動作確認環境は Ubuntu 24.04（WSL2）+ NVIDIA GPU です。Python 環境は uv、ロボットとの連携には ROS 2 Jazzy を使います。

手順の詳細と、動作を確認したバージョンは [docs/setup/environment.md](docs/setup/environment.md) にまとめています。概要は次のとおりです。

```bash
# 1. Python 依存の導入（リポジトリのルートで）
source scripts/setup/env.sh
uv sync

# 2. ROS 2 Jazzy の導入（sudo が必要）
bash scripts/setup/install_ros2_jazzy.sh

# 3. 環境の動作確認（例: Genesis の剛体シミュレーション）
uv run --frozen python scripts/p0_env_check/genesis_rigid_smoke.py
```

Genesis を GPU で動かすシェルでは、毎回 `source scripts/setup/env.sh` が必要です。
動作確認スクリプトの一覧は、手順書の「動作確認」の節を参照してください。
