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

## 歩行方策の学習と評価

Go2 の歩行方策（速度指令 `vx, vy, yaw_rate` → 12 関節の目標位置）を PPO で学習する。設定は `configs/gait.yaml` にまとめ、読み込み時にキーの過不足・型・値の範囲を検証する。

```bash
source scripts/setup/env.sh

# 学習（出力: outputs/gait/<run-name>/。同名のディレクトリがあれば上書きせずに止まる）
uv run --frozen go2-train-gait --config configs/gait.yaml --run-name gait-baseline

# 評価（指令を固定して追従誤差と転倒率を JSON で出力。--video で評価動画を撮る）
uv run --frozen go2-evaluate-gait --run-dir outputs/gait/gait-baseline --checkpoint 100 \
    --vx 0.5 --vy 0.0 --yaw-rate 0.0 --num-envs 1 --duration-s 10 --seed 1001 \
    --video outputs/gait/gait-baseline/eval_forward.mp4
```

- 評価のシードは学習のシード（`training.seed`）と別の値にする。同じ値を指定するとエラーで止まる。
- 評価の時間はエピソードの長さ（`env.episode_length_s`）以内にする。
- 現状の評価は決定的で、全環境が同じ初期状態から始まるため、シードや環境数を変えても結果は変わらない。
- 学習ログ（TensorBoard）には、報酬の項ごとの 1 秒あたりの値と、エピソードの終了理由（時間切れ・転倒・物理計算の発散）の割合が出る。

## テスト

```bash
uv run --frozen pytest
```

## 参考にしたもの

| 対象 | 出典 | ライセンス | 本リポジトリでの扱い |
|---|---|---|---|
| Go2 の歩行学習の環境・報酬・PPO の設定 | Genesis 公式の例 `examples/locomotion/`（[genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) v1.4.5） | Apache-2.0 | コードは複製せず、観測・報酬の構成と初期値を参考に `src/cooperative_go2/` で実装した |
| Go2 の URDF・平面の URDF | genesis-world パッケージ同梱のアセット | 同梱物に個別の表記なし（原典は提出前に確認する） | パッケージから実行時に読み込む。本リポジトリには含めない |
