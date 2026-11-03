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

Go2 の歩行方策（速度指令 `vx, vy, yaw_rate` → 12 関節の目標位置）を PPO で学習する。前進だけでなく、停止・旋回・後退・横移動の指令にも追従させる。学習の設定は `configs/gait.yaml`、評価シナリオは `configs/gait_eval.yaml` にまとめ、どちらも読み込み時にキーの過不足・型・値の範囲を検証する。コマンドはリポジトリのルートで実行する。

```bash
source scripts/setup/env.sh

# 学習（出力: outputs/gait/<run-name>/。同名のディレクトリがあれば上書きせずに止まる）
uv run --frozen go2-train-gait --config configs/gait.yaml --run-name gait-command-tracking

# 評価（出力: outputs/gait/<run-name>/eval/model_<checkpoint>_<シナリオのファイル名>_envs_<num-envs>_seed_<seed>/。
#       --video でシナリオごとに評価動画を撮る）
uv run --frozen go2-evaluate-gait --run-dir outputs/gait/gait-command-tracking --checkpoint 1500 \
    --scenarios configs/gait_eval.yaml --num-envs 64 --seed 1001 --video
```

- 評価の結果は `segments.csv`（シナリオの区間ごとの追従誤差・転倒率）と `summary.json`（条件と全結果）に保存し、シナリオのファイルの写しも置く。`summary.json` は最後に書くので、これがあれば保存は最後まで終わっている。保存先が既にあれば上書きせずに止まる。
- 評価のシードは学習のシード（`training.seed`）と別の値にする。同じ値を指定するとエラーで止まる。環境ごとにリセット時の関節角がシードに従ってばらつくので、シードを変えると結果も変わる。乱数はシナリオを順に走らせながら引くため、シナリオを追加・並べ替えると、同じシードでも他のシナリオの初期姿勢が変わる。
- 区間の長さと `settle_time_s` は制御周期（`env.control_dt_s`）の整数倍に、各シナリオの長さはエピソードの長さ（`env.episode_length_s`）以内に、指令は学習時の範囲（`command` 節）の中にする。外れるとエラーで止まる。
- 追従誤差の集計からは、指令を切り替えた直後の過渡（`settle_time_s`）と、転倒した環境が初期姿勢から立ち直るまでの同じ長さの過渡を除く。転倒は除かずに数える。全環境が転倒し続けて集計できなかった区間の追従誤差は `null`（CSV では空欄）になる。
- 学習ログは `uv run --frozen tensorboard --logdir outputs/gait` で見る。報酬の項ごとの 1 秒あたりの値（`rew_*`）と、エピソードの終了理由（時間切れ・転倒・物理計算の発散）の割合（`termination_*`）が出る。どちらも、1 反復の中でエピソードが終わったステップごとの値を単純に平均したもの。
- シーンの構築や学習・評価の途中で失敗した場合も出力先は残る。同じ名前では再実行できないので、出力先を削除するか別の名前を使う。
- 合格基準: `configs/gait_eval.yaml` の停止（`stop`）・前進（`forward`）・左右のその場旋回（`turn_in_place_left` / `turn_in_place_right`）の各区間で、水平速度の追従誤差の平均が 0.10 m/s 以下、ヨー角速度の追従誤差の平均が 0.15 rad/s 以下、転倒率が 5 % 以下であること。学習に使っていないシードで測った実測値（それぞれ最大 0.032 m/s・0.038 rad/s・0 %）に約 3 倍の余裕を持たせた値。
- 学習済みの方策を他の処理から使うときは、`interface_adapters.gait_checkpoint.load_gait_controller` で推論器（`GaitController`）を読み込む。自己受容の状態と速度指令から、12 関節の目標角（関節の順序は `GO2_JOINT_NAMES`）を返す。

## 学習済みモデルの取得

学習済みの歩行方策と箱押しの方策は、GitHub の Release（タグ `v0.2.0`）で配布している。学習をせずに評価だけを再現するときは、次の手順で `outputs/` に置く。

```bash
# Release の添付ファイルを取得する（gh を使わない場合は Release のページから同じファイルをダウンロードする）
gh release download v0.2.0 -R iiou16/pai-advanced1-final-project -p '*.tar.gz' -D outputs/release

# 歩行方策（model_1500.pt と学習時の設定 gait.yaml）→ outputs/gait/gait-command-tracking/
mkdir -p outputs/gait outputs/push
tar -xzf outputs/release/gait-command-tracking.tar.gz -C outputs/gait

# 箱押しの方策（model_200.pt と学習時の設定 push_single.yaml・scene.yaml）→ outputs/push/push-stage2-wide60/
tar -xzf outputs/release/push-stage2-wide60.tar.gz -C outputs/push
```

- 歩行方策は `configs/push_single.yaml` の `gait_policy`（`outputs/gait/gait-command-tracking` の `model_1500.pt`）から読み込む。
- 箱押しの方策は、下の評価のコマンドで `--policy-run-dir outputs/push/push-stage2-wide60 --checkpoint 200` を指定して使う。

## 箱押しの高レベル方策の学習と評価

学習済みの歩行方策を固定し、その上で 1 台の Go2 が軽量の箱（0.4 × 0.4 × 0.3 m、2 kg）を目標領域へ押す高レベル方策を PPO で学習する。高レベル方策は 10 Hz で速度指令 `[vx, vy, yaw_rate]` を出し、その 1 ステップの間に歩行方策を 50 Hz で 5 回呼ぶ。設定は `configs/push_single.yaml`（学習環境・規則ベースの制御・PPO）と `configs/scenes/s1_push.yaml`（箱と物理エンジンの設定）にまとめ、読み込み時に検証する。歩行方策は `gait_policy.run_dir` の学習結果を使う（歩行方策の学習を先に済ませておく）。

```bash
source scripts/setup/env.sh

# シーンの物理の確認（箱が置かれる・静止摩擦で止まる・押すと滑る・離すと止まる・床との接触が途切れない。CPU で動く）
uv run --frozen python scripts/p2_scene_check/check_s1_push_scene.py

# 学習（出力: outputs/push/<run-name>/。チェックポイント（50 反復ごと）・TensorBoard のログ・設定の写し）
#   目標が箱の正面にある配置（configs/push_single.yaml）と、目標の方向を ±60° に広げた配置
#   （configs/push_single_wide.yaml）の 2 通りがある。配布しているモデルは後者の 200 反復目（model_200.pt）
uv run --frozen go2-train-push --config configs/push_single_wide.yaml --run-name push-stage2-wide60

# 評価（出力: outputs/push_eval/<run-name>/。--video で環境 0 を追従した評価動画も撮る）
#   規則ベースの制御（比較の基準）
uv run --frozen go2-evaluate-push --config configs/push_single_wide.yaml --algorithm rule_based \
    --num-envs 64 --seed 2001 --device cuda --run-name e1-wide60-seed2001
#   学習済みの方策
uv run --frozen go2-evaluate-push --config configs/push_single_wide.yaml --algorithm policy \
    --policy-run-dir outputs/push/push-stage2-wide60 --checkpoint 200 \
    --num-envs 64 --seed 2001 --device cuda --run-name e2-wide60-seed2001

# 学習曲線を CSV と PNG に書き出す
uv run --frozen python scripts/p2_materials/export_training_curves.py outputs/push/push-stage2-wide60 outputs/materials/training
```

- 評価の配置は `--config` で選ぶ。目標が正面なら `configs/push_single.yaml`、±60° なら `configs/push_single_wide.yaml`。評価のシードは 2001・2002・2003 を使った（学習のシードとは別）。

- 評価は並列環境ごとに最初の 1 試行だけを数える（早く終わる試行ほど多く数える偏りを避けるため）。結果は試行ごとに 1 行の `trials.csv`（成功・終了理由・所要時間・箱の移動量と回転・転倒・物理ステップ数・実時間など）と、条件と評価指標（成功率・目標方向の移動量・最終目標距離・箱の回転角・成功までの時間・転倒率）の `summary.json` に保存する。
- 学習済みの方策を評価するときは、観測の正規化係数・高レベル方策の周期・指令の範囲・歩行方策が学習時と同じでなければ止まる。配置（目標の距離・方向）や成功判定は、汎化を確かめるために変えてよい。
- 成功は、箱の中心が目標領域（半径 0.3 m の円）に入った状態を 1.0 秒続けること。終了理由は、成功・時間切れ（20 秒）・転倒・シーン外への離脱・物理計算の発散のいずれか。
- 箱を置くシーンでは、物理エンジンの摩擦円錐を `elliptic` にしている。Genesis の既定の `pyramidal` では、滑っている箱に床から押し上げる誤った力が生じ、箱が浮いたり飛んで転がったりする。歩行方策は `pyramidal` で学習したものだが、`elliptic` でも合格基準を満たすことを確かめている。比較の動画は `scripts/p2_scene_check/record_friction_cone_demo.py` で撮れる。
- 学習ログは `uv run --frozen tensorboard --logdir outputs/push` で見る。

## テスト

```bash
uv run --frozen pytest
```

## 参考にしたもの

| 対象 | 出典 | ライセンス | 本リポジトリでの扱い |
|---|---|---|---|
| Go2 の歩行学習の環境・報酬・PPO の設定 | Genesis 公式の例 `examples/locomotion/`（[genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) v1.4.5） | Apache-2.0 | コードは複製せず、観測・報酬の構成と初期値を参考に `src/cooperative_go2/` で実装した |
| Go2 の URDF・平面の URDF | genesis-world パッケージ同梱のアセット | 同梱物に個別の表記なし（原典のライセンスは未確認） | パッケージから実行時に読み込む。本リポジトリには含めない |
