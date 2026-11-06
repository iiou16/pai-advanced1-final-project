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

## ROS 2 を経由した箱押し（推論だけを行う手順）

学習済みの歩行方策と箱押しの方策を、ROS 2 の別々のノードとして動かし、1 台の Go2 が箱を目標領域へ押す。学習はしない（学習済みモデルは上の「学習済みモデルの取得」で置く）。

| ノード（Namespace `/go2_a`） | 実行ファイル | 役割 |
|---|---|---|
| `genesis_sim` | `go2-ros2-sim` | Genesis のシーンを進める唯一のプロセス。`/clock`・`odom`・`joint_states`・`box_contact`・`/team/object_state`・TF を配信し、`joint_targets` を受けて PD 制御する。試行ごとに歩行方策をリセットし、高レベル方策に目標を `ClearCorridor` で依頼して、成功と終了を判定・記録する |
| `gait_policy` | `go2-ros2-gait` | `joint_states`・`odom`・`cmd_vel` から 12 関節の目標（`joint_targets`）を出す（50 Hz） |
| `coord_policy` | `go2-ros2-coord` | `odom`・`/team/object_state`・`box_contact` と目標から速度指令（`cmd_vel`）を出す（10 Hz） |

歩行方策と高レベル方策のノードは Genesis を読み込まないので、物理を進められない。受け手は古い指令（速度指令 0.3 s、関節目標 0.1 s。シミュレーション時間）を停止の指令に切り替えるので、方策のノードを止めるとロボットはその場で止まる。設定は `configs/ros2_push_single.yaml`。

```bash
# 準備（初回と ros2_ws の msg / action を変えたとき）。venv を外してビルドする
bash scripts/setup/build_ros2_ws.sh

# 実行するシェルで毎回 source する
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
source scripts/setup/env.sh

# 3 ノードと rosbag2 の記録を 1 コマンドで起動する。全試行を終えると自動で止まる
ros2 launch ros2_ws/launch/push_single.launch.py seed:=2001 num_trials:=64 run_name:=e2-ros2-wide60-seed2001 \
    device:=cuda video:=false record_bag:=true

# 方策のノードを止めるとロボットが止まることを確かめる（高レベル方策 → 歩行方策の順に止める）
uv run --frozen python scripts/p3_ros2_check/check_stop_on_node_loss.py --run-name p3-stop-check --seed 2002 \
    --device cuda --video true

# 記録の時刻が /clock と整合しているかを確かめる
uv run --frozen python scripts/p3_ros2_check/check_bag_clock.py outputs/ros2_bags/e2-ros2-wide60-seed2001
```

- 結果は `outputs/ros2_eval/<run_name>/` に、`go2-evaluate-push` と同じ形式の `trials.csv`・`summary.json`（`algorithm` は `policy_ros2`）と、
  ROS 2 の経路の計測値 `ros2_metrics.json`（適用した関節目標の遅れ・往復の遅延・実時間比）、各ノードの推論時間（`nodes/`）を保存する。
  rosbag2 は `outputs/ros2_bags/<run_name>/`（mcap）。保存先が既にあれば止まる。
  `ros2 launch` 自体の終了コードはノードの失敗を表さないので、成否は `summary.json` があるかで判断する（失敗したときはシミュレーションノードのログに理由が出る）。
- 全試行の配置と初期姿勢を学習環境の評価と同じ順で引くので、同じシード・`device:=cuda`・試行の数で `go2-evaluate-push --num-envs` を
  評価すると、試行ごとに同じ初期条件になる。
- `video:=true` で評価動画を撮ると、描画のため実時間比が 0.1 程度まで下がる（シミュレーション時間で進むので結果は変わらない）。
- 物理は方策の応答を待たずに制御周期ごとに進め、関節目標は学習時と同じく、元の状態から 1 制御周期遅れで適用する。
  一方、速度指令は高レベル方策のノードから通信を経て届くので、歩行方策が使うのは学習時より約 1 制御周期（0.02 秒）古い指令になり、
  試行の始めの 1 ステップは指令がまだ届かず停止の指令で歩く。歩行方策が指令を待つと関節目標が制御周期に間に合わなくなりやすいので、
  関節目標の遅れを学習時とそろえることを優先し、歩行方策は指令を待たない。
  各ノードは PyTorch の CPU のスレッドを 1 本にしている（`runtime.torch_num_threads`）。既定のスレッド数では、他のプロセスと
  CPU を取り合うと歩行方策の推論が制御周期（20 ms）を超えて遅れ、転倒が増えた。
- 学習環境での評価（目標の方向 ±60°、シード 2001・2002・2003、各 64 試行）と同じ初期条件で、ROS 2 経由でも成功率は 3 シードとも 100 %
  （CPU に負荷をかけた状態・かけない状態の両方）。成功までの時間は、学習環境での評価より平均 0.06 秒長い。

## ROS 2 を経由した 2 台の規則ベース協調（通路の開通）

壁で挟んだ通路（幅 1.4 m）をふさぐ重い箱（0.4 × 1.0 × 0.3 m、5 kg）を、Go2 が通路の出口の先の退避領域へ押し出して通路を開ける。
規則ベースの制御で、1 台（E3）・2 台で通信なし（E4）・2 台で協調（E5）を同じ条件で比べる。学習はしない（歩行方策は上の「学習済みモデルの取得」で置く）。

| ノード | 実行ファイル | 役割 |
|---|---|---|
| `genesis_sim` | `go2-ros2-team-sim` | Genesis のシーンを進める唯一のプロセス。各ロボットの `/go2_x/odom`・`joint_states`、`/team/object_state`・TF を配信し、各ロボットの `joint_targets` を受けて PD 制御する。試行ごとに開始位置を引き、タスク管理に通路の開通を依頼し、真値（通路の開通・箱が退避領域に入ったまま 1 s 続いたか・転倒）とタスク管理の結果を記録する |
| `task_manager` | `go2-ros2-task-manager` | `/task/clear_corridor`（`ClearCorridor` の Action）で依頼を受け、各ロボットの協調のノードへ割り振り、進捗を返し、成功・失敗・時間切れを判定する |
| `/go2_x/coord_policy` | `go2-ros2-team-coord` | 状態機械（INIT → APPROACH → ALIGN → READY → PUSH → VERIFY → DONE / FAILED）で判断し、`cmd_vel` と作業の状態（`/team/agent_status`）を配信する（10 Hz）。協調のモードでは相手の状態を受け、相手の準備を待って同時に押し始め、相手の状態が 0.5 s 途絶えたら・相手が押すのをやめたら押すのをやめて待つ |
| `/go2_x/gait_policy` | `go2-ros2-gait` | 1 台の箱押しと同じ歩行方策のノード |

設定は `configs/ros2_team.yaml`（2 台）・`configs/ros2_team_single.yaml`（1 台）と、それが指す `configs/team_rule_based*.yaml`・`configs/scenes/s3_*.yaml`。

E4 と E5 は、ロボットごとの制御を同じにし、ロボット同士の通信（`/team/agent_status`）を使うかどうかだけを変える（協調のノードの `mode`）。

| | E4（`mode:=independent`） | E5（`mode:=cooperative`） |
|---|---|---|
| 押し始め | 自分の準備ができたらすぐ押し始める | 相手も READY か PUSH になるのを待ってから押し始める |
| 相手が押すのをやめたとき（向きが外れた・転倒した・相手の状態が 0.5 s 届かない） | 1 台で押し続ける | 自分も押すのをやめて待つ |

押す側（`go2_a` が左、`go2_b` が右）は設定で前もって決め、通信では決めない。箱の回転を戻す規則（2 台の押す線を遅れている側へずらす）は、
箱の姿勢（`/team/object_state`、シミュレーションの真値でロボット同士の通信ではない）だけで判断できるので E4 にも残す。
外すと、通信の効果と回転を戻す規則の効果が 1 つの比較に混ざるため。

```bash
# 実行するシェルで毎回 source する（ros2_ws のビルドは 1 台の手順と同じ）
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
source scripts/setup/env.sh

# E5（2 台・協調）・E4（2 台・通信なし）・E3（1 台）。全試行を終えると自動で止まる
ros2 launch ros2_ws/launch/team.launch.py mode:=cooperative seed:=6001 num_trials:=16 run_name:=e5-seed6001 device:=cpu
ros2 launch ros2_ws/launch/team.launch.py mode:=independent seed:=6001 num_trials:=16 run_name:=e4-seed6001 device:=cpu
ros2 launch ros2_ws/launch/team.launch.py config:=configs/ros2_team_single.yaml robots:=go2_a mode:=independent \
    seed:=6001 num_trials:=16 run_name:=e3-seed6001 device:=cpu

# 押している最中に go2_b の協調のノードを止めると、go2_a が押すのをやめて待つことを確かめる
uv run --frozen python scripts/p4_ros2_check/check_stop_on_partner_loss.py --run-name p4-partner-loss --seed 6101

# ROS 2 を通さずに、同じ判断を並列環境で動かして確かめる（制御のパラメータの調整に使った）
uv run python scripts/p4_coordination_check/run_team_push.py --device cuda --config configs/team_rule_based.yaml \
    --mode cooperative --output-dir outputs/p4_coordination_check/e5-check
```

- 結果は `outputs/ros2_team_eval/<run_name>/` に `trials.csv`・`summary.json`・`ros2_metrics.json`（関節目標の遅れ・実時間比）、
  各ノードの計測値（`nodes/`。協調のノードは判断した時点の相手の状態の古さ）を保存する。保存先が既にあれば止まる。
  成否は `summary.json` があるかで判断する。`video:=true` で全試行を 1 本の評価動画に撮る（描画のため実時間比が下がる）。
- シミュレーションノードは `device:=cpu` で動かす。並列環境 1 つでは GPU の方が遅く、実時間比が 0.38 まで下がった（CPU では 0.82〜0.95）。
- 開始位置を押す位置の後ろへシードで散らす（3 条件で同じシードなら同じ開始位置）。通信なしでも開始位置が同じだと 2 台が同時に着いて、
  協調の「同時に押し始める」効果が測れないため。
- シード 6001 の 16 試行の結果（E4・E5 は同じ 16 試行を 2 回流した合計の 32 試行。E3 は 16 試行とも箱がほとんど動かなかったので 1 回）:

  | 条件 | 成功 | 通路の開通 | 箱の移動量の平均 | 箱の回転の最大（平均） | 押し始めの時刻の差（平均） | 転倒 |
  |---|---|---|---|---|---|---|
  | E3 1 台 | 0/16 | 0/16 | 0.25 m | 4° | — | 0 |
  | E4 2 台・通信なし | 21/32 | 26/32 | 1.87 m | 46° | 1.18 s | 3 |
  | E5 2 台・協調 | 11/32 | 14/32 | 1.32 m | 20° | 0.01 s | 0 |

  2 台なら 1 台では動かせない箱を動かせる。協調は押し始めをそろえ、箱の回転を抑え、転倒を無くしたが、成功率は通信なしより低い。
  協調では片方が押すのをやめるともう片方も止まるので、片方が回った箱の後ろの押す位置に戻れないと 2 台とも止まって時間切れになる。
  並列環境の 16 試行で、押している途中で PUSH を抜けた場面を数えると、抜けている間に相手が押していた割合は E5 1 %・E4 51 %、
  その間の箱の移動は平均 E5 +0.01 m・E4 +0.12 m だった。通信なしでは残った 1 台が押し続けて箱と 2 台の位置関係が変わり、
  外れた方が押す位置に戻りやすくなる。協調では箱も 2 台も止まったままになり、戻れない位置関係が変わらない。
  タスク管理の結果は 3 条件とも全試行で真値の判定と一致した。
- 物理は方策の応答を待たずに進めるので、同じ開始位置でも実行ごとに結果が変わる（2 回の実行で成功の数はほぼ同じだが、
  E5 で 2 回とも成功した試行は 1 つだけ）。条件の比較は試行ごとではなく、試行の割合で行う。

## テスト

```bash
# ユニットテスト（シミュレータと ROS 2 を使わない）
uv run --frozen pytest

# ROS 2 のメッセージを使うテスト（/opt/ros/jazzy と ros2_ws/install の setup.bash を source してから）
uv run --frozen pytest tests/ros2
```

## 参考にしたもの

| 対象 | 出典 | ライセンス | 本リポジトリでの扱い |
|---|---|---|---|
| Go2 の歩行学習の環境・報酬・PPO の設定 | Genesis 公式の例 `examples/locomotion/`（[genesis-world](https://github.com/Genesis-Embodied-AI/genesis-world) v1.4.5） | Apache-2.0 | コードは複製せず、観測・報酬の構成と初期値を参考に `src/cooperative_go2/` で実装した |
| Go2 の URDF・平面の URDF | genesis-world パッケージ同梱のアセット | 同梱物に個別の表記なし（原典のライセンスは未確認） | パッケージから実行時に読み込む。本リポジトリには含めない |

## ライセンス

本リポジトリのコードは [MIT License](LICENSE) で公開する。上の「参考にしたもの」と、実行時にパッケージから読み込むアセットは、それぞれの出典のライセンスに従う。
