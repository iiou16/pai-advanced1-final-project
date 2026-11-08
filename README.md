# PAI Final Project

Physical AI 応用編1 知能ロボティクス講座（2026 Summer）最終課題のリポジトリです。

課題テーマ: **AI×ロボティクスによる創造的タスクの実現**

四足歩行ロボット Unitree Go2 が、障害物を「避ける」のではなく「動かして」環境を変える。シミュレータ Genesis の上で、
1 台の Go2 が箱を目標へ押すことと、1 台では動かせない重い箱でふさがれた通路を 2 台が協調して押し開けることを、強化学習と ROS 2 で実現した。
実機での検証は対象外。

## 概要

方策を 2 層に分けている。下の層の歩行方策は速度指令 `[vx, vy, yaw_rate]` に追従して 12 関節を動かす（50 Hz）。
上の層の高レベル方策は、箱と目標の位置から速度指令を決める（10 Hz）。歩行方策は 1 回学習して固定し、すべての段階で使い回す。

| 段階 | 内容 | 主な結果（学習に使っていないシードでの評価） |
|---|---|---|
| 歩行 | 停止・前進・旋回・後退・横移動の指令に追従する歩行方策（PPO） | 指令ごとの速度の追従誤差 0.04 m/s 以下・転倒 0 % |
| 1 台の箱押し | 2 kg の箱を 1 m 先の目標（方向 ±60°）へ押す高レベル方策（PPO） | 成功率 100 %（192 試行）。規則ベースの制御は 53〜70 % |
| ROS 2 での実行 | シミュレーション・歩行方策・高レベル方策を別々のノードにし、ROS 2 の通信だけでつなぐ | 同じ初期条件で成功率 100 %。方策のノードを止めるとロボットは止まる |
| 2 台の規則ベースの協調 | 5 kg の箱でふさがれた通路を、2 台が状態を伝え合って押し開ける | 1 台は 0/16。2 台の協調は 19/32 で、押し始めをそろえて箱の回転を抑えた |
| 2 台の協調学習 | IPPO・MAPPO（skrl）で協調を学習し、ROS 2 のノードで動かす | MAPPO は 32/32・成功までの平均 8.2 s。箱の回転は規則ベースより大きい |

ノード構成・比べた条件・数値の詳細は、下の各節に書いた。

## リポジトリの構成

| パス | 内容 |
|---|---|
| `src/cooperative_go2/` | 本体。Clean Architecture の 4 層（`entities` → `use_cases` → `interface_adapters` → `frameworks`）に分ける。報酬・観測・成功判定などの計算は `entities` にあり、Genesis にも学習ライブラリにも依存しない |
| `configs/` | 学習・評価・シーン・ROS 2 の設定。すべてのキーが必須で、読み込み直後に検証する |
| `ros2_ws/` | 独自の ROS 2 のメッセージ・アクション（`go2_interfaces`）と launch |
| `scripts/` | 環境の確認・物理の確認・評価をまとめて流すスクリプト |
| `tests/` | ユニットテスト（`tests/unit`）と ROS 2 のメッセージを使うテスト（`tests/ros2`） |

## 推論だけで評価シーンを再実行する

学習をせずに、配布している学習済みモデルで評価シーンを流す手順。下の「セットアップ」と「学習済みモデルの取得」を済ませてから、
リポジトリのルートで実行する。結果は `outputs/` の下に保存される（保存先が既にあれば上書きせずに止まる）。

```bash
source scripts/setup/env.sh

# 1 台の箱押し（ROS 2 を通さない並列環境、64 試行。--video で評価動画も撮る）
uv run --frozen go2-evaluate-push --config configs/push_single_wide.yaml --algorithm policy \
    --policy-run-dir outputs/push/push-stage2-wide60 --checkpoint 200 \
    --num-envs 64 --seed 2001 --device cuda --run-name e2-wide60-seed2001 --video

# 以降は ROS 2 を使う（ros2_ws のビルドは「ROS 2 を経由した箱押し」の節）
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash

# 1 台の箱押しを ROS 2 の 3 ノードで流す
ros2 launch ros2_ws/launch/push_single.launch.py seed:=2001 num_trials:=64 run_name:=e2-ros2-wide60-seed2001 \
    device:=cuda video:=false record_bag:=true

# 2 台の通路の開通を、規則ベースの協調（E5）と、学習した協調（MAPPO・別々）で流す
ros2 launch ros2_ws/launch/team.launch.py mode:=cooperative seed:=6001 num_trials:=16 run_name:=e5-seed6001 device:=cpu
ros2 launch ros2_ws/launch/team.launch.py mode:=learned policy_run_dir:=outputs/team_marl/mappo-separate-seed21 \
    policy_iteration:=1000 seed:=6001 num_trials:=16 run_name:=learned-mappo-separate-seed6001 device:=cpu
```

学習からやり直す手順は、段階ごとの節（「歩行方策の学習と評価」「箱押しの高レベル方策の学習と評価」「2 台の協調学習（IPPO・MAPPO）と比較」）に書いた。

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

学習済みのモデルは GitHub の Release で配布している。歩行方策と 1 台の箱押しの方策はタグ `v0.2.0`、
2 台の協調の方策（IPPO・MAPPO）はタグ `v0.3.0` にある。学習をせずに評価だけを再現するときは、次の手順で `outputs/` に置く。

```bash
# Release の添付ファイルを取得する（gh を使わない場合は Release のページから同じファイルをダウンロードする）
gh release download v0.2.0 -R iiou16/pai-advanced1-final-project -p '*.tar.gz' -D outputs/release
gh release download v0.3.0 -R iiou16/pai-advanced1-final-project -p '*.tar.gz' -D outputs/release

# 歩行方策（model_1500.pt と学習時の設定 gait.yaml）→ outputs/gait/gait-command-tracking/
mkdir -p outputs/gait outputs/push
tar -xzf outputs/release/gait-command-tracking.tar.gz -C outputs/gait

# 箱押しの方策（model_200.pt と学習時の設定 push_single.yaml・scene.yaml）→ outputs/push/push-stage2-wide60/
tar -xzf outputs/release/push-stage2-wide60.tar.gz -C outputs/push

# 2 台の協調の方策（1000 回目の更新のチェックポイントと、学習時の設定 push_multi.yaml・scene.yaml・学習方式 variant.json）
#   → outputs/team_marl/{mappo-separate-seed21,mappo-shared-seed21,ippo-shared-seed21}/
mkdir -p outputs/team_marl
for run in mappo-separate-seed21 mappo-shared-seed21 ippo-shared-seed21; do
    tar -xzf outputs/release/${run}.tar.gz -C outputs/team_marl
done
```

- 歩行方策は `configs/push_single.yaml` の `gait_policy`（`outputs/gait/gait-command-tracking` の `model_1500.pt`）から読み込む。
- 箱押しの方策は、下の評価のコマンドで `--policy-run-dir outputs/push/push-stage2-wide60 --checkpoint 200` を指定して使う。
- 2 台の協調の方策は、`policy_run_dir:=outputs/team_marl/<run> policy_iteration:=1000` を指定して使う。

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

## 1 台の箱押しの頑健性の評価

学習した条件だけで成功していないかを確かめるため、箱押しの方策（第 2 段階、`model_200.pt`）と規則ベースの制御を、
条件を 1 種類ずつ変えて評価する。条件は `configs/push_robustness.yaml` にまとめた。学習時の条件（`nominal`: 箱 2 kg・摩擦 1.0・
箱の位置のずれ ±0.1 m・目標まで 1 m・目標の方向 ±60°・指令の遅れなし）から、箱の質量・摩擦・配置・速度指令の遅れのどれか 1 つだけを変える。

```bash
source scripts/setup/env.sh

# 全条件 × 2 方式 × 3 シード（2001〜2003、各 64 試行）を順に評価する（RTX 2080 Ti で約 2 時間）。
# 評価済みの組み合わせは飛ばすので、止めても続きから流せる。--condition・--seed で一部だけも流せる
uv run --frozen python scripts/p6_robustness/run_push_robustness.py

# 条件 × 方式ごとに 3 シードを合わせた表（Markdown と CSV）にする
uv run --frozen python scripts/p6_robustness/summarize_push_robustness.py --output-dir outputs/push_eval/e8-summary

# 1 つの条件だけを評価する（go2-evaluate-push に条件の一覧と名前を渡す）
uv run --frozen go2-evaluate-push --config configs/push_single_wide.yaml --algorithm policy \
    --policy-run-dir outputs/push/push-stage2-wide60 --checkpoint 200 \
    --robustness-config configs/push_robustness.yaml --condition delay_0.10s \
    --num-envs 64 --seed 2001 --device cuda --run-name e8-policy-delay_0.10s-seed2001

# 箱を水平の外力で押し、滑るか倒れるかを確かめる（摩擦の条件の確認）
uv run --frozen python scripts/p6_robustness/probe_box_friction.py --mass-kg 2.0 --friction 1.3 --force-n 35
```

- 速度指令の遅れは、高レベル方策が決めた指令を歩行の制御周期（0.02 s）の単位で遅らせて歩行方策に渡す（遅れの間は停止の指令）。
  並列環境の中で模擬した遅れで、ROS 2 の通信の遅れそのものではない。ROS 2 の実行系で測った遅れは約 1 周期（0.02 s）。
  届いた指令は高レベル方策の周期（0.1 s）のあいだ使い続けるので、使う時点の指令の古さは遅れより最大 0.08 s 大きい。
  ROS 2 の実行系は古さが 0.3 s を超えた指令を捨てて止まるが、この評価は捨てずに使う（遅れそのものへの強さを見るため）。
  そのため 0.30 s の条件は、ROS 2 の実行系なら止まる場面を含む。
- 箱の摩擦は床（1.0）より下げられない（接触する 2 面の大きい方が使われる）ので、上げる方向だけを評価する。箱の表面の摩擦なので、
  箱と Go2 の間の摩擦も上がる。重心を水平に押すとき、摩擦係数が箱の幅 / 高さ（約 1.33）を超えると、箱は滑らずに倒れる
  （1.5・2.0 の条件）。
- `nominal` は、箱押しの評価（±60°、シード 2001〜2003）と試行ごとに同じ初期条件になる。
- 評価の結果は `outputs/push_eval/e8-<方式>-<条件>-seed<シード>/` に保存する。接頭辞 `e8` は、この評価の実行を
  箱押しの評価の実行（上の `e1-…`（規則ベース）・`e2-…`（学習した方策））と見分けるためのもの。評価済みの組み合わせを飛ばす前と
  集計する前に、記録に残った条件の値・シード・方式・チェックポイント・試行の数が今の設定と一致するかを確かめ、違えば止まる。
- 結果（成功率。各 64 試行 × 3 シード = 192 試行）:

  | 種類 | 条件 | 学習した方策 | 規則ベース |
  |---|---|---|---|
  | — | 学習時の条件（nominal） | 100.0 % | 60.9 % |
  | 箱の質量 | 1 kg / 3 kg / 4 kg | 100.0 / 96.9 / 39.6 % | 100.0 / 3.6 / 0.5 % |
  | 箱の摩擦（滑る範囲） | 1.2 / 1.3 | 34.9 / 3.6 % | 3.1 / 1.6 % |
  | 箱の摩擦（倒れる範囲） | 1.5 / 2.0 | 0.0 / 0.0 % | 0.0 / 0.0 % |
  | 配置 | 箱のずれ ±0.2 m / ±0.3 m | 100.0 / 99.0 % | 60.4 / 65.1 % |
  | 配置 | 目標まで 0.5〜2.0 m / 目標の方向 ±90° | 92.2 / 97.4 % | 43.2 / 52.6 % |
  | 速度指令の遅れ | 0.02 / 0.04 / 0.10 / 0.20 / 0.30 s | 100.0 / 99.5 / 96.9 / 97.4 / 88.5 % | 55.7 / 59.4 / 61.5 / 76.0 / 61.5 % |

  学習した方策は、箱のずれ・目標の方向（±90°）と 0.2 s までの指令の遅れでは、成功率 96 % 以上を保った。学習した距離（1 m）を超える
  目標（1.5〜2.0 m）では成功が 47/61 に下がり、0.3 s の遅れでは 88.5 % で、箱の回転が大きくなった（平均 54°、学習時の条件では 11°）。
  箱の質量は 3 kg までは 96.9 % だが、4 kg では 39.6 % に下がった。最も弱いのは箱の摩擦で、床との摩擦力が 3 kg の箱より小さい
  摩擦 1.2（2 kg）でも 34.9 % に下がった。Go2 が押すと、摩擦 1.2・1.3 の箱は倒れはしないが（シード 2001 の 64 試行で傾き 30° を
  超えた試行なし）、押している間に前へ傾く量が学習時の条件の約 2 倍になり（傾きの最大の中央値 8.6〜8.8°、学習時 4.2°・3 kg 5.5°）、
  失敗した試行ほど傾いていた（平均 9.0°、成功した試行 6.4°）。傾きが失敗の原因か、Go2 と箱の間の摩擦が増えたことの影響かは確かめていない。
  規則ベースは押す指令が 0.3 m/s と遅く、3 kg 以上や摩擦 1.2 以上ではほとんど押せない（この指令で前進し続けても、
  3 kg の箱は 10 秒で平均 0.06 m しか動かないことを、箱の質量を決めるときに測っている）。
  規則ベースの成功率は、同じ条件・同じシードで流し直しても数試行ぶれる（GPU の計算の非決定性で、時間切れぎりぎりの試行の結果が変わる）。

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
| `/go2_x/coord_policy` | `go2-ros2-team-coord` | 状態機械（INIT → APPROACH → ALIGN → READY → PUSH → VERIFY → DONE / FAILED）で判断し、`cmd_vel` と作業の状態（`/team/agent_status`。段階・押す意図・自分の位置と向きと指令）を配信する（10 Hz）。協調のモードでは相手の状態を受け、相手の準備を待って同時に押し始める。相手の状態が 0.5 s 途絶えたら・相手が失敗したら押すのをやめて待つ。`mode:=learned` では、状態機械の代わりに学習した協調の方策で判断する（下の「2 台の協調学習」） |
| `/go2_x/gait_policy` | `go2-ros2-gait` | 1 台の箱押しと同じ歩行方策のノード |

設定は `configs/ros2_team.yaml`（2 台）・`configs/ros2_team_single.yaml`（1 台）と、それが指す `configs/team_rule_based*.yaml`・`configs/scenes/s3_*.yaml`。

E4 と E5 は、ロボットごとの制御を同じにし、ロボット同士の通信（`/team/agent_status`）を使うかどうかだけを変える（協調のノードの `mode`）。

| | E4（`mode:=independent`） | E5（`mode:=cooperative`） | E5 の改良前（`mode:=cooperative_wait`） |
|---|---|---|---|
| 押し始め | 自分の準備ができたらすぐ押し始める | 相手も READY か PUSH になるのを待ってから押し始める | E5 と同じ |
| 相手が押す位置を直し始めたとき（向きが外れた） | 自分の線のまま 1 台で押し続ける | 押す線を箱の中央へ寄せて押し続ける | 押すのをやめて待つ |
| 相手が転倒した・相手の状態が 0.5 s 届かない | 1 台で押し続ける | 押すのをやめて待つ | 押すのをやめて待つ |

押す側（`go2_a` が左、`go2_b` が右）は設定で前もって決め、通信では決めない。箱の回転を戻す規則（2 台の押す線を遅れている側へずらす）は、
箱の姿勢（`/team/object_state`、シミュレーションの真値でロボット同士の通信ではない）だけで判断できるので E4 にも残す。
外すと、通信の効果と回転を戻す規則の効果が 1 つの比較に混ざるため。

```bash
# 実行するシェルで毎回 source する（ros2_ws のビルドは 1 台の手順と同じ）
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
source scripts/setup/env.sh

# E5（2 台・協調）・E4（2 台・通信なし）・E3（1 台）。全試行を終えると自動で止まる。E5 の改良前は mode:=cooperative_wait
ros2 launch ros2_ws/launch/team.launch.py mode:=cooperative seed:=6001 num_trials:=16 run_name:=e5-seed6001 device:=cpu
ros2 launch ros2_ws/launch/team.launch.py mode:=independent seed:=6001 num_trials:=16 run_name:=e4-seed6001 device:=cpu
ros2 launch ros2_ws/launch/team.launch.py config:=configs/ros2_team_single.yaml robots:=go2_a mode:=independent \
    seed:=6001 num_trials:=16 run_name:=e3-seed6001 device:=cpu

# 評価と同じ開始位置の一部の試行だけを動画付きで流す（trial_indices に試行の番号）
ros2 launch ros2_ws/launch/team.launch.py mode:=cooperative seed:=6001 num_trials:=16 trial_indices:=8,1 \
    run_name:=e5-video-seed6001-trials8-1 device:=cpu video:=true

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

  | 条件 | 成功 | 通路の開通 | 箱の移動量の平均 | 箱の回転の最大（平均／全体） | 押し始めの時刻の差（平均） | 転倒 |
  |---|---|---|---|---|---|---|
  | E3 1 台 | 0/16 | 0/16 | 0.25 m | 4° / 4° | — | 0 |
  | E4 2 台・通信なし | 21/32 | 26/32 | 1.87 m | 46° / 87° | 1.18 s | 3 |
  | E5 2 台・協調 | 19/32 | 20/32 | 1.64 m | 17° / 31° | 0.01 s | 0 |
  | E5 の改良前（止まって待つ） | 11/32 | 14/32 | 1.32 m | 20° / 65° | 0.01 s | 0 |

  2 台なら 1 台では動かせない箱を動かせる。協調は通信なしとほぼ同じ成功率で、押し始めをそろえ、箱の回転を抑え、転倒を無くし、
  通路を早く開けた（開通までの平均 E5 18.3 s・E4 24.3 s）。
  改良前の協調では、片方が押すのをやめるともう片方も止まるので、片方が回った箱の後ろの押す位置に戻れないと 2 台とも止まって時間切れになった。
  並列環境の 16 試行で、押している途中で PUSH を抜けた場面を数えると、抜けている間に相手が押していた割合は改良前の E5 1 %・E4 51 %、
  その間の箱の移動は平均 +0.01 m・+0.12 m だった。通信なしでは残った 1 台が押し続けて箱と 2 台の位置関係が変わり、
  外れた方が押す位置に戻りやすくなる。そこで E5 を、相手が押す位置を直している間は押す線を箱の中央へ寄せて押し続ける規則に改良し、
  成功は 11 から 19 に増えた（並列環境の 32 試行、シード 5001 では改良前 8・改良後 20・通信なし 18）。
  残る失敗の多くは、片方が回った箱の後ろの押す位置に戻れない時間切れ。タスク管理の結果は全条件・全試行で真値の判定と一致した。
- 物理は方策の応答を待たずに進めるので、同じ開始位置でも実行ごとに結果が変わる（2 回の実行で成功の数はほぼ同じだが、
  E5 で 2 回とも成功した試行は 1 つだけ）。条件の比較は試行ごとではなく、試行の割合で行う。

## 2 台の協調学習（IPPO・MAPPO）と比較

上と同じ通路・重い箱・開始位置の散らし方で、2 台の協調を強化学習で獲得する。各ロボットの方策（Actor）は ROS 2 の実行時にも
受け取れる情報だけを観測する（自分の odom、箱の状態、退避領域、自分が前に出した指令、自分の側、`/team/agent_status` で
最後に届いた相手の位置・向き・指令とその古さ）。学習時は ROS 2 を通さず、相手の状態を 0.1 s（1 周期）遅らせ、1 割を落として届ける。
価値関数（Critic）は、IPPO（E6）では Actor と同じ観測、MAPPO（E7）では学習時にだけ使える全体の真の状態（2 台と箱の状態・接触など）を読む。
ライブラリは skrl 2.1.1 で、IPPO と MAPPO の違いは Critic の入力だけになる。2 台で Actor と Critic を共有する方式と、別々に学習する方式を選べる。

報酬は、箱が退避領域に近づいた量・成功・箱の回転を 2 台で共有し、自分の押す位置（規則ベースと同じ、箱の後ろに横に並ぶ位置）に
近づいた量・意図した面での接触・転倒・シーン外への離脱・ロボット同士の接近・指令の大きさをロボットごとに与える。
シーン・開始位置の散らし方と引く順・判断の周期・転倒の判定・時間切れ（60 s）・成功の判定（退避領域に入ったまま 1 s）・指令の上限
（0.6 m/s）は規則ベースの協調（E4・E5）とそろえる（`tests/unit/test_push_multi_config.py`）。設定は `configs/push_multi.yaml`。

```bash
source scripts/setup/env.sh

# 学習（出力: outputs/team_marl/<run-name>/。チェックポイント（50 回の更新ごと）・TensorBoard のログ・設定の写し・学習方式・コミット）。
# 1024 環境・1000 回の更新で、RTX 2080 Ti で約 2 時間 45 分
uv run go2-train-team-marl --config configs/push_multi.yaml --algorithm mappo --parameter-sharing separate \
    --run-name mappo-separate-seed21

# ROS 2 を通さない並列環境での評価（出力: outputs/team_marl_eval/<name>/ に trials.csv と summary.json）。
# 規則ベースの協調の並列環境での確認（scripts/p4_coordination_check/run_team_push.py）と同じシード・試行の数なら同じ開始位置
uv run go2-evaluate-team-marl --run-dir outputs/team_marl/mappo-separate-seed21 --iteration 1000 --num-envs 32 --seed 5001 \
    --drop-probability 0.0 --device cuda --output-dir outputs/team_marl_eval/mappo-separate-seed21-it1000-seed5001

# ROS 2 を経由した評価（上の規則ベースの協調と同じ launch。ros2_ws の msg を変えたのでビルドし直してから）
source /opt/ros/jazzy/setup.bash
source ros2_ws/install/setup.bash
ros2 launch ros2_ws/launch/team.launch.py mode:=learned policy_run_dir:=outputs/team_marl/mappo-separate-seed21 \
    policy_iteration:=1000 seed:=6001 num_trials:=16 run_name:=learned-mappo-separate-seed6001 device:=cpu

# 方式ごとに複数の実行を合わせて表にする
uv run python scripts/p5_marl_check/compare_ros2_team_runs.py --output outputs/p5_marl_check/comparison.json \
    --method "E5=e5-seed6001,e5-seed6001-rerun" --method "MAPPO・別々=learned-mappo-separate-seed6001,learned-mappo-separate-seed6001-rerun"
```

- 学習した協調の方策は、上の「学習済みモデルの取得」の `v0.3.0` で配布している（MAPPO・別々、MAPPO・共有、IPPO・共有の 1000 回目の更新）。
- ROS 2 を経由して、E3〜E7 を同じ条件・同じコードで比べた（シード 6001 の 16 試行を 2 回、E3 は 1 回。Genesis は CPU。
  E3〜E5 もこのコードで流し直したので、上の規則ベースの表と少し違う）。学習はどれもシード 21・1000 回の更新:

  | 条件 | 成功 | 通路の開通 | 開通までの平均 | 成功までの平均 | 箱の回転の最大（平均／全体） | 押し始めの時刻の差（平均） | 転倒 |
  |---|---|---|---|---|---|---|---|
  | E3 1 台 | 0/16 | 0/16 | — | — | 4° / 4° | — | 0 |
  | E4 2 台・通信なし | 20/32 | 22/32 | 20.4 s | 26.0 s | 39° / 85° | 1.17 s | 0 |
  | E5 2 台・規則ベースの協調 | 19/32 | 20/32 | 17.0 s | 23.1 s | 17° / 42° | 0.01 s | 0 |
  | E6 IPPO（共有） | 23/32 | 25/32 | 12.8 s | 17.9 s | 53° / 90° | — | 6 |
  | E7 MAPPO（共有） | 32/32 | 32/32 | 8.6 s | 10.9 s | 77° / 90° | — | 0 |
  | E7 MAPPO（別々） | 32/32 | 32/32 | 6.1 s | 8.2 s | 36° / 89° | — | 0 |

  MAPPO は成功・開通・所要時間で規則ベースと IPPO を上回った。転倒は規則ベースと同じ 0（IPPO は 6）。同じ共有の方式で比べると、
  ROS 2 では集中 Critic で成功が 23 から 32 に増え、転倒が 6 から 0 になった。劣ったのは箱の回転で、回転の最大の平均は
  MAPPO（別々）36°・MAPPO（共有）77° で、押し始めをそろえる E5（17°）より大きい。ただし回転したままでも全試行で通路を開けた。
  タスク管理の結果は全条件・全試行で真値の判定と一致した。
- ROS 2 を通さない並列環境（シード 5001・32 試行・欠落なし）では、成功は E4 18・E5 20・IPPO（共有）28・MAPPO（共有）28・
  MAPPO（別々）32、回転の最大の平均は 48°・17°・49°・74°・28°、転倒は 4・0・3・4・0。並列環境では IPPO と MAPPO（共有）の
  成功は同じで、集中 Critic の効果は ROS 2 の評価ほどはっきりしない。
- 学習した方策には段階が無いので、押し始めの時刻（PUSH に入った時刻）は記録しない。
- どの方式も学習のシードは 1 つで、シード間のばらつきは測っていない。共有の方式は skrl がロボットごとに順に更新するので、
  共有と別々の差には更新の仕方の違いも含まれる。
- 回転の減点を 5 倍にした設定（`configs/push_multi_rotation.yaml`）で MAPPO（別々）を学習し直しても、ROS 2 で成功 31/32・
  回転の最大は平均 35°（全体 86°）で回転は減らず、成功までの平均は 10.7 s に延びた。

## 既知の制約

- すべてシミュレーションの中での結果で、実機では確かめていない。
- 箱の位置・速度と、ロボットと箱の接触は、シミュレーションの真値をそのまま配信している（`/team/object_state`・`box_contact`）。実機ではカメラなどでの推定が要る。
- ROS 2 のノードは 1 台の計算機の上で動かした。ノード間の通信は同じ計算機の中に限られる。
- ROS 2 の実行系は方策の応答を待たずに物理を進めるので、同じシード・同じ開始位置でも実行ごとに結果が変わる。比較は試行ごとではなく、試行の割合で行う。
- 評価は平らな床で、外乱が無い条件に限られる。段差・坂・外乱への強さは確かめていない。
- 箱の摩擦は、床の摩擦（1.0）より下げられない（Genesis は接触する 2 面の摩擦の大きい方を使う）。
- 1 台の箱押しの目標の方向は ±60° で学習した。箱の真横や後ろの目標のように、大きく回り込む配置は学習していない。
- 2 台の協調学習は、どの方式も学習のシードが 1 つで、シード間のばらつきは測っていない。学習した協調は、規則ベースの協調より箱を大きく回す。
- 動作を確かめた環境は Ubuntu 24.04（WSL2）＋ NVIDIA RTX 2080 Ti。ほかの GPU・OS では確かめていない。

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
| PPO の実装 | [RSL-RL](https://github.com/leggedrobotics/rsl_rl)（rsl-rl-lib 5.5.2） | BSD-3-Clause | ライブラリとして使う（歩行方策と 1 台の箱押しの方策） |
| IPPO・MAPPO の実装 | [skrl](https://github.com/Toni-SM/skrl) 2.1.1 | MIT | ライブラリとして使う（2 台の協調学習）。Genesis の並列環境との接続は自作した |
| ROS 2 との接続 | [genesis_ros](https://github.com/signalbotics/genesis_ros) を調べた | — | 関節目標を制御周期ごとに受けて学習時と同じ遅れで適用する経路が無かったため採用せず、`rclpy` で自作した |

本プロジェクトで独自に作ったもの:

- 歩行方策の速度指令の範囲を広げ、停止とその場旋回の指令を意図的に混ぜた学習（公式の例は前進 0.5 m/s に固定）
- 箱押しのシーン・通路のシーンと、報酬・観測・成功判定・終了判定、規則ベースの制御（1 台・2 台の協調）
- 箱を置くシーンの摩擦円錐を `elliptic` にした物理の設定と、その確認のスクリプト
- ROS 2 のノード（シミュレーション・歩行方策・高レベル方策・タスク管理）、独自のメッセージ・アクション、launch
- 2 台の協調学習の環境（通信の遅れと欠落の模擬を含む）と、学習した方策を ROS 2 で動かす判断
- 評価と記録（同じシードで同じ初期条件にする評価、ROS 2 の経路の計測、頑健性の評価）

## ライセンス

本リポジトリのコードは [MIT License](LICENSE) で公開する。上の「参考にしたもの」と、実行時にパッケージから読み込むアセットは、それぞれの出典のライセンスに従う。
