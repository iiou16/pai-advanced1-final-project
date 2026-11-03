"""箱押しの学習の出力先に置くファイルの名前。

評価時は push_single.yaml の写しから方策のネットワーク構成を読み、評価の設定と学習時の条件が食い違っていないかを確かめる。
scene.yaml は学習時のシーンの記録として残す。シミュレータを読み込まずに学習済みの方策を探せるよう、学習器と分けて置く。
"""

PUSH_RUN_CONFIG_FILENAME = "push_single.yaml"
PUSH_RUN_SCENE_CONFIG_FILENAME = "scene.yaml"
