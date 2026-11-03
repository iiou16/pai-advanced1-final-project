"""学習・評価の流れが区別して扱う、シミュレーションの失敗。"""


class SimulationSetupError(RuntimeError):
    """シミュレーションを用意できない、または設定が物理計算の容量を満たさないことを表す。

    アセットやロボットのモデルの不整合、衝突候補数などの上限の超過が当たる。設定かモデルを直すまで再実行しても失敗する。
    """


class SimulationDivergedError(RuntimeError):
    """物理計算が発散し（NaN など）、その結果を信頼できないことを表す。"""
