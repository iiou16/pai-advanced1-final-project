"""Genesis のプロセス全体の初期化と、シードを固定した乱数生成器の用意。"""

import genesis as gs
import torch

# 学習・評価とも単精度で計算する。RTX 2080 Ti の倍精度は遅く、公式の歩行学習の例も単精度である
GENESIS_PRECISION = "32"
GENESIS_LOGGING_LEVEL = "warning"


def init_genesis_on_cuda(seed: int) -> None:
    """Genesis を CUDA で初期化し、Genesis と PyTorch の大域の乱数シードを固定する。プロセスで 1 回だけ呼ぶ。"""
    gs.init(
        backend=gs.cuda,
        precision=GENESIS_PRECISION,
        logging_level=GENESIS_LOGGING_LEVEL,
        seed=seed,
        performance_mode=True,
    )


def init_genesis_on_cpu(seed: int) -> None:
    """Genesis を CPU で初期化し、Genesis と PyTorch の大域の乱数シードを固定する。プロセスで 1 回だけ呼ぶ。

    GPU で学習が走っていても取り合わずに、少数の環境でシーンの物理を確かめるために使う。
    Genesis は環境変数 QD_NUM_THREADS が無ければ CPU のスレッドを 1 本に絞るので、他の処理の邪魔をしにくい。
    """
    gs.init(backend=gs.cpu, precision=GENESIS_PRECISION, logging_level=GENESIS_LOGGING_LEVEL, seed=seed)


def make_seeded_generator(seed: int) -> torch.Generator:
    """Genesis と同じデバイスに置いた、シードを固定した乱数生成器を返す。

    前提: init_genesis_on_cuda() か init_genesis_on_cpu() 済み。
    """
    return torch.Generator(device=gs.device).manual_seed(seed)
