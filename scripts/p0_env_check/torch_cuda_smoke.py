"""PyTorch が RTX 2080 Ti (sm_75) 上で実際に CUDA カーネルを実行できることを確かめるスモークテスト.

`nvidia-smi` で GPU が見えても、wheel が対象アーキテクチャのカーネルを含まなければ
演算時に "no kernel image is available" で失敗するため、実演算まで確認する。
"""

import torch

REQUIRED_ARCH = "sm_75"
MATRIX_SIZE = 2048


class CudaSmokeError(RuntimeError):
    """PyTorch から GPU 演算を実行できないことを表す。"""


def main() -> None:
    """CUDA の可視性・対応アーキテクチャ・行列演算の順に検証する。"""
    print(f"torch={torch.__version__} cuda_runtime={torch.version.cuda}")
    if not torch.cuda.is_available():
        raise CudaSmokeError("torch.cuda.is_available() が False")

    arch_list = torch.cuda.get_arch_list()
    print(f"device={torch.cuda.get_device_name(0)} arch_list={arch_list}")
    if REQUIRED_ARCH not in arch_list:
        raise CudaSmokeError(f"この torch ビルドは {REQUIRED_ARCH} を含まない: {arch_list}")

    matrix = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device="cuda")
    product = matrix @ matrix
    torch.cuda.synchronize()
    if not torch.isfinite(product).all():
        raise CudaSmokeError("GPU 行列積の結果に非有限値が含まれる")
    print(f"matmul {MATRIX_SIZE}x{MATRIX_SIZE} on cuda: OK")
    print("RESULT: PASS")


if __name__ == "__main__":
    main()
