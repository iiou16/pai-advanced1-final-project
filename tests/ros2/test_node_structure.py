"""物理ステップを進められるのがシミュレーションノードだけであることの構造の検証。

方策のノードは Genesis を読み込まないので、シーンを持てず、物理ステップを進められない。
"""

import subprocess
import sys

import pytest

POLICY_NODE_MODULES = (
    "cooperative_go2.frameworks.ros2_gait_node",
    "cooperative_go2.frameworks.ros2_coord_node",
)


@pytest.mark.parametrize("module", POLICY_NODE_MODULES)
def test_policy_node_does_not_load_simulator(module: str) -> None:
    # 他のテストが既に読み込んだモジュールの影響を受けないよう、新しいプロセスで確かめる
    code = f"import sys, {module}; print('genesis' in sys.modules)"

    completed = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    assert completed.stdout.strip() == "False"
