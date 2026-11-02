from pathlib import Path

import pytest

from cooperative_go2.interface_adapters.gait_run_store import InvalidRunNameError, resolve_run_dir


def test_run_name_becomes_directory_under_root(tmp_path: Path) -> None:
    assert resolve_run_dir(tmp_path, "gait-baseline") == tmp_path / "gait-baseline"


@pytest.mark.parametrize(
    "run_name",
    ["", ".", "..", "../escape", "nested/run", "/absolute", "back\\slash"],
    ids=["empty", "dot", "dotdot", "parent", "nested", "absolute", "backslash"],
)
def test_run_name_outside_single_directory_is_rejected(tmp_path: Path, run_name: str) -> None:
    with pytest.raises(InvalidRunNameError):
        resolve_run_dir(tmp_path, run_name)
