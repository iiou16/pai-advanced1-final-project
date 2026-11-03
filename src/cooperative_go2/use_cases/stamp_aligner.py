"""別々のトピックで届く、同じ時刻の状態を組にする。

シミュレーションノードは 1 物理ステップの状態を同じ時刻で複数のトピックに配信するが、届く順は保証されない。
方策の観測を 1 時点の状態だけから組み立てるため、すべてのトピックがそろった最も新しい時刻の値を取り出す。
"""

from collections.abc import Mapping
from dataclasses import dataclass


class UnknownSourceError(KeyError):
    """組にする対象として登録していないトピックの値を受け取ったことを表す。"""


@dataclass(frozen=True)
class AlignedValues[T]:
    """同じ時刻にそろった値。

    Attributes:
        stamp_ns: 時刻 [ns]。
        values: 対象の名前 → その時刻の値。登録したすべての対象を含む。
    """

    stamp_ns: int
    values: Mapping[str, T]


class StampAligner[T]:
    """対象ごとに時刻付きの値を受け取り、すべての対象がそろった時刻の値を取り出す。スレッド間では共有しない。"""

    def __init__(self, sources: tuple[str, ...], max_pending_stamps: int) -> None:
        """組にする対象の名前と、そろうのを待つ時刻の数の上限を受け取る。上限を超えたら古い時刻から捨てる。"""
        self._sources = sources
        self._max_pending = max_pending_stamps
        self._pending: dict[int, dict[str, T]] = {}

    def put(self, source: str, stamp_ns: int, value: T) -> None:
        """対象 source の時刻 stamp_ns の値を受け取る。

        Raises:
            UnknownSourceError: source が登録した対象ではない。
        """
        if source not in self._sources:
            raise UnknownSourceError(f"登録していない対象: {source}（登録済み: {self._sources}）")
        self._pending.setdefault(stamp_ns, {})[source] = value
        while len(self._pending) > self._max_pending:
            del self._pending[min(self._pending)]

    def pop_latest_complete_or_none(self) -> AlignedValues[T] | None:
        """すべての対象がそろった最も新しい時刻の値を取り出し、その時刻以前の値を捨てる。そろった時刻が無ければ None。"""
        complete = [stamp for stamp, values in self._pending.items() if len(values) == len(self._sources)]
        if not complete:
            return None
        stamp = max(complete)
        values = self._pending[stamp]
        self._pending = {pending: entry for pending, entry in self._pending.items() if pending > stamp}
        return AlignedValues(stamp_ns=stamp, values=values)
