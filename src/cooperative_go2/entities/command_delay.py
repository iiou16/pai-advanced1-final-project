"""速度指令が歩行方策に届くまでの遅れを、歩行の制御周期の単位で再現する遅延線。

通信の遅れに対する頑健性を評価するために使う。遅れのあいだ（試行の始めなど、まだ指令が届いていない間）は、
ROS 2 の実行系で指令が届かないときと同じく停止の指令（すべて 0）を渡す。
"""

import torch

from cooperative_go2.entities.velocity_command import VELOCITY_COMMAND_SIZE


class InvalidCommandDelayError(ValueError):
    """遅れのステップ数が 0 以上の整数でないことを表す。"""


class CommandDelayLine:
    """環境ごとに、delay_steps 回前に入れた速度指令を取り出す。

    delay_steps が 0 なら、入れた指令をそのまま返す（遅れなし）。
    """

    def __init__(self, delay_steps: int, num_envs: int, device: torch.device | str) -> None:
        """遅れのステップ数と環境の数を受け取り、全環境を停止の指令で満たす。

        Args:
            delay_steps: 遅らせる歩行の制御周期の数。0 以上の整数。
            num_envs: 並列環境の数。
            device: 指令のテンソルを置くデバイス。

        Raises:
            InvalidCommandDelayError: delay_steps が負、または整数でない。
        """
        if isinstance(delay_steps, bool) or not isinstance(delay_steps, int) or delay_steps < 0:
            raise InvalidCommandDelayError(f"遅れのステップ数は 0 以上の整数が必要: {delay_steps!r}")
        self._delay_steps = delay_steps
        # 先頭が最も古い指令。push のたびに先頭を取り出し、末尾に新しい指令を入れる
        self._queue = torch.zeros((delay_steps, num_envs, VELOCITY_COMMAND_SIZE), device=device)

    @property
    def delay_steps(self) -> int:
        """遅らせる歩行の制御周期の数。"""
        return self._delay_steps

    def push(self, commands: torch.Tensor) -> torch.Tensor:
        """指令 (num_envs, 3) を入れ、delay_steps 回前に入れた指令 (num_envs, 3) を返す。"""
        if self._delay_steps == 0:
            return commands
        delayed = self._queue[0]
        self._queue = torch.cat((self._queue[1:], commands.unsqueeze(0)), dim=0)
        return delayed

    def reset_where(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の、まだ取り出していない指令を停止の指令にする。"""
        self._queue = self._queue.masked_fill(mask.view(1, -1, 1), 0.0)
