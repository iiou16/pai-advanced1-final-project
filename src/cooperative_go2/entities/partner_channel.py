"""学習時に、相手の状態を通信の遅れと欠落つきで Actor に届ける模擬。

ROS 2 の実行時、協調のノードは /team/agent_status で相手の状態（位置・向き・押す意図としての速度指令）を受け取り、
最後に届いた状態を判断に使う。学習時は ROS 2 を通さないため、同じ届き方をここで模擬する。
相手の真の現在の状態は Actor に渡さず、配信された状態のうち届いたものだけを渡す。
"""

from dataclasses import dataclass, fields

import torch

from cooperative_go2.entities.team_marl_config import TeamCommunicationConfig


@dataclass(frozen=True)
class PartnerMessage:
    """1 台のロボットが配信する状態。テンソルは並列環境のバッチ。

    Attributes:
        position_xy_m: 配信した時点の自分の胴体の位置 (num_envs, 2)、ワールド座標系 [m]。
        yaw_rad: 配信した時点の自分のヨー角 (num_envs,) [rad]。
        command: 配信した時点に決めた速度指令 (num_envs, 3)（押す意図）。列は vx [m/s]、vy [m/s]、yaw_rate [rad/s]。
        stamp_s: 配信した時刻 (num_envs,)、エピソードの始めから数えたシミュレーション時間 [s]。
    """

    position_xy_m: torch.Tensor
    yaw_rad: torch.Tensor
    command: torch.Tensor
    stamp_s: torch.Tensor

    def where(self, mask: torch.Tensor, other: "PartnerMessage") -> "PartnerMessage":
        """mask (num_envs,) が True の環境は self、False の環境は other の値にした状態を返す。"""
        return PartnerMessage(
            **{
                field.name: _where_rows(mask, getattr(self, field.name), getattr(other, field.name))
                for field in fields(self)
            }
        )


@dataclass(frozen=True)
class PartnerReception:
    """受け手が判断に使える、最後に届いた相手の状態。

    Attributes:
        message: 最後に届いた状態。received が False の環境の値は意味を持たない（0 で埋めてある）。
        received: このエピソードで相手の状態が一度でも届いた環境のマスク (num_envs,)。
        age_s: 最後に届いた状態の古さ (num_envs,) [s]。現在の時刻から stamp_s を引き、max_age_s で頭打ちにする。
            一度も届いていない環境は max_age_s。
    """

    message: PartnerMessage
    received: torch.Tensor
    age_s: torch.Tensor


def _where_rows(mask: torch.Tensor, chosen: torch.Tensor, other: torch.Tensor) -> torch.Tensor:
    rows = mask.reshape(-1, *([1] * (chosen.dim() - 1)))
    return torch.where(rows, chosen, other)


def _zero_message(num_envs: int, device: torch.device) -> PartnerMessage:
    return PartnerMessage(
        position_xy_m=torch.zeros((num_envs, 2), device=device),
        yaw_rad=torch.zeros(num_envs, device=device),
        command=torch.zeros((num_envs, 3), device=device),
        stamp_s=torch.zeros(num_envs, device=device),
    )


class DelayedPartnerChannel:
    """1 方向（送り手 1 台 → 受け手 1 台）の通信の模擬。並列環境のバッチで、環境ごとに独立に届く・届かないが決まる。

    送り手が高レベル方策の周期ごとに配信した状態は、delay_steps 周期あとの判断に届く。届く時点で drop_probability の
    確率で欠落し、受け手は最後に届いた状態を使い続ける。エピソードの始め（reset_where）には、送られている途中の状態と
    受け手が持つ最後の状態を捨てる。前のエピソードの状態を次のエピソードの判断に使わないため。
    """

    def __init__(self, config: TeamCommunicationConfig, delay_steps: int, num_envs: int, device: torch.device) -> None:
        """何も届いていない状態で通信を用意する。

        Args:
            config: 欠落の確率と、古さの上限。
            delay_steps: 届くまでの遅れ（高レベル方策のステップ数）。1 以上。TeamMarlTaskConfig.communication_delay_steps()。
            num_envs: 並列環境の数。
            device: テンソルを置くデバイス。

        Raises:
            ValueError: delay_steps が 1 未満。
        """
        if delay_steps < 1:
            raise ValueError(f"delay_steps は 1 以上が必要: {delay_steps}")
        self._config = config
        self._delay_steps = delay_steps
        self._num_envs = num_envs
        self._device = device
        empty = _zero_message(num_envs, device)
        no_envs = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # 送られている途中の状態。先頭が最も古い。各要素は (状態, その状態が有効な環境のマスク)。
        # ステップ t に配信した状態は、そのステップの物理を進めたあとの判断（時刻 t + 1 周期）で 1 周期の遅れになる。
        # そのため途中に置くのは delay_steps - 1 個
        self._in_flight: list[tuple[PartnerMessage, torch.Tensor]] = [(empty, no_envs) for _ in range(delay_steps - 1)]
        self._latest = empty
        self._received = no_envs

    def reset_where(self, mask: torch.Tensor) -> None:
        """mask (num_envs,) が True の環境の、送られている途中の状態と最後に届いた状態を捨てる。"""
        self._in_flight = [(message, valid & ~mask) for message, valid in self._in_flight]
        self._received = self._received & ~mask

    def transmit(self, sent: PartnerMessage, generator: torch.Generator) -> None:
        """送り手が配信した状態を送り、次の判断が delay_steps 周期前の状態を使えるように届ける。

        高レベル方策の 1 ステップの物理を進めたあと、次の判断の前に 1 回呼ぶ。

        Args:
            sent: このステップに送り手が配信した状態。
            generator: 欠落を引く乱数生成器。drop_probability が 0 なら使わない（乱数の並びを変えない）。
        """
        self._in_flight.append((sent, torch.ones(self._num_envs, dtype=torch.bool, device=self._device)))
        arriving, valid = self._in_flight.pop(0)
        if self._config.drop_probability > 0.0:
            kept = torch.rand(self._num_envs, generator=generator, device=self._device) >= self._config.drop_probability
            valid = valid & kept
        self._latest = arriving.where(valid, self._latest)
        self._received = self._received | valid

    def reception(self, now_s: torch.Tensor) -> PartnerReception:
        """現在の時刻 now_s (num_envs,) [s] に受け手が判断に使える、最後に届いた相手の状態を返す。"""
        max_age_s = self._config.max_age_s
        age_s = torch.clamp(now_s - self._latest.stamp_s, max=max_age_s)
        age_s = torch.where(self._received, age_s, torch.full_like(age_s, max_age_s))
        zero = _zero_message(self._num_envs, self._device)
        return PartnerReception(message=self._latest.where(self._received, zero), received=self._received, age_s=age_s)
