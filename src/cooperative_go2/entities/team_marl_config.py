"""2 台の協調学習（IPPO・MAPPO）の学習環境の設定の値オブジェクト。

既定値はここに持たず、設定ファイル（configs/push_multi.yaml）にすべての値を書く。
各クラスは生成時に値の不変条件を検証し、違反すれば InvalidTeamMarlConfigError を送出する。
1 台の箱押しと意味が同じ部品（エピソードの進め方・指令の範囲・有効範囲・発散・観測の正規化）は push の設定を使う。
"""

from dataclasses import dataclass, fields

from cooperative_go2.entities.config_validation import (
    ConfigValueChecker,
    InvalidConfigError,
    ceil_ratio,
    is_integer_ratio,
)
from cooperative_go2.entities.push_config import PushArenaConfig, PushDivergenceConfig, PushObservationScales
from cooperative_go2.entities.push_task_config import PushCommandLimits, PushEpisodeConfig
from cooperative_go2.entities.team_task_config import TeamResetConfig


class InvalidTeamMarlConfigError(InvalidConfigError):
    """2 台の協調学習の設定値が不変条件を満たさないことを表す。"""


_check = ConfigValueChecker(InvalidTeamMarlConfigError)
# 自分の側の符号。押す方向の左が +1、右が -1
_SIDE_SIGNS = frozenset({1.0, -1.0})


@dataclass(frozen=True)
class TeamCommunicationConfig:
    """相手の状態を Actor に届ける通信の模擬。

    Attributes:
        delay_s: 相手が状態を配信してから届くまでの遅れ [s]（シミュレーション時間）。高レベル方策の周期の 1 倍以上の
            整数倍。2 台は同じ時刻に指令を決めるので、相手がその時刻に決めた指令は早くても次の周期にしか届かない。
            1 周期なら、相手が 1 つ前の周期に配信した状態が届く。
        drop_probability: 1 回の配信が届かない確率。[0, 1) の値。届かなければ、受け手は最後に届いた状態を使い続ける。
        max_age_s: 観測に入れる「最後に届いた相手の状態の古さ」の上限 [s]。正の値。これより古い状態は上限の古さとして
            観測に入れる。古さの値の範囲を限り、観測の正規化を一定にするため。
    """

    delay_s: float
    drop_probability: float
    max_age_s: float

    def __post_init__(self) -> None:
        _check.positive("communication.delay_s", self.delay_s)
        _check.unit_interval("communication.drop_probability", self.drop_probability)
        # 確率 1 では一度も届かず、相手の状態の観測が常に空になるため除く
        if self.drop_probability >= 1.0:
            raise InvalidTeamMarlConfigError(f"communication.drop_probability は 1 未満が必要: {self.drop_probability}")
        _check.positive("communication.max_age_s", self.max_age_s)

    def delay_steps(self, high_level_period_s: float) -> int:
        """遅れを高レベル方策のステップ数で返す。

        Raises:
            InvalidTeamMarlConfigError: 遅れが周期の 1 倍以上の整数倍でない。
        """
        if not is_integer_ratio(self.delay_s, high_level_period_s) or round(self.delay_s / high_level_period_s) < 1:
            raise InvalidTeamMarlConfigError(
                f"communication.delay_s ({self.delay_s}) は high_level_period_s ({high_level_period_s}) の 1 倍以上の"
                "整数倍にする"
            )
        return round(self.delay_s / high_level_period_s)


@dataclass(frozen=True)
class TeamMarlRewardScales:
    """報酬の各項の係数。正は報酬、負は減点。0 ならその項を使わない。フィールド名は TeamMarlRewardTerms と一致させる。

    box_progress・success・box_rotation は 2 台で共有する項、それ以外はロボットごとの項。
    """

    box_progress: float
    success: float
    box_rotation: float
    approach_progress: float
    intended_face_contact: float
    robot_fall: float
    out_of_bounds: float
    robot_collision: float
    control_effort: float

    def __post_init__(self) -> None:
        for field in fields(self):
            _check.finite(f"reward.scales.{field.name}", getattr(self, field.name))


@dataclass(frozen=True)
class TeamMarlRewardConfig:
    """報酬の計算に使うパラメータ。

    Attributes:
        intended_face_min_cos: 「意図した面を押している」とみなす、ロボット→箱の向きと箱→退避領域の向きの
            なす角の余弦の下限。-1 以上 1 以下。
        min_robot_separation_m: 2 台の胴体の中心がこの距離 [m] より近ければ、ロボット同士の衝突とみなす。正の値。
            Go2 の胴体の幅（約 0.31 m）に合わせ、横に並んで押す間隔（0.5 m）より短くする。
        push_standoff_m: 近づく報酬の目標にする押す位置の、箱の中心から押す方向の逆への距離 [m]。正の値。
        push_lateral_offset_m: 押す位置の、押す方向の線から自分の側（agent_side_signs）への横のずれの大きさ [m]。0 以上。
        scales: 各項の係数。
    """

    intended_face_min_cos: float
    min_robot_separation_m: float
    push_standoff_m: float
    push_lateral_offset_m: float
    scales: TeamMarlRewardScales

    def __post_init__(self) -> None:
        if not -1.0 <= self.intended_face_min_cos <= 1.0:
            raise InvalidTeamMarlConfigError(
                f"reward.intended_face_min_cos は -1 以上 1 以下が必要: {self.intended_face_min_cos}"
            )
        _check.positive("reward.min_robot_separation_m", self.min_robot_separation_m)
        _check.positive("reward.push_standoff_m", self.push_standoff_m)
        _check.non_negative("reward.push_lateral_offset_m", self.push_lateral_offset_m)


@dataclass(frozen=True)
class TeamMarlTaskConfig:
    """2 台の協調学習の環境に必要な、学習アルゴリズムによらない設定一式。

    Attributes:
        episode: エピソードの進め方。転倒の判定はロール・ピッチの両方に使う。
        commands: 各ロボットの高レベル方策が出せる速度指令の範囲。
        reset: 試行の始めのロボットの開始位置の散らし方（規則ベースの協調と同じ引き方）。
        agent_side_signs: ロボットごとの、押す方向から見た自分の側（左 +1、右 -1）。シーンの robots と同じ順・同じ数。
            パラメータを 2 台で共有しても、自分がどちらの側かを観測から分かるようにするため。
        success_hold_time_s: 箱の中心が退避領域に入った状態を続けたら成功とする時間 [s]。正の値。
        arena: シーンの有効範囲。
        divergence: 物理計算の発散とみなす条件。
        communication: 相手の状態を届ける通信の模擬。
        observation: 観測の正規化係数。
        reward: 報酬の計算に使うパラメータ。
    """

    episode: PushEpisodeConfig
    commands: PushCommandLimits
    reset: TeamResetConfig
    agent_side_signs: tuple[float, ...]
    success_hold_time_s: float
    arena: PushArenaConfig
    divergence: PushDivergenceConfig
    communication: TeamCommunicationConfig
    observation: PushObservationScales
    reward: TeamMarlRewardConfig

    def __post_init__(self) -> None:
        for index, sign in enumerate(self.agent_side_signs):
            if sign not in _SIDE_SIGNS:
                raise InvalidTeamMarlConfigError(f"agent_side_signs[{index}] は 1 か -1 にする: {sign}")
        if len(set(self.agent_side_signs)) != len(self.agent_side_signs):
            raise InvalidTeamMarlConfigError(f"agent_side_signs はロボットごとに別の側にする: {self.agent_side_signs}")
        _check.positive("success_hold_time_s", self.success_hold_time_s)
        # 遅れが周期の整数倍であることを、読み込んだ時点で確かめる
        self.communication.delay_steps(self.episode.high_level_period_s)

    @property
    def num_agents(self) -> int:
        """ロボット（Actor）の数。"""
        return len(self.agent_side_signs)

    def required_hold_steps(self) -> int:
        """成功判定に必要な、箱の中心が退避領域に入っていると続けて観測する高レベル方策のステップ数を返す。

        規則ベースの協調の VERIFY と同じ数え方にする。VERIFY は入ったと観測した時刻に始まり、そこから
        success_hold_time_s たった時刻の観測まで入ったままなら DONE になるので、入っている観測は
        success_hold_time_s / 周期 より 1 回多い（1.0 s・0.1 s なら 11 回）。
        """
        return ceil_ratio(self.success_hold_time_s, self.episode.high_level_period_s) + 1

    def communication_delay_steps(self) -> int:
        """相手の状態が届くまでの遅れを、高レベル方策のステップ数で返す。"""
        return self.communication.delay_steps(self.episode.high_level_period_s)
