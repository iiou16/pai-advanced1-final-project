import csv
import dataclasses
import json
import math
from pathlib import Path

import pytest
import torch

from cooperative_go2.entities.gait_config import ValueRange
from cooperative_go2.entities.push_observation import PlanarBodyState, PushTaskSnapshot
from cooperative_go2.entities.push_robustness import PushRobustnessCondition
from cooperative_go2.entities.push_task_config import PushResetConfig
from cooperative_go2.entities.push_termination import PushTerminationReason
from cooperative_go2.entities.push_trial import (
    EmptyTrialSetError,
    PlanarPose,
    PushEpisodeResult,
    summarize_push_trials,
)
from cooperative_go2.interface_adapters.config_loader import PushTrainingSettings, load_push_training_settings
from cooperative_go2.interface_adapters.push_evaluation_adapters import (
    PolicyConditionMismatchError,
    require_same_policy_conditions,
)
from cooperative_go2.interface_adapters.push_evaluation_store import (
    TRIAL_COLUMNS,
    FilePushEvaluationStore,
    PushEvaluationConditions,
    PushEvaluationOutputExistsError,
)
from cooperative_go2.use_cases.evaluate_push import (
    EvaluationDidNotFinishError,
    PushEnvironmentView,
    PushEvaluationStep,
    evaluate_push,
)

HIGH_LEVEL_PERIOD_S = 0.1


def _result(reason: PushTerminationReason, steps: int, box_end: PlanarPose) -> PushEpisodeResult:
    return PushEpisodeResult(
        env_index=0,
        robot_start=PlanarPose(0.0, 0.0, 0.0),
        box_start=PlanarPose(1.0, 0.0, 0.0),
        goal_x_m=2.0,
        goal_y_m=0.0,
        box_end=box_end,
        termination_reason=reason,
        high_level_steps=steps,
        wall_time_s=1.0,
        collisions=0,
        falls=int(reason == PushTerminationReason.FALL),
    )


def test_summary_computes_requirement_metrics() -> None:
    results = [
        _result(PushTerminationReason.SUCCESS, 50, PlanarPose(2.0, 0.0, math.radians(10.0))),
        _result(PushTerminationReason.FALL, 20, PlanarPose(1.2, 0.5, 0.0)),
    ]

    summary = summarize_push_trials(results, HIGH_LEVEL_PERIOD_S)

    assert summary.success_rate == 0.5
    # 目標方向（+x）への移動量: 1.0 と 0.2。横へのずれは数えない
    assert summary.mean_progress_toward_goal_m == pytest.approx(0.6)
    assert summary.mean_final_goal_distance_m == pytest.approx((0.0 + math.hypot(0.8, 0.5)) / 2)
    assert summary.mean_rotation_deg == pytest.approx(5.0)
    assert summary.mean_success_time_s == pytest.approx(5.0)
    assert summary.fall_rate == 0.5


def test_summary_without_success_has_no_success_time() -> None:
    summary = summarize_push_trials(
        [_result(PushTerminationReason.TIME_OUT, 200, PlanarPose(1.0, 0.0, 0.0))], HIGH_LEVEL_PERIOD_S
    )

    assert summary.success_rate == 0.0
    assert summary.mean_success_time_s is None


def test_summary_of_no_trials_is_rejected() -> None:
    with pytest.raises(EmptyTrialSetError):
        summarize_push_trials([], HIGH_LEVEL_PERIOD_S)


class ScriptedEnvironment:
    """環境ごとに、何ステップ目にどの理由で試行が終わるかを決めておく評価用の環境。"""

    def __init__(self, finish_plan: list[tuple[int, PushTerminationReason]], max_steps: int) -> None:
        self._finish_plan = finish_plan
        self._max_steps = max_steps
        self._step = 0
        self.commands_seen: list[torch.Tensor] = []

    @property
    def num_envs(self) -> int:
        return len(self._finish_plan)

    @property
    def max_episode_steps(self) -> int:
        return self._max_steps

    def view(self) -> PushEnvironmentView:
        return PushEnvironmentView(snapshot=self._snapshot(), observation=torch.zeros(self.num_envs, 4))

    def step(self, commands: torch.Tensor) -> PushEvaluationStep:
        self.commands_seen.append(commands)
        self._step += 1
        # 2 試行目以降も同じ周期で終わる（最初の 1 試行だけが数えられることを確かめるため）
        done = torch.tensor([self._step % finish_step == 0 for finish_step, _ in self._finish_plan])
        reasons = torch.tensor(
            [int(reason) if finished else 0 for (_, reason), finished in zip(self._finish_plan, done, strict=True)]
        )
        return PushEvaluationStep(done=done, termination_reasons=reasons, pre_reset=self._snapshot(box_x=1.5))

    def _snapshot(self, box_x: float = 1.0) -> PushTaskSnapshot:
        count = self.num_envs

        def body(x: float) -> PlanarBodyState:
            return PlanarBodyState(
                position_xy_m=torch.tensor([[x, 0.0]] * count),
                yaw_rad=torch.zeros(count),
                lin_vel_xy_mps=torch.zeros(count, 2),
                yaw_rate_radps=torch.zeros(count),
            )

        return PushTaskSnapshot(robot=body(0.0), box=body(box_x), goal_xy_m=torch.tensor([[2.0, 0.0]] * count))


def test_evaluation_counts_only_first_trial_of_each_env() -> None:
    environment = ScriptedEnvironment([(2, PushTerminationReason.SUCCESS), (5, PushTerminationReason.FALL)], 10)

    results = evaluate_push(environment, lambda view: torch.zeros(2, 3), clock=lambda: 0.0)

    # 環境 0 は 2・4 ステップ目にも終わるが、最初の試行（2 ステップ）だけを数える
    assert [result.termination_reason for result in results] == [
        PushTerminationReason.SUCCESS,
        PushTerminationReason.FALL,
    ]
    assert [result.high_level_steps for result in results] == [2, 5]
    assert [result.falls for result in results] == [0, 1]
    assert results[0].box_end.x_m == pytest.approx(1.5)


def test_evaluation_passes_view_to_command_source() -> None:
    environment = ScriptedEnvironment([(1, PushTerminationReason.SUCCESS)], 10)
    seen_goals: list[torch.Tensor] = []

    def source(view: PushEnvironmentView) -> torch.Tensor:
        seen_goals.append(view.snapshot.goal_xy_m)
        return torch.ones(1, 3)

    evaluate_push(environment, source, clock=lambda: 0.0)

    assert torch.equal(environment.commands_seen[0], torch.ones(1, 3))
    assert torch.allclose(seen_goals[0], torch.tensor([[2.0, 0.0]]))


def test_evaluation_fails_when_a_trial_never_ends() -> None:
    environment = ScriptedEnvironment([(100, PushTerminationReason.TIME_OUT)], max_steps=10)

    with pytest.raises(EvaluationDidNotFinishError, match=r"\[0\]"):
        evaluate_push(environment, lambda view: torch.zeros(1, 3), clock=lambda: 0.0)


def _conditions() -> PushEvaluationConditions:
    return PushEvaluationConditions(
        run_id="e1-test",
        commit_sha="abc123",
        scene_id="S1_push",
        seed=2001,
        algorithm="rule_based",
        checkpoint=None,
        mass_kg=2.0,
        friction=1.0,
        high_level_period_s=HIGH_LEVEL_PERIOD_S,
        gait_steps_per_high_level_step=5,
        num_envs=1,
        robustness_condition=None,
    )


def test_store_writes_one_row_per_trial_with_requirement_columns(tmp_path: Path) -> None:
    store = FilePushEvaluationStore(tmp_path / "e1-test")
    store.initialize()
    result = _result(PushTerminationReason.SUCCESS, 50, PlanarPose(2.0, 0.0, 0.0))

    store.save(_conditions(), [result], summarize_push_trials([result], HIGH_LEVEL_PERIOD_S))

    with (tmp_path / "e1-test" / "trials.csv").open(encoding="utf-8") as trials_file:
        rows = list(csv.DictReader(trials_file))
    assert tuple(rows[0]) == TRIAL_COLUMNS
    assert rows[0]["termination_reason"] == "SUCCESS"
    assert rows[0]["checkpoint"] == ""
    assert float(rows[0]["elapsed_sim_s"]) == pytest.approx(5.0)
    assert int(rows[0]["sim_steps"]) == 250
    summary = json.loads((tmp_path / "e1-test" / "summary.json").read_text(encoding="utf-8"))
    assert summary["summary"]["success_rate"] == 1.0


def test_store_writes_robustness_condition_to_summary(tmp_path: Path) -> None:
    store = FilePushEvaluationStore(tmp_path / "robustness-test")
    store.initialize()
    reset = PushResetConfig(
        box_xy_jitter_m=0.1, goal_distance_m=ValueRange(1.0, 1.0), goal_direction_deg=ValueRange(-60.0, 60.0)
    )
    condition = PushRobustnessCondition(
        name="delay_0.10s", box_mass_kg=2.0, box_friction=1.0, reset=reset, command_delay_gait_steps=5
    )
    result = _result(PushTerminationReason.SUCCESS, 50, PlanarPose(2.0, 0.0, 0.0))

    conditions = dataclasses.replace(_conditions(), robustness_condition=condition)
    store.save(conditions, [result], summarize_push_trials([result], HIGH_LEVEL_PERIOD_S))

    summary = json.loads((tmp_path / "robustness-test" / "summary.json").read_text(encoding="utf-8"))
    assert summary["conditions"]["robustness_condition"] == dataclasses.asdict(condition)


def test_store_refuses_to_overwrite_existing_output(tmp_path: Path) -> None:
    (tmp_path / "e1-test").mkdir()

    with pytest.raises(PushEvaluationOutputExistsError):
        FilePushEvaluationStore(tmp_path / "e1-test").initialize()


def _training_settings() -> PushTrainingSettings:
    return load_push_training_settings(Path(__file__).resolve().parents[2] / "configs" / "push_single.yaml")


def test_policy_conditions_allow_different_layout() -> None:
    trained = _training_settings()
    wide = dataclasses.replace(trained.task.reset, goal_direction_deg=ValueRange(low=-60.0, high=60.0))
    evaluated = dataclasses.replace(trained, task=dataclasses.replace(trained.task, reset=wide))

    # 配置は汎化を確かめるために変えてよい
    require_same_policy_conditions(trained, evaluated)


def test_policy_conditions_reject_different_observation_scales() -> None:
    trained = _training_settings()
    scales = dataclasses.replace(trained.task.observation, position=trained.task.observation.position * 2)
    evaluated = dataclasses.replace(trained, task=dataclasses.replace(trained.task, observation=scales))

    with pytest.raises(PolicyConditionMismatchError, match="task.observation"):
        require_same_policy_conditions(trained, evaluated)
