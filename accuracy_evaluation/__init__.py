"""Static-point gaze-tracking evaluation toolkit."""

from accuracy_evaluation.evaluate import (
    CONDITIONS,
    TrialRecorder,
    aggregate_trials,
    build_pixel_mapper,
    compute_trial_metrics,
    generate_static_points,
    run_live_trial,
    run_mock_trial,
)

__all__ = [
    "CONDITIONS",
    "TrialRecorder",
    "aggregate_trials",
    "build_pixel_mapper",
    "compute_trial_metrics",
    "generate_static_points",
    "run_live_trial",
    "run_mock_trial",
]
