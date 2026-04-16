# Data Model: Real Experiment Readiness

## Actual Experiment Run

- `backend`: `gymnasium`
- `environment_id`: Gymnasium environment id such as `CartPole-v1` or `Humanoid-v4`
- `execution_mode`: `offline_test` or `actual_backend`
- `metrics`: persisted run metrics
- `artifact_refs`: manifest, metrics, traces, and optional media

## Humanoid PPO Evaluation Metrics

- `fitness_metric_name`: `mean_x_velocity`
- `per_run_best_mean_x_velocity`: list of best checkpoint metrics from each PPO run
- `checkpoint_mean_x_velocity`: list of checkpoint metrics for each PPO run
- `evaluation_run_count`: expected default `5`
- `checkpoint_count`: expected default `10`
- `train_timesteps`: PPO training budget

## Backend Runtime Status

- `backend`: `gymnasium`
- `ready`: whether the requested environment and execution prerequisites are available
- `status_reason`: actionable explanation
- `missing_prerequisites`: explicit operator next steps
