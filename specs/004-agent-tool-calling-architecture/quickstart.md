# Quickstart: Agent Tool-Calling Experiments

This document describes the planned operator workflow for the new
`agent_tools` control mode.

Status: implemented beta in this worktree via `rewardlab experiment ...`
commands. The legacy `rewardlab session ...` path still exists but is no longer
the recommended path for autonomous optimization.

## 1) Choose a Spec

Reference specs:

- `tools/fixtures/experiments/agent_humanoid_balanced.yaml`
- `tools/fixtures/experiments/agent_humanoid_high_budget.yaml`
- `tools/fixtures/experiments/agent_cartpole_lowcost.yaml`

## 2) Validate

```powershell
.\.venv\Scripts\rewardlab.exe experiment validate `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --json
```

## 3) Run

```powershell
.\.venv\Scripts\rewardlab.exe experiment run `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --json
```

## 4) Monitor

```powershell
.\.venv\Scripts\rewardlab.exe experiment status `
  --experiment-id <EXPERIMENT_ID> `
  --json

.\.venv\Scripts\rewardlab.exe experiment trace `
  --experiment-id <EXPERIMENT_ID> `
  --json
```

## 5) Stop or Resume

```powershell
.\.venv\Scripts\rewardlab.exe experiment stop --experiment-id <EXPERIMENT_ID> --json
.\.venv\Scripts\rewardlab.exe experiment resume --experiment-id <EXPERIMENT_ID> --json
```

## 6) Human Feedback (When Enabled)

If the experiment pauses with `stop_reason=awaiting_human_feedback`, submit
feedback and resume:

```powershell
.\.venv\Scripts\rewardlab.exe experiment submit-human-feedback `
  --experiment-id <EXPERIMENT_ID> `
  --candidate-id <CANDIDATE_ID> `
  --comment "Looks stable enough to continue." `
  --request-id <REQUEST_ID> `
  --json

.\.venv\Scripts\rewardlab.exe experiment resume `
  --experiment-id <EXPERIMENT_ID> `
  --json
```

## 7) Multi-Seed Benchmark

Run the same spec across explicit seeds and export aggregate stats:

```powershell
.\.venv\Scripts\rewardlab.exe experiment benchmark-run `
  --file tools/fixtures/experiments/agent_humanoid_balanced.yaml `
  --seed 1 `
  --seed 2 `
  --seed 3 `
  --json
```

The benchmark report is written under:
`.rewardlab/reports/agent_benchmarks/<BENCHMARK_ID>.benchmark.json`

## Notes

- Budgets in the spec are hard runtime constraints.
- Human-feedback tools are unavailable unless explicitly enabled in the spec.
- In `agent_tools` mode, the controller agent chooses action sequence and stop timing.
- Analysis tools (`estimate_cost_and_risk`, `compare_candidates`) can consume
  analyzer-model tokens when credentials are available.
- Benchmark reports include score distributions, improvement rates, confidence
  intervals, action-mix distributions, stop-reason counts, and resource-efficiency
  metrics such as improvement-per-1k-tokens.
