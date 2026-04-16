# Agentic Run Specs

These files are run specs for the in-progress agentic runtime.

Current status:
- `rewardlab agent run --spec-file <file>` is implemented as an in-progress
  agentic runtime with decision-turn traces.
- Primary planning can run in two modes:
  - `planner_provider: heuristic` (default fallback policy)
  - `planner_provider: openai` (LLM decides next tool call or stop)
- Active tools in the current build:
  - `run_experiment` (adapter-backed execution)
  - `run_probe_suite` (robustness-runner integration)
  - `compare_candidates` (aggregate ranking)
  - `export_report` (JSON artifact export)
  - `budget_snapshot`
  - `read_artifact`
- Use `rewardlab session ...` for full production PPO experiment execution until
  tool migration phases are complete.

## What To Change Most Often

- `environment.id`: target env (example: `Humanoid-v4`, `Ant-v4`)
- `agent.primary_model`: model version
- `agent.reasoning_effort`: `low|medium|high`
- `agent.planner_provider`: `heuristic|openai`
- `agent.planner_fallback_enabled`: when `false`, stop instead of heuristic fallback
- `agent.planner_system_prompt_file`: optional extra planner instructions
- `agent.planner_context_window`: recent context rows included in planner prompt
- `agent.planner_max_output_tokens`: planner response budget
- `agent.planner_max_retries`: planner retry attempts after invalid output
- `budgets.hard.*`: hard caps
- `budgets.soft.*`: guidance thresholds
- `decision.max_turns`: upper bound on decision turns
- `decision.min_candidates_before_compare`: earliest compare trigger size
- `decision.compare_every_new_candidates`: compare cadence as candidate pool grows
- `decision.require_probe_before_compare`: require robustness evidence before compare
- `training_defaults.ppo_policy_hidden_sizes`: PPO policy MLP widths
- `training_defaults.ppo_activation_fn`: PPO policy activation (`tanh|relu|elu|...`)
- `training_defaults.ppo_gamma`: discount factor
- `training_defaults.ppo_gae_lambda`: GAE lambda
- `training_defaults.ppo_clip_range`: PPO clip range
- `training_defaults.ppo_n_epochs`: PPO optimization epochs per rollout
- `training_defaults.ppo_ent_coef`: entropy coefficient
- `training_defaults.ppo_vf_coef`: value-function loss coefficient
- `training_defaults.ppo_max_grad_norm`: gradient clipping norm
- `tools.enabled`: what the primary agent is allowed to call

## Budget Knob Meanings

- `max_wall_clock_minutes`: hard elapsed-time limit
- `max_training_timesteps`: total PPO train steps budget
- `max_evaluation_episodes`: total eval episode budget
- `max_api_input_tokens`: total model input token cap
- `max_api_output_tokens`: total model output token cap
- `max_api_usd`: total API spend cap
- `max_calls_per_model`: per-model quota map

## Stop Guidance Knobs

- `plateau_window_turns`: lookback window for gains
- `min_delta_return`: minimum meaningful gain in env return
- `min_gain_per_1k_usd`: stop if value/cost drops below threshold
- `risk_ceiling`: refuse candidates above this risk tier
- `target_env_return`: stop early if objective target is met

Note on score scale:
- In the current examples, `training_defaults.execution_mode` is set to
  `deterministic`, so `target_env_return` and `min_delta_return` are tuned for
  normalized scores near `[0.0, 1.0]`.
- If you switch to PPO execution, retune these values for raw environment
  returns (often much larger magnitudes).

## Fixture Notes

- `ant_main.yaml` references `tools/fixtures/objectives/ant.txt` and
  `tools/fixtures/rewards/ant_baseline.py` as example inputs. Create those
  files before using an Ant profile in the future agentic runtime.

## OpenAI Planner Example

Use `humanoid_main_openai.yaml` to let the primary agent call OpenAI for
decision turns:

```powershell
venv\Scripts\rewardlab.exe agent run --spec-file configs\agentic\humanoid_main_openai.yaml --json
```

Prerequisites:
- install optional dependency: `venv\Scripts\python.exe -m pip install -e .[llm]`
- set `OPENAI_API_KEY` (or place it in repo-local `.env`)

Verification tip:
- inspect `report.json` decision rows and confirm `decision_source: "llm_openai"`
  appears when planner calls succeed.
- planner token usage is reflected in `remaining_budget.remaining_api_input_tokens`
  and `remaining_budget.remaining_api_output_tokens`.
- planner validation failures now emit `planner.validation_failed` events and
  are summarized in report fields:
  - `planner_feedback`
  - `planner_feedback_summary`

Strict mode example:
- `humanoid_main_openai_strict.yaml` sets `planner_fallback_enabled: false`.
  In strict mode, planner unavailability causes an immediate stop instead of
  heuristic fallback.

Dexterous hand example:
- `adroit_hand_pen_openai_eurekaish.yaml` provides a high-budget Adroit hand
  profile using Eureka-like PPO architecture settings on Gymnasium Robotics.

Improvement tracking:
- Keep architecture improvement ideas updated in:
  `specs/agentic-improvement-backlog.md`
