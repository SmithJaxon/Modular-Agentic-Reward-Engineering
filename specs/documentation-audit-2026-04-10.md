# Documentation Audit (2026-04-10)

Update: later on 2026-04-10, Phase 0-1 agentic scaffold commands were
implemented; status notes in this file were updated accordingly.

## Scope

Reviewed and reconciled:
- `README.md`
- `specs/001-iterative-reward-design/quickstart.md`
- `specs/agentic-tool-calling-plan.md`
- `configs/agentic/README.md`
- `configs/agentic/*.yaml`

## Conflicts Found And Resolved

1. Agentic command implication mismatch:
- Issue: `configs/agentic/README.md` implied `rewardlab agent run --spec` was
  currently available.
- Resolution: Added explicit status that agentic runtime is planned and not yet
  implemented; pointed readers to current `rewardlab session ...` workflow.

2. Fixture availability mismatch:
- Issue: quickstart stated only CartPole fixtures were shipped.
- Resolution: Updated quickstart to state CartPole and Humanoid fixtures are
  available by default.

3. Implemented-vs-planned architecture ambiguity:
- Issue: no explicit status boundary between current iterative runtime and
  planned agentic runtime.
- Resolution: added “Agentic Rework Status (2026-04-10)” section in README and
  status marker in `specs/agentic-tool-calling-plan.md`.

## Current Documentation Contract

- Executable today:
  - `rewardlab session start/step/report/stop`
  - PPO-backed Gymnasium flow documented in README and quickstart
- In-progress scaffold:
  - `rewardlab agent run/status/events/report`
  - Agentic decision-turn orchestration scaffold from
    `specs/agentic-tool-calling-plan.md`
  - YAML inputs in `configs/agentic/`
  - Active tools: `run_experiment`, `run_probe_suite`, `compare_candidates`,
    `export_report`, `budget_snapshot`, `read_artifact`

## Known Intentional Gaps

- `configs/agentic/ant_main.yaml` references example files that are not yet in
  the repo:
  - `tools/fixtures/objectives/ant.txt`
  - `tools/fixtures/rewards/ant_baseline.py`
- This is documented in `configs/agentic/README.md` as expected setup for
  future use.
