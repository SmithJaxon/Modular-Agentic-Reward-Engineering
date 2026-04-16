---

description: "Task list for Gymnasium-only real experiment readiness and Humanoid PPO evaluation"

---

# Tasks: Real Experiment Readiness

## Phase 1: Scope Reset

- [x] T001 Remove Isaac-specific runtime branches from `src/rewardlab/`
- [x] T002 Remove Isaac-specific tests, fixtures, and quality-wrapper paths from `tests/` and `tools/`
- [x] T003 Update schemas and contract validation so `environment_backend` is Gymnasium-only
- [x] T004 Add checked-in Gymnasium Humanoid fixtures in `tools/fixtures/`

## Phase 2: Stable Gymnasium Runtime

- [x] T005 Keep the actual Gymnasium smoke path working for `CartPole-v1`
- [x] T006 Keep real robustness assessment and artifact persistence on the shared Gymnasium path
- [x] T007 Add offline-safe coverage for runtime failure fallback in peer feedback and execution service

## Phase 3: Humanoid PPO Evaluation

- [x] T008 Add a Gymnasium Humanoid PPO evaluation protocol in `src/rewardlab/experiments/gymnasium_runner.py`
- [x] T009 Add fake-trainer integration coverage for Humanoid PPO checkpoint aggregation in `tests/integration/test_gymnasium_real_experiment.py`
- [x] T010 Fail Humanoid execution with an explicit prerequisite error when `stable_baselines3` is unavailable
- [x] T011 With user approval, install `stable-baselines3` into `.venv` and record the exact command and versions in docs
- [x] T012 Run an actual `Humanoid-v4` session using the checked-in fixtures and capture the resulting report evidence

## Phase 4: Docs And Handoff

- [x] T013 Rewrite `README.md` for Gymnasium-only scope plus Humanoid PPO usage
- [x] T014 Rewrite `specs/003-real-experiment-readiness/` planning docs for the Gymnasium Humanoid target
- [x] T015 Rewrite `NEXT_AGENT_HANDOFF.md` and `AGENTS.md` to reflect current scope and blockers

## Phase 5: Agent-Driven Reward Iteration

- [x] T016 Add an explicit reward-designer abstraction with deterministic and OpenAI-backed modes
- [x] T017 Wire actual-backend session stepping to use the latest reflection and run metrics during reward generation
- [x] T018 Pause sessions on reward-design failures instead of silently fabricating local comment-only revisions
- [x] T019 Add unit and integration coverage for OpenAI-backed reward iteration and failure handling

## Current Remaining Work

- Optional broader full-suite validation after the Gymnasium-only pivot
- With user approval, run a paid multi-iteration Humanoid session with `REWARDLAB_REWARD_DESIGN_MODE=openai`
- Optional cleanup of residual historical Isaac wording outside the active runtime docs
