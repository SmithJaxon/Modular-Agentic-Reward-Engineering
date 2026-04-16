# Implementation Plan: Real Experiment Readiness

**Branch**: `003-real-experiment-readiness`  
**Date**: 2026-04-06  
**Spec**: [spec.md](C:\Users\smith\LocalOnlyClasses\AdvAi\Project-agent-autonomous-pass\specs\003-real-experiment-readiness\spec.md)

## Summary

RewardLab now targets a Gymnasium-only real execution path. `CartPole-v1`
remains the lightweight actual-backend smoke, while `Humanoid-v4` is the main
full-pipeline target using PPO-based scoring that mirrors the EUREKA paper's
evaluation shape.

## Architecture

- `src/rewardlab/experiments/gymnasium_runner.py`
  - single-rollout scoring for lightweight environments
  - PPO training plus checkpoint evaluation for `Humanoid-v4` / `Humanoid-v5`
- `src/rewardlab/experiments/execution_service.py`
  - reward loading, artifact writing, and failed-run handling
- `src/rewardlab/orchestrator/session_service.py`
  - unchanged lifecycle integration using the Gymnasium runner as the execution surface
- `tools/fixtures/`
  - checked-in CartPole and Humanoid fixtures for operator runs

## Validation Strategy

1. Keep offline unit, contract, and integration coverage green.
2. Keep the real Gymnasium smoke wrapper green on `CartPole-v1`.
3. After approval, install PPO dependencies and run a real `Humanoid-v4`
   session end to end.

## Status

- Humanoid PPO runtime validation has been completed in this worktree.
- OpenAI-backed reward iteration has been validated for real runs.
- Follow-on architecture work is now tracked in
  `specs/004-agent-tool-calling-architecture/`.
