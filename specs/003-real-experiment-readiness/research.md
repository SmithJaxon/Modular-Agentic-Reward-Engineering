# Research Notes: Real Experiment Readiness

## Decision 1: Remove Isaac from the active runtime scope

- Decision: Active code, tests, tooling, and docs should be Gymnasium-only for
  this worktree.
- Rationale: The current machine does not support Isaac well enough to justify
  keeping it on the critical path.

## Decision 2: Keep CartPole as the actual-backend smoke

- Decision: Preserve the existing `CartPole-v1` path as the lightweight real
  Gymnasium smoke.
- Rationale: It validates the session lifecycle cheaply and reproducibly.

## Decision 3: Use PPO for Gymnasium Humanoid scoring

- Decision: `Humanoid-v4` and `Humanoid-v5` use PPO training with checkpoint
  evaluation instead of the single-rollout heuristic.
- Rationale: The user wants the final metric shape to match the EUREKA paper.

## Decision 4: Measure Humanoid fitness with mean `x_velocity`

- Decision: Use Gymnasium Humanoid step info `x_velocity` as the task fitness
  metric.
- Rationale: The EUREKA appendix reports forward velocity for the Mujoco
  Humanoid environment.
