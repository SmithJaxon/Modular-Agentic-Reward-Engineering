# Data Model: LLM-Guided Reward Function Iteration

## Entity: OptimizationSession

Purpose: Represents one full iterative reward-design run.

Fields:
- `session_id` (string, unique, immutable)
- `objective_text` (string, required)
- `environment_id` (string, required)
- `environment_backend` (enum: gymnasium, isaacgym)
- `status` (enum: draft, running, paused, interrupted, completed, failed)
- `no_improve_limit` (integer, required, >0)
- `max_iterations` (integer, required, >0)
- `feedback_gate` (enum: none, one_required, both_required)
- `started_at` (datetime)
- `ended_at` (datetime, nullable)
- `stop_reason` (enum: user_interrupt, convergence, iteration_cap, api_failure_pause, error)
- `best_candidate_id` (string, nullable)

Validation rules:
- `no_improve_limit` and `max_iterations` MUST be present before status can
  transition to `running`.
- `environment_backend` MUST be present before status can transition to `running`.
- `ended_at` MUST be set when status enters terminal states.

## Entity: RewardCandidate

Purpose: One candidate reward-function revision within a session.

Fields:
- `candidate_id` (string, unique)
- `session_id` (string, foreign key -> OptimizationSession)
- `parent_candidate_id` (string, nullable)
- `iteration_index` (integer, >=0)
- `reward_definition` (text blob, required)
- `change_summary` (string, required)
- `aggregate_score` (float, nullable)
- `selected_final` (boolean, default false)
- `minor_robustness_risk_accepted` (boolean, default false)
- `created_at` (datetime)

Validation rules:
- `iteration_index` MUST be unique within a session.
- At most one candidate per session may have `selected_final=true`.

## Entity: ExperimentRun

Purpose: Records one executable evaluation for a candidate.

Fields:
- `run_id` (string, unique)
- `candidate_id` (string, foreign key -> RewardCandidate)
- `run_type` (enum: performance, reflection, robustness)
- `variant_label` (string, required)
- `seed` (integer)
- `status` (enum: queued, running, paused, completed, failed)
- `metrics` (object)
- `artifact_refs` (array of strings)
- `started_at` (datetime)
- `ended_at` (datetime, nullable)

Validation rules:
- `run_type=robustness` MUST include non-default variant labels.
- Completed runs MUST include at least one metric entry.

## Entity: ReflectionRecord

Purpose: Captures agent reasoning and proposed next-step edits.

Fields:
- `reflection_id` (string, unique)
- `candidate_id` (string, foreign key -> RewardCandidate)
- `source_run_ids` (array of strings)
- `summary` (string, required)
- `proposed_changes` (array of strings, required)
- `confidence` (float, 0.0-1.0)
- `created_at` (datetime)

## Entity: RobustnessAssessment

Purpose: Summarizes reward-hacking and overfitting risk for a candidate.

Fields:
- `assessment_id` (string, unique)
- `candidate_id` (string, foreign key -> RewardCandidate)
- `variant_count` (integer, >=1)
- `degradation_ratio` (float)
- `risk_level` (enum: low, medium, high)
- `risk_notes` (string)
- `created_at` (datetime)

## Entity: FeedbackEntry

Purpose: Stores human or peer feedback tied to a candidate.

Fields:
- `feedback_id` (string, unique)
- `candidate_id` (string, foreign key -> RewardCandidate)
- `source_type` (enum: human, peer)
- `score` (float, nullable)
- `comment` (string, required)
- `artifact_ref` (string, nullable)
- `created_at` (datetime)

Validation rules:
- `source_type=human` SHOULD include `artifact_ref` when a demonstration is used.
- `comment` cannot be empty.

## Entity: SelectionDecision

Purpose: Auditable final recommendation record.

Fields:
- `decision_id` (string, unique)
- `session_id` (string, foreign key -> OptimizationSession)
- `selected_candidate_id` (string, foreign key -> RewardCandidate)
- `decision_summary` (string, required)
- `tradeoff_rationale` (string, required when minor robustness risk accepted)
- `created_at` (datetime)

## Relationships

- One `OptimizationSession` has many `RewardCandidate`.
- One `RewardCandidate` has many `ExperimentRun`, `ReflectionRecord`,
  `RobustnessAssessment` (typically one latest summary), and `FeedbackEntry`.
- One `OptimizationSession` has one final `SelectionDecision`.

## State Transitions

OptimizationSession:
- `draft -> running`
- `running -> paused` (API failure retries exhausted)
- `running -> interrupted` (user stop)
- `running -> completed` (convergence or iteration cap)
- `running -> failed` (non-recoverable error)
- `paused -> running` (resume)

ExperimentRun:
- `queued -> running -> completed`
- `running -> failed`
- `running -> paused -> running`
