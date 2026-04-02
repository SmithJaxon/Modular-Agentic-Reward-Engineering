# Feature Specification: LLM-Guided Reward Function Iteration

**Feature Branch**: `001-iterative-reward-design`  
**Created**: 2026-04-02  
**Status**: Draft  
**Input**: User description: "Create a pipeline that iteratively improves reinforcement learning reward functions through automated experiments, reflection, robustness checks against reward hacking, human feedback from visual demonstrations, and external peer-agent feedback until convergence or user interruption."

## Clarifications

### Session 2026-04-02

- Q: What automatic stop rule should govern optimization sessions? -> A: Stop when either no improvement is observed for N consecutive iterations or a maximum total iteration limit is reached.
- Q: Should convergence thresholds use defaults or be user-specified? -> A: The no-improvement threshold and maximum iteration limit are session input parameters, not fixed constants.
- Q: Should final selection enforce a hard robustness gate? -> A: Use robustness-aware selection guidance with agent discretion, allowing minor robustness issues when performance gains are compelling and explicitly justified.
- Q: Should human/peer feedback be mandatory before final recommendation? -> A: Feedback gating is a session parameter that can require none, one, or both feedback channels.
- Q: How should model/API failures be handled during a session? -> A: Retry with bounded attempts and backoff; if retries fail, pause the session and preserve resumable state.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Iterative Reward Optimization Loop (Priority: P1)

As a reinforcement learning researcher, I want the system to iteratively propose,
test, and refine reward function candidates so I can obtain a stronger final reward
function with clear evidence.

**Why this priority**: This is the core value of the feature; without this loop,
none of the other checks or feedback flows matter.

**Independent Test**: Start one optimization session on a known environment and
verify that multiple candidate iterations are evaluated, reflected on, ranked, and
that the current best candidate can be exported at any time.

**Acceptance Scenarios**:

1. **Given** an environment objective and an initial baseline candidate,
   **When** the user starts a session, **Then** the system runs an evaluation,
   records outcomes, and proposes a revised reward candidate.
2. **Given** an active session with several completed iterations,
   **When** the user interrupts the process,
   **Then** the system stops safely and returns the best-known reward function with
   its supporting experiment and reflection history.

---

### User Story 2 - Reward Hacking and Overfitting Detection (Priority: P2)

As a reinforcement learning researcher, I want robustness experiments that vary
training setups so I can detect reward hacking and avoid reward functions that only
work in one narrow setup.

**Why this priority**: A high-performing reward function is not useful if it is
fragile or exploitative under slight condition changes.

**Independent Test**: Run a session that includes robustness checks and confirm
that candidates are assessed under varied conditions and explicitly flagged when
generalization risk is detected.

**Acceptance Scenarios**:

1. **Given** a candidate that performs well in the primary setup,
   **When** robustness experiments are run across varied training conditions,
   **Then** the system reports whether performance remains stable and flags potential
   reward-hacking risk when it does not.

---

### User Story 3 - Human and External Peer Feedback (Priority: P3)

As a reinforcement learning researcher, I want human and independent peer feedback
integrated into the iteration process so the final reward design reflects both
observed behavior quality and external critique.

**Why this priority**: Human and independent review improves confidence that the
recommended reward is aligned with real goals, not just internal metrics.

**Independent Test**: Request feedback on an iteration, submit human reviewer
comments and peer review comments, and verify both are captured and influence the
next candidate rationale.

**Acceptance Scenarios**:

1. **Given** a completed experiment iteration,
   **When** the user requests review,
   **Then** the system presents a demonstration for human review, records the
   feedback, and stores it with that iteration.
2. **Given** the same iteration,
   **When** independent peer feedback is requested,
   **Then** the system captures peer critique from an isolated review context and
   includes it in the next iteration summary.

### Edge Cases

- When the model service is unavailable, retries use bounded attempts with backoff;
  if retries fail, the session is paused and resumable state is preserved.
- What happens when the user interrupts while experiments are still running?
- How does the system behave when context or resource limits are reached mid-session?
- How does the system reconcile conflicting feedback from human and peer reviewers?
- Sessions auto-stop when no improvement persists for N consecutive iterations or when the maximum total iteration limit is reached.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST create an optimization session from a user-provided
  objective and baseline reward function.
- **FR-002**: System MUST run at least one evaluation experiment for each candidate
  reward iteration and record comparable outcomes.
- **FR-003**: System MUST generate a reflection for each iteration that explains
  observed behavior and proposed next changes.
- **FR-004**: System MUST maintain a ranked history of reward candidates and their
  evidence across the full session.
- **FR-005**: Users MUST be able to interrupt a running session at any time and
  receive the best-known reward candidate with supporting evidence.
- **FR-006**: System MUST run robustness experiments that vary training conditions
  to evaluate generalization and reward-hacking risk.
- **FR-007**: System MUST flag candidates as risk-prone when robustness performance
  materially degrades relative to primary evaluation.
- **FR-008**: System MUST support human feedback collection by presenting a visual
  demonstration of candidate behavior and storing structured reviewer input.
- **FR-009**: System MUST support independent peer feedback generated from an
  isolated review context and associate that critique with the iteration.
- **FR-010**: System MUST isolate orchestration context from experiment execution
  and peer review contexts, sharing only bounded summaries between them.
- **FR-011**: System MUST allow operators to provide model-access credentials via
  secure runtime configuration, with no secret values stored in source artifacts.
- **FR-012**: System MUST persist best-known candidate state and evidence when a
  session exits due to interruption or context/resource limits.
- **FR-013**: System MUST support automatic session termination when either
  (a) no improvement is observed for N consecutive iterations or (b) a maximum
  total iteration limit is reached.
- **FR-014**: System MUST require both convergence thresholds (no-improvement
  streak limit and maximum iteration limit) as user-provided session parameters
  and MUST validate them before session start.
- **FR-015**: System MUST select final candidates using a robustness-aware
  multi-signal policy that considers primary performance, robustness outcomes,
  human feedback, and peer feedback without forcing a rigid single-metric rule.
- **FR-016**: When the selected final candidate includes known minor robustness
  risk, the system MUST record explicit agent rationale describing why the
  performance tradeoff was accepted over safer alternatives.
- **FR-017**: System MUST support a session-level feedback gating parameter that
  specifies whether final recommendation requires no feedback, at least one
  feedback channel, or both human and peer feedback.
- **FR-018**: System MUST handle model/API failures using bounded retries with
  backoff; if retries are exhausted, the system MUST pause the session and
  persist resumable state without losing best-known candidate evidence.

### Constitution Alignment Requirements

- **CAR-001**: The feature MUST decompose responsibilities into modular components
  for orchestration, experimentation, robustness analysis, and feedback handling.
- **CAR-002**: Planning artifacts MUST identify touched files for iteration
  orchestration, experiment execution, feedback management, and session reporting.
- **CAR-003**: Non-trivial routines, including stop-condition evaluation,
  risk-flagging decisions, and feedback synthesis, MUST be documented with
  structured method headers.
- **CAR-004**: Tasks MUST include explicit cleanup of deprecated reward variants,
  stale experiment logic, and unused decision paths before merge.

### Key Entities *(include if feature involves data)*

- **Optimization Session**: A full run context containing objectives,
  configuration, iteration history, stop reason, and final recommendation.
- **Reward Candidate**: A specific reward-function proposal version with rationale,
  parameters, and ranking status.
- **Experiment Result**: Structured outcome of a candidate evaluation, including
  behavior metrics, run metadata, and evaluation summary.
- **Reflection Record**: Analysis note that interprets experiment outcomes and
  proposes concrete next-step modifications.
- **Robustness Assessment**: A set of varied-condition evaluation outcomes and a
  resulting risk classification for reward hacking or overfitting.
- **Feedback Entry**: Human or peer review input linked to a candidate iteration,
  including feedback type, summary, and disposition.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In at least 90% of sessions, the system completes three or more full
  iteration cycles (evaluate, reflect, revise) without manual recovery steps.
- **SC-002**: In 100% of user-interrupted sessions, the system returns a best-known
  reward candidate with complete supporting evidence within 60 seconds.
- **SC-003**: In controlled validation scenarios containing known exploit-prone
  rewards, the system flags generalization risk before final recommendation in at
  least 95% of cases.
- **SC-004**: In evaluator review, at least 80% of final recommended rewards are
  rated as more aligned and generalizable than the session baseline candidate.

## Assumptions

- Primary users are researchers who can provide environment objectives and assess
  experiment outcomes.
- Target environments can produce demonstration artifacts suitable for human review.
- Human feedback is asynchronous and may arrive after experiment completion.
- Independent peer feedback is available from a separate, stronger reviewer context.
- A session may terminate when context or budget limits are reached, and preserving
  the best-known candidate at that point is acceptable for this project stage.
- The first release supports one active optimization session per environment.
