# Research Notes: Real Experiment Readiness

## Decision 1: Keep deterministic execution only as explicit offline validation mode

- Decision: Preserve the current heuristic/deterministic iteration path for
  unit, contract, and low-cost offline validation, but separate it clearly from
  the actual experiment path used for real sessions.
- Rationale: The repository already has fast, stable regression coverage that
  should not be discarded. The real gap is production-style backend execution,
  not offline testability.
- Alternatives considered:
  - Remove deterministic execution entirely: rejected because it would slow down
    iteration and reduce test stability.
  - Keep heuristic execution as the default session path: rejected because it
    does not satisfy actual experiment readiness.

## Decision 2: Introduce a constrained reward-program loading interface

- Decision: Load candidate reward definitions through a narrow runtime contract
  rather than evaluating arbitrary snippets ad hoc in orchestration code.
- Rationale: Real backend execution needs a predictable callable shape, clearer
  validation errors, and better separation between candidate source and runtime.
- Alternatives considered:
  - Inline `exec` logic inside the session service: rejected because it would
    blur module boundaries and make failures harder to manage.
  - Hard-code reward functions outside candidate definitions: rejected because it
    would break the core reward-iteration model.

## Decision 3: Persist run evidence by run id under `.rewardlab/`

- Decision: Store metrics, manifests, and review artifacts under a run-scoped
  directory tree inside `.rewardlab/`.
- Rationale: This keeps evidence local to the worktree, resumable, and easy to
  reference from reports, feedback, and checkpointing.
- Alternatives considered:
  - Store only aggregate metrics in SQLite: rejected because actual experiment
    readiness needs reviewable artifacts and file-based evidence.
  - Write artifacts outside the project tree: rejected by the workspace rules.

## Decision 4: Make Gymnasium the first real backend milestone

- Decision: Deliver real Gymnasium execution before real Isaac execution.
- Rationale: Gymnasium is the lighter path to proving the architecture and
  validating the session lifecycle with actual backend evidence.
- Alternatives considered:
  - Build both backends simultaneously from the start: rejected because it
    increases integration risk on the critical path.

## Decision 5: Treat Isaac runtime readiness as explicit, not implicit

- Decision: When Isaac runtime prerequisites are not available, surface an
  actionable runtime-status error and stop there instead of silently falling
  back to fake execution.
- Rationale: The user wants actual experiments on both backends, so silent
  degradation would hide the real blocker.
- Alternatives considered:
  - Continue using injected fake environments for Isaac in normal runs:
    rejected because that would misrepresent readiness.

## Decision 6: Generate metrics-first artifact bundles with optional render outputs

- Decision: Require every real run to produce a manifest and metrics files, with
  frames or video captured when the environment can render in the current setup.
- Rationale: Rendering support can vary by backend and local environment, but
  actual run evidence should still be available even when media capture is
  limited.
- Alternatives considered:
  - Require video output for every backend: rejected because it would make
    readiness depend on rendering even when the experiment itself is valid.

## Decision 7: Use opt-in backend smoke validation after approval

- Decision: Keep heavy real-backend smokes separate from the default offline
  suite and run them only after approved dependency setup.
- Rationale: This preserves fast local iteration while still requiring final
  evidence for Gymnasium and Isaac readiness.
- Alternatives considered:
  - Fold heavy backend runs into every pytest invocation: rejected because it
    would slow iteration and fail before approval-gated installs are completed.
