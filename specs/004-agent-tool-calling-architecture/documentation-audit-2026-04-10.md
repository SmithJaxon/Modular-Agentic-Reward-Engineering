# Documentation Audit (2026-04-10)

## Scope

Audited operator/developer-facing docs for contradictions between:

- current runnable behavior (`session` pipeline)
- completed `003-real-experiment-readiness` work
- planned `004-agent-tool-calling-architecture` work

## Findings And Resolutions

1. `agent_tools` quickstart read as if already implemented.
   - Resolution: initially marked planned-only to avoid confusion, then updated
     after implementation to "implemented beta" status with active commands.
   - File: `specs/004-agent-tool-calling-architecture/quickstart.md`

2. `README.md` implied a remaining Humanoid PPO dependency blocker.
   - Resolution: changed wording to conditional ("if missing") and removed stale blocker framing.
   - File: `README.md`

3. `README.md` and `003 quickstart` used "agent-style" wording for current mode.
   - Resolution: clarified that current OpenAI mode is model-backed reward revision within the `session` pipeline, not autonomous tool-calling control.
   - Files:
     - `README.md`
     - `specs/003-real-experiment-readiness/quickstart.md`

4. `003` plan still described unresolved approval gate.
   - Resolution: replaced with current completion status and pointer to active `004` follow-on.
   - File: `specs/003-real-experiment-readiness/plan.md`

5. `AGENTS.md` guardrails pointed backlog to completed `003` tasks only.
   - Resolution: updated active backlog target to `004-agent-tool-calling-architecture`.
   - File: `AGENTS.md`

6. `NEXT_AGENT_HANDOFF.md` contained historical blockers likely to mislead.
   - Resolution: added top-level superseded notice with pointers to active `004` docs.
   - File: `NEXT_AGENT_HANDOFF.md`

7. `003` verification report stopped at now-resolved OpenAI blocker.
   - Resolution: added 2026-04-10 status section with resolved state and token-cap clarification.
   - File: `specs/003-real-experiment-readiness/verification-report.md`

8. `001` quickstart had a dangling/truncated backend-routing line.
   - Resolution: removed invalid trailing text.
   - File: `specs/001-iterative-reward-design/quickstart.md`

## Residual Caveats

- `rewardlab experiment ...` is implemented as the active autonomous path.
- `rewardlab session ...` remains present but is legacy pipeline behavior.
