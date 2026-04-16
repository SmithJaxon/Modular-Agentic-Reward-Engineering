"""
Summary: Rich decision context state for the primary optimizer agent.
Created: 2026-04-10
Last Updated: 2026-04-10
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rewardlab.schemas.agentic_run import AgentDecision
from rewardlab.schemas.tool_contracts import ToolResult, ToolResultStatus


@dataclass(slots=True)
class ContextStore:
    """
    Hold run-level decision history and compact tool-result summaries.
    """

    decisions: list[AgentDecision] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    best_score: float | None = None

    def record_decision(self, decision: AgentDecision) -> None:
        """
        Append one primary decision to the context history.
        """
        self.decisions.append(decision)

    def record_tool_result(self, result: ToolResult) -> None:
        """
        Append one tool result and update best-score summary when available.
        """
        self.tool_results.append(result)
        score_raw = result.output.get("score")
        if isinstance(score_raw, int | float):
            score = float(score_raw)
            self.best_score = score if self.best_score is None else max(self.best_score, score)

    @property
    def completed_tool_calls(self) -> int:
        """
        Count successful tool calls currently recorded in context.
        """
        return sum(1 for row in self.tool_results if row.status is ToolResultStatus.COMPLETED)

    @property
    def latest_tool_result(self) -> ToolResult | None:
        """
        Return the most recent tool result when available.
        """
        return self.tool_results[-1] if self.tool_results else None

    def experiment_scores(self) -> list[float]:
        """
        Return numeric scores from completed `run_experiment` tool results.
        """
        scores: list[float] = []
        for result in self.tool_results:
            if result.tool_name != "run_experiment":
                continue
            if result.status is not ToolResultStatus.COMPLETED:
                continue
            raw = result.output.get("score")
            if isinstance(raw, int | float):
                scores.append(float(raw))
        return scores

    def experiment_results(self) -> list[ToolResult]:
        """
        Return completed run-experiment tool results in execution order.
        """
        return [
            result
            for result in self.tool_results
            if result.tool_name == "run_experiment" and result.status is ToolResultStatus.COMPLETED
        ]

    def latest_experiment_result(self) -> ToolResult | None:
        """
        Return the most recent completed run-experiment result.
        """
        experiments = self.experiment_results()
        return experiments[-1] if experiments else None

    def probe_results(self) -> list[ToolResult]:
        """
        Return completed run-probe-suite tool results in execution order.
        """
        return [
            result
            for result in self.tool_results
            if result.tool_name == "run_probe_suite"
            and result.status is ToolResultStatus.COMPLETED
        ]

    def latest_probe_result(self) -> ToolResult | None:
        """
        Return the most recent completed run-probe-suite result.
        """
        probes = self.probe_results()
        return probes[-1] if probes else None

    def probed_candidate_ids(self) -> set[str]:
        """
        Return candidate IDs that already have completed probe-suite results.
        """
        identifiers: set[str] = set()
        for result in self.tool_results:
            if result.tool_name != "run_probe_suite":
                continue
            if result.status is not ToolResultStatus.COMPLETED:
                continue
            raw_id = result.output.get("candidate_id")
            if isinstance(raw_id, str) and raw_id:
                identifiers.add(raw_id)
        return identifiers

    def compare_completed_count(self) -> int:
        """
        Count completed compare-candidates tool calls.
        """
        return sum(
            1
            for result in self.tool_results
            if result.tool_name == "compare_candidates"
            and result.status is ToolResultStatus.COMPLETED
        )

    def export_completed_count(self) -> int:
        """
        Count completed export-report tool calls.
        """
        return sum(
            1
            for result in self.tool_results
            if result.tool_name == "export_report" and result.status is ToolResultStatus.COMPLETED
        )

    def candidate_snapshots(self) -> list[dict[str, object]]:
        """
        Build candidate snapshots from experiment and probe tool outputs.
        """
        probes_by_candidate: dict[str, dict[str, object]] = {}
        for result in self.tool_results:
            if result.tool_name != "run_probe_suite":
                continue
            if result.status is not ToolResultStatus.COMPLETED:
                continue
            candidate_id = result.output.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id:
                probes_by_candidate[candidate_id] = {
                    "risk_level": result.output.get("risk_level"),
                    "robustness_bonus": result.output.get("robustness_bonus"),
                }

        snapshots: list[dict[str, object]] = []
        for result in self.experiment_results():
            candidate_id_raw = result.output.get("candidate_id")
            if not isinstance(candidate_id_raw, str) or not candidate_id_raw:
                continue
            score_raw = result.output.get("score")
            if not isinstance(score_raw, int | float):
                continue
            probe = probes_by_candidate.get(candidate_id_raw, {})
            robustness_bonus = probe.get("robustness_bonus", 0.0)
            if not isinstance(robustness_bonus, int | float):
                robustness_bonus = 0.0
            snapshots.append(
                {
                    "candidate_id": candidate_id_raw,
                    "score": float(score_raw),
                    "robustness_bonus": float(robustness_bonus),
                    "risk_level": probe.get("risk_level", "unknown"),
                }
            )
        return snapshots
