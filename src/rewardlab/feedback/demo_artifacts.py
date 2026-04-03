"""
Summary: Demonstration artifact tracking helpers for human feedback evidence links.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from typing import Any


class DemoArtifactTracker:
    """
    Normalize and track demonstration artifact references in session metadata.
    """

    def register_artifact(
        self,
        session: dict[str, Any],
        candidate_id: str,
        artifact_ref: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Resolve and persist one demonstration artifact reference.

        Args:
            session: Session metadata dictionary.
            candidate_id: Candidate identifier.
            artifact_ref: Optional caller-provided artifact reference.

        Returns:
            Tuple of resolved artifact reference and updated metadata payload.
        """
        metadata = dict(session.get("metadata", {}))
        artifacts = {
            key: list(value)
            for key, value in dict(metadata.get("demo_artifacts", {})).items()
        }
        resolved = (
            artifact_ref.strip()
            if artifact_ref is not None and artifact_ref.strip()
            else f"demo://{session['session_id']}/{candidate_id}/latest"
        )
        candidate_refs = list(artifacts.get(candidate_id, []))
        if resolved not in candidate_refs:
            candidate_refs.append(resolved)
        artifacts[candidate_id] = candidate_refs
        metadata["demo_artifacts"] = artifacts
        return resolved, metadata

    def artifacts_for_candidate(
        self,
        session: dict[str, Any],
        candidate_id: str,
    ) -> tuple[str, ...]:
        """
        Return tracked demonstration artifacts for a candidate.

        Args:
            session: Session metadata dictionary.
            candidate_id: Candidate identifier.

        Returns:
            Tuple of artifact references.
        """
        metadata = dict(session.get("metadata", {}))
        artifacts = dict(metadata.get("demo_artifacts", {}))
        return tuple(str(value) for value in artifacts.get(candidate_id, []))
