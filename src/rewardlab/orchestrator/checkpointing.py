"""
Summary: Checkpoint payload builders for resumable session state persistence.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from typing import Any


def build_checkpoint_payload(
    session: dict[str, Any],
    candidates: list[dict[str, Any]],
    latest_reflection: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Build minimal checkpoint payload for session resume flows.

    Args:
        session: Session metadata.
        candidates: Candidate metadata records.
        latest_reflection: Most recent reflection payload.

    Returns:
        Serializable checkpoint payload.
    """
    return {
        "session": {
            "session_id": session["session_id"],
            "status": session["status"],
            "best_candidate_id": session.get("best_candidate_id"),
            "stop_reason": session.get("stop_reason"),
            "metadata": session.get("metadata", {}),
        },
        "candidates": candidates,
        "latest_reflection": latest_reflection,
    }
