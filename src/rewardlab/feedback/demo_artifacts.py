"""
Summary: Lightweight local artifact tracking for human and peer feedback reviews.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rewardlab.schemas.feedback_entry import FeedbackEntry


@dataclass(frozen=True)
class DemoArtifactBundle:
    """Paths to the files generated for a reviewable feedback artifact bundle."""

    root: Path
    manifest_path: Path
    feedback_path: Path
    review_markdown_path: Path


class DemoArtifactTracker:
    """Create small local artifacts suitable for later human review."""

    def __init__(self, root_dir: Path) -> None:
        """Store the root directory used for review artifacts."""

        self.root_dir = root_dir

    def write_feedback_bundle(
        self,
        feedback: FeedbackEntry,
        *,
        title: str | None = None,
        notes: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DemoArtifactBundle:
        """Create a compact, reviewable artifact bundle on disk."""

        bundle_root = self.root_dir / feedback.feedback_id
        bundle_root.mkdir(parents=True, exist_ok=True)

        manifest_path = bundle_root / "manifest.json"
        feedback_path = bundle_root / "feedback.json"
        review_markdown_path = bundle_root / "review.md"

        manifest = {
            "feedback_id": feedback.feedback_id,
            "candidate_id": feedback.candidate_id,
            "source_type": feedback.source_type.value,
            "artifact_ref": feedback.artifact_ref,
            "created_at": feedback.created_at.isoformat(),
            "files": {
                "feedback": feedback_path.name,
                "review": review_markdown_path.name,
            },
        }
        if metadata:
            manifest["metadata"] = metadata

        feedback_payload = feedback.model_dump(mode="json")
        review_lines = [
            f"# {title or 'Feedback Review'}",
            "",
            f"- Feedback ID: {feedback.feedback_id}",
            f"- Candidate ID: {feedback.candidate_id}",
            f"- Source: {feedback.source_type.value}",
        ]
        if feedback.artifact_ref:
            review_lines.append(f"- Artifact: {feedback.artifact_ref}")
        review_lines.extend(
            [
                "",
                "## Comment",
                "",
                feedback.comment,
            ]
        )
        if notes:
            review_lines.extend(["", "## Notes", ""])
            review_lines.extend(f"- {note}" for note in notes)

        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        feedback_path.write_text(json.dumps(feedback_payload, indent=2), encoding="utf-8")
        review_markdown_path.write_text("\n".join(review_lines).rstrip() + "\n", encoding="utf-8")

        return DemoArtifactBundle(
            root=bundle_root,
            manifest_path=manifest_path,
            feedback_path=feedback_path,
            review_markdown_path=review_markdown_path,
        )
