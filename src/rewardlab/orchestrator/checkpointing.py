"""
Summary: JSON checkpoint writer for resumable RewardLab session state.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import json
from pathlib import Path

from rewardlab.schemas.reflection_record import ReflectionRecord
from rewardlab.schemas.reward_candidate import RewardCandidate
from rewardlab.schemas.session_config import SessionRecord


class CheckpointManager:
    """Persist lightweight JSON snapshots for session recovery and inspection."""

    def __init__(self, checkpoint_dir: Path) -> None:
        """Store the checkpoint directory and ensure it exists on demand."""

        self.checkpoint_dir = checkpoint_dir

    def write_checkpoint(
        self,
        *,
        session: SessionRecord,
        candidates: list[RewardCandidate],
        reflections: list[ReflectionRecord],
    ) -> Path:
        """Write a full JSON checkpoint for the current session state."""

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = self.checkpoint_dir / f"{session.session_id}.json"
        payload = {
            "session": session.model_dump(mode="json"),
            "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
            "reflections": [reflection.model_dump(mode="json") for reflection in reflections],
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return checkpoint_path
