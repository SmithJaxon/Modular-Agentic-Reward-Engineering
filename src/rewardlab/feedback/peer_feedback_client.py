"""
Summary: Offline-safe peer feedback client for RewardLab recommendation review.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from datetime import UTC, datetime

from rewardlab.llm.openai_client import ChatCompletionRequest, ChatMessage, OpenAIClient
from rewardlab.schemas.feedback_entry import FeedbackEntry, FeedbackSourceType


class PeerFeedbackClient:
    """Generate peer feedback with an injected model client or deterministic fallback."""

    def __init__(self, openai_client: OpenAIClient | None = None) -> None:
        """Store an optional OpenAI client without performing network activity."""

        self.openai_client = openai_client

    def request_feedback(
        self,
        *,
        session_id: str,
        candidate_id: str,
        objective_text: str,
        reward_definition: str,
        aggregate_score: float | None,
    ) -> FeedbackEntry:
        """Create a peer feedback entry using a local fallback unless credentials exist."""

        if self.openai_client is not None and self.openai_client.has_credentials:
            comment = self._request_model_feedback(
                objective_text=objective_text,
                reward_definition=reward_definition,
            )
        else:
            comment = (
                "Peer review: the reward structure appears aligned overall, "
                "but edge-case robustness should still be checked."
            )

        score = 0.85 if (aggregate_score or 0.0) >= 1.0 else 0.4
        return FeedbackEntry(
            feedback_id=_feedback_id(session_id, FeedbackSourceType.PEER),
            candidate_id=candidate_id,
            source_type=FeedbackSourceType.PEER,
            score=round(score, 2),
            comment=comment,
        )

    def _request_model_feedback(self, *, objective_text: str, reward_definition: str) -> str:
        """Request feedback from the configured model client using a minimal prompt."""

        assert self.openai_client is not None
        response = self.openai_client.chat_completion(
            ChatCompletionRequest(
                model="gpt-5-nano",
                messages=(
                    ChatMessage(
                        role="system",
                        content="Provide one short peer-review critique for an RL reward function.",
                    ),
                    ChatMessage(
                        role="user",
                        content=(
                            f"Objective:\n{objective_text}\n\n"
                            f"Reward definition:\n{reward_definition}\n\n"
                            "Respond with one concise critique."
                        ),
                    ),
                ),
                reasoning_effort="minimal",
                max_tokens=120,
            )
        )
        return response.content.strip()


def _feedback_id(session_id: str, source_type: FeedbackSourceType) -> str:
    """Return a timestamp-based feedback identifier."""

    timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S%f")
    return f"{session_id}-{source_type.value}-feedback-{timestamp}"
