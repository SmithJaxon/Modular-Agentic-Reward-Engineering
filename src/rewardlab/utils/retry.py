"""
Summary: Bounded retry and exponential backoff utilities for RewardLab.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from time import sleep
from typing import TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Configuration for retry attempts and backoff timing."""

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    max_delay_seconds: float = 5.0
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """Validate retry configuration ranges."""

        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.base_delay_seconds < 0:
            raise ValueError("base_delay_seconds must be non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must be >= base_delay_seconds")
        if self.backoff_multiplier < 1.0:
            raise ValueError("backoff_multiplier must be >= 1.0")


@dataclass(frozen=True)
class AttemptFailure:
    """Captured information about a failed retry attempt."""

    attempt_number: int
    error: Exception


class RetryError(RuntimeError):
    """Raised when all retry attempts have been exhausted."""

    def __init__(self, message: str, failures: list[AttemptFailure]) -> None:
        """Store failed attempts for later inspection."""

        super().__init__(message)
        self.failures = failures


def compute_backoff_delays(policy: RetryPolicy) -> list[float]:
    """Return the sleep delays used between retry attempts."""

    if policy.max_attempts == 1:
        return []

    delays: list[float] = []
    delay = policy.base_delay_seconds
    for _ in range(policy.max_attempts - 1):
        delays.append(min(delay, policy.max_delay_seconds))
        delay *= policy.backoff_multiplier
    return delays


def retry_call(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    sleeper: Callable[[float], None] = sleep,
) -> T:
    """Run an operation with bounded retries and exponential backoff."""

    active_policy = policy or RetryPolicy()
    failures: list[AttemptFailure] = []
    delays = compute_backoff_delays(active_policy)

    for attempt_number in range(1, active_policy.max_attempts + 1):
        try:
            return operation()
        except retryable_exceptions as exc:
            failures.append(AttemptFailure(attempt_number=attempt_number, error=exc))
            if attempt_number == active_policy.max_attempts:
                break
            sleeper(delays[attempt_number - 1])

    raise RetryError("retry attempts exhausted", failures)


def retry_each(
    operations: Iterable[Callable[[], T]],
    *,
    policy: RetryPolicy | None = None,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    sleeper: Callable[[float], None] = sleep,
) -> list[T]:
    """Apply retry behavior to each operation in an iterable and return results."""

    return [
        retry_call(
            operation,
            policy=policy,
            retryable_exceptions=retryable_exceptions,
            sleeper=sleeper,
        )
        for operation in operations
    ]
