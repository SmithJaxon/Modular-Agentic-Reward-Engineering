"""
Summary: Bounded retry and backoff helpers for resilient operation execution.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """
    Configure retry behavior for transient failures.
    """

    max_attempts: int = 3
    base_backoff_seconds: float = 0.1
    backoff_multiplier: float = 2.0

    def __post_init__(self) -> None:
        """
        Validate retry policy bounds at construction.

        Raises:
            ValueError: When any policy parameter is invalid.
        """
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if self.base_backoff_seconds < 0:
            raise ValueError("base_backoff_seconds must be >= 0")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be >= 1")


class RetryError(RuntimeError):
    """
    Represent exhaustion of retry attempts for an operation.
    """


def run_with_retry(
    operation: Callable[[], T],
    policy: RetryPolicy | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> T:
    """
    Execute an operation with bounded retries and exponential backoff.

    Args:
        operation: Callable operation that may raise exceptions.
        policy: Retry behavior configuration; defaults to `RetryPolicy()`.
        should_retry: Optional predicate to classify retryable exceptions.
        sleep_fn: Sleep function injection for deterministic tests.

    Returns:
        Operation result value.

    Raises:
        RetryError: If retries are exhausted.
        Exception: Immediate exception when should_retry returns False.
    """
    cfg = policy or RetryPolicy()
    predicate = should_retry or (lambda _exc: True)
    attempt = 1
    backoff = cfg.base_backoff_seconds
    last_error: Exception | None = None

    while attempt <= cfg.max_attempts:
        try:
            return operation()
        except Exception as exc:  # noqa: BLE001 - caller controls predicate and error boundary.
            last_error = exc
            if attempt >= cfg.max_attempts or not predicate(exc):
                break
            sleep_fn(backoff)
            backoff *= cfg.backoff_multiplier
            attempt += 1

    assert last_error is not None
    raise RetryError(f"operation failed after {cfg.max_attempts} attempts") from last_error
