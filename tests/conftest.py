"""
Summary: Shared pytest configuration for RewardLab tests.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register opt-in flags for real backend smoke tests."""

    parser.addoption(
        "--run-real-gymnasium",
        action="store_true",
        default=False,
        help="run tests marked real_gymnasium against the approved worktree-local .venv",
    )
    parser.addoption(
        "--run-real-isaacgym",
        action="store_true",
        default=False,
        help="run tests marked real_isaacgym against the approved worktree-local .venv",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom backend markers for local test selection help."""

    config.addinivalue_line(
        "markers",
        "real_gymnasium: requires approved Gymnasium dependencies in the worktree .venv",
    )
    config.addinivalue_line(
        "markers",
        "real_isaacgym: requires an approved Isaac Gym runtime in the worktree .venv",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Skip opt-in backend smokes unless the required flag and runtime are present."""

    run_real_gymnasium = bool(config.getoption("--run-real-gymnasium"))
    run_real_isaacgym = bool(config.getoption("--run-real-isaacgym"))
    has_gymnasium = _module_available("gymnasium")
    has_isaacgym = _module_available("isaacgym")

    for item in items:
        if item.get_closest_marker("real_gymnasium") is not None:
            if not run_real_gymnasium:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "need --run-real-gymnasium to run real Gymnasium smokes; "
                            "default pytest runs keep them skipped"
                        )
                    )
                )
                continue
            if not has_gymnasium:
                item.add_marker(
                    pytest.mark.skip(
                        reason="gymnasium is not installed in the approved worktree-local .venv"
                    )
                )

        if item.get_closest_marker("real_isaacgym") is not None:
            if not run_real_isaacgym:
                item.add_marker(
                    pytest.mark.skip(
                        reason=(
                            "need --run-real-isaacgym to run real Isaac Gym smokes; "
                            "default pytest runs keep them skipped"
                        )
                    )
                )
                continue
            if not has_isaacgym:
                item.add_marker(
                    pytest.mark.skip(
                        reason="isaacgym is not installed in the approved worktree-local .venv"
                    )
                )


def _module_available(module_name: str) -> bool:
    """Return whether the requested module can be imported in the active interpreter."""

    return importlib.util.find_spec(module_name) is not None


@pytest.fixture()
def workspace_tmp_path() -> Path:
    """Provide a worktree-local temporary directory for sandbox-safe test files."""

    root = ROOT / ".tmp" / f"pytest-{uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        shutil.rmtree(root, ignore_errors=True)
