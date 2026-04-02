"""
Summary: Root Typer application for RewardLab CLI commands.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import typer

from rewardlab.cli.session_commands import session_app

app = typer.Typer(help="RewardLab command-line interface.")
app.add_typer(session_app, name="session")


def main() -> None:
    """Run the RewardLab Typer application."""

    app()
