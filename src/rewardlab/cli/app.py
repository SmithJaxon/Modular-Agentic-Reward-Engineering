"""
Summary: Root Typer application for RewardLab CLI commands.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import typer

from rewardlab.cli.experiment_commands import experiment_app
from rewardlab.cli.feedback_commands import feedback_app
from rewardlab.cli.session_commands import session_app

app = typer.Typer(help="RewardLab command-line interface.")
app.add_typer(experiment_app, name="experiment")
app.add_typer(feedback_app, name="feedback")
app.add_typer(session_app, name="session")


def main() -> None:
    """Run the RewardLab Typer application."""

    app()
