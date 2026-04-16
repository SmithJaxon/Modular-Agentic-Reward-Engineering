"""
Summary: Root Typer application entrypoint for rewardlab command groups.
Created: 2026-04-02
Last Updated: 2026-04-02
"""

from __future__ import annotations

import typer

from rewardlab.cli.agent_commands import agent_app
from rewardlab.cli.feedback_commands import feedback_app
from rewardlab.cli.session_commands import session_app

app = typer.Typer(no_args_is_help=True)
app.add_typer(session_app, name="session")
app.add_typer(feedback_app, name="feedback")
app.add_typer(agent_app, name="agent")


def main() -> None:
    """
    Execute Typer CLI app entrypoint.
    """
    app()


if __name__ == "__main__":
    main()
