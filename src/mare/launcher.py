from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from .launch import PPORunContract


@dataclass(frozen=True)
class LaunchScript:
    """Generated shell script for a PPO contract."""

    path: Path
    content: str

    def write(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(self.content, encoding="utf-8")
        return self.path


class LaunchScriptBuilder:
    """Render a portable shell wrapper around a PPO contract."""

    def build(self, contract: PPORunContract, script_path: Path) -> LaunchScript:
        lines: List[str] = [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f'cd "{contract.launch_target.working_directory}"',
            'if [ -f ".env" ]; then',
            '  set -a',
            '  . ".env"',
            '  set +a',
            'fi',
        ]
        for key, value in sorted(contract.launch_target.environment_variables.items()):
            lines.append(f'export {key}="{value}"')
        lines.append(" ".join(contract.render_command()))
        content = "\n".join(lines) + "\n"
        return LaunchScript(path=script_path, content=content)
