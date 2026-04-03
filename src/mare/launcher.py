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
            'SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            'PROJECT_ROOT="${SCRIPT_DIR}"',
            'while [ ! -f "${PROJECT_ROOT}/pyproject.toml" ] && [ "${PROJECT_ROOT}" != "/" ]; do',
            '  PROJECT_ROOT="$(dirname "${PROJECT_ROOT}")"',
            "done",
            'if [ ! -f "${PROJECT_ROOT}/pyproject.toml" ]; then',
            '  echo "Unable to locate project root from ${SCRIPT_DIR}" >&2',
            "  exit 1",
            "fi",
            'cd "${PROJECT_ROOT}"',
            'if [ -f ".env" ]; then',
            '  set -a',
            '  . ".env"',
            '  set +a',
            'fi',
        ]
        for key, value in sorted(contract.launch_target.environment_variables.items()):
            lines.append(f'export {key}="{value}"')
        lines.append(self._render_command(contract))
        content = "\n".join(lines) + "\n"
        return LaunchScript(path=script_path, content=content)

    def _render_command(self, contract: PPORunContract) -> str:
        command = contract.render_command()
        rendered: List[str] = []
        for part in command:
            if part == str(contract.run_dir):
                rendered.append('"${SCRIPT_DIR}"')
            else:
                rendered.append(part)
        return " ".join(rendered)
