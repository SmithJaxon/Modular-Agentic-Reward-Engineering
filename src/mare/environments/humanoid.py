from __future__ import annotations

from ..environment_registry import get_environment_preset
from .base import BaseIsaacGymAdapter


class HumanoidAdapter(BaseIsaacGymAdapter):
    def __init__(self) -> None:
        super().__init__(profile=get_environment_preset("Humanoid").profile)

