"""Task registry check for target IsaacGymEnvs tasks used by RewardLab."""

import isaacgym
import isaacgymenvs
from isaacgymenvs.tasks import isaacgym_task_map

print("imports: ok")
print("task_count:", len(isaacgym_task_map))
for name in ("Cartpole", "Humanoid", "AllegroHand"):
    print(name, "present:", name in isaacgym_task_map)
