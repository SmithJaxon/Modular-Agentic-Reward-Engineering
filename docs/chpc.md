# RewardLab on Utah CHPC

This guide sets up split-runtime execution:

- Controller/orchestrator in modern Python (for example 3.12).
- Isaac worker in legacy Python (3.8 with Isaac Gym Preview 4 stack).

## 1. Preflight on login node

```bash
module purge
module load python/3.12
python -m venv .venv-controller
source .venv-controller/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Run Isaac runtime preflight (this should run in the same environment used by jobs):

```bash
rewardlab experiment runtime-check --backend isaacgym --json
rewardlab experiment runtime-check --backend isaacgym --file tools/fixtures/experiments/agent_isaacgym_chpc_profile.yaml --json
```

## 2. Configure split runtime

Use one of:

- `execution.isaac.worker_command` in spec YAML (recommended for reproducibility).
- `REWARDLAB_ISAAC_WORKER_COMMAND` env var.
- `REWARDLAB_ISAAC_WORKER_PYTHON` env var.

Example worker command:

```bash
python3.8 -m rewardlab.experiments.isaacgym_worker
```

For non-standard IsaacGymEnvs locations, set either:

- `execution.isaac.cfg_dir` in spec YAML, or
- `REWARDLAB_ISAAC_CFG_DIR` env var.

## 3. Slurm batch template

Use [`tools/slurm/chpc_rewardlab_isaac.sbatch`](/C:/Users/smith/LocalClasses/AdvAi/Modular-Agentic-Reward-Engineering/tools/slurm/chpc_rewardlab_isaac.sbatch).

The script:

1. Runs `runtime-check` before launching full experiment.
2. Fails fast if runtime prerequisites are missing.
3. Executes one experiment spec with bounded worker timeout.

## 4. CHPC profile spec

Start from:

- [`agent_isaacgym_chpc_profile.yaml`](/C:/Users/smith/LocalClasses/AdvAi/Modular-Agentic-Reward-Engineering/tools/fixtures/experiments/agent_isaacgym_chpc_profile.yaml)

Adjust:

- `environment.id` (`Cartpole`, `Humanoid`, `AllegroHand`).
- `execution.isaac.worker_command`.
- `execution.isaac.cfg_dir` (if needed).
- budgets/timesteps for your allocation.

## 5. Container option

If CHPC policy supports Apptainer/Singularity, keep Isaac stack isolated:

1. Build image from your Isaac-compatible base.
2. Bind project path into container.
3. Set `execution.isaac.worker_command` to call worker inside container.

Example command shape:

```bash
apptainer exec --nv /path/to/isaac.sif python3.8 -m rewardlab.experiments.isaacgym_worker
```
