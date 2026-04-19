"""
Summary: Helpers to build execution runners from agent experiment specs.
Created: 2026-04-17
Last Updated: 2026-04-17
"""

from __future__ import annotations

from rewardlab.experiments.execution_service import ExperimentRunner
from rewardlab.experiments.gymnasium_runner import (
    GymnasiumExperimentRunner,
    HumanoidPpoEvaluationConfig,
)
from rewardlab.experiments.isaacgym_runner import (
    IsaacGymExperimentRunner,
    IsaacGymPolicyConfig,
    IsaacGymSubprocessConfig,
)
from rewardlab.schemas.agent_experiment import ExecutionIsaacConfig, ExecutionPpoConfig
from rewardlab.schemas.session_config import EnvironmentBackend


def build_runner(
    *,
    environment_backend: EnvironmentBackend,
    ppo_config: ExecutionPpoConfig | None,
    isaac_config: ExecutionIsaacConfig | None = None,
    total_timesteps_override: int | None = None,
    eval_runs_override: int | None = None,
) -> ExperimentRunner:
    """Return a configured execution runner for the requested backend."""

    if environment_backend == EnvironmentBackend.GYMNASIUM:
        return GymnasiumExperimentRunner(
            humanoid_ppo_config=(
                HumanoidPpoEvaluationConfig(
                    total_timesteps=(
                        total_timesteps_override
                        if total_timesteps_override is not None
                        else ppo_config.total_timesteps
                    ),
                    checkpoint_count=ppo_config.checkpoint_count,
                    evaluation_run_count=(
                        eval_runs_override
                        if eval_runs_override is not None
                        else ppo_config.eval_runs
                    ),
                    evaluation_episodes_per_checkpoint=(
                        ppo_config.eval_episodes_per_checkpoint
                    ),
                    n_envs=ppo_config.n_envs,
                    device=ppo_config.device,
                )
                if ppo_config is not None
                else None
            )
        )
    if environment_backend == EnvironmentBackend.ISAAC_GYM:
        policy_config = (
            IsaacGymPolicyConfig(
                total_timesteps=(
                    total_timesteps_override
                    if total_timesteps_override is not None
                    else ppo_config.total_timesteps
                ),
                checkpoint_count=ppo_config.checkpoint_count,
                evaluation_run_count=(
                    eval_runs_override
                    if eval_runs_override is not None
                    else ppo_config.eval_runs
                ),
                evaluation_episodes_per_checkpoint=ppo_config.eval_episodes_per_checkpoint,
                n_envs=ppo_config.n_envs,
                device=ppo_config.device,
            )
            if ppo_config is not None
            else IsaacGymPolicyConfig()
        )
        worker_command = (
            isaac_config.worker_command
            if isaac_config is not None and isaac_config.worker_command is not None
            and isaac_config.worker_command.strip()
            else None
        )
        subprocess_config = (
            IsaacGymSubprocessConfig(worker_command=worker_command)
            if worker_command is not None
            else None
        )
        return IsaacGymExperimentRunner(
            policy_config=policy_config,
            backend_kwargs={
                "cfg_dir_override": (
                    isaac_config.cfg_dir if isaac_config is not None else None
                )
            },
            subprocess_config=subprocess_config,
        )

    raise ValueError(f"unsupported environment backend: {environment_backend.value!r}")
