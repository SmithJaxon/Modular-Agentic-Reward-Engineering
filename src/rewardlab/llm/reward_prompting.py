"""
Summary: Prompt builders for LLM-guided reward synthesis and mutation.
Created: 2026-04-06
Last Updated: 2026-04-06
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any

from rewardlab.experiments.gymnasium_runtime import normalize_reward_program_source

_PROMPT_PREAMBLE = """
You are a reward engineer trying to write reward functions for reinforcement learning tasks.

Return only Python code, preferably inside a ```python fenced block.
The code must define exactly one function:

def compute_reward(observation, action, next_observation, env_reward, terminated, truncated, info):
    ...
    return total_reward, {"component_name": component_value}

Rules:
- Use only Python builtins, math, and numpy.
- Treat observation, action, and next_observation as numpy arrays.
- Return a float reward and a dictionary of named scalar reward components.
- Keep the code self-contained and executable without extra files.
- Prefer interpretable components with explicit names.
- Use environment reward as a fallback sanity signal when needed.
""".strip()


def build_gymnasium_environment_context(environment_id: str) -> str:
    """
    Extract a compact environment summary for reward synthesis prompts.
    """
    try:
        import gymnasium as gym
    except ImportError:
        return f"Environment ID: {environment_id}"

    env = gym.make(environment_id)
    try:
        observation_space = env.observation_space
        action_space = env.action_space
        class_name = env.unwrapped.__class__.__name__
        doc = inspect.getdoc(env.unwrapped.__class__) or ""
        doc = textwrap.shorten(doc, width=3000, placeholder="\n... [truncated]")
        obs_source = _safe_source(getattr(env.unwrapped, "_get_obs", None), max_chars=2500)
        step_source = _safe_source(getattr(env.unwrapped, "step", None), max_chars=1200)
        return "\n".join(
            [
                f"Environment ID: {environment_id}",
                f"Environment class: {class_name}",
                f"Observation space: {observation_space}",
                f"Action space: {action_space}",
                "Environment documentation:",
                doc or "No class docstring available.",
                "Observation implementation snippet:",
                obs_source or "No _get_obs source available.",
                "Step implementation snippet:",
                step_source or "No step source available.",
            ]
        )
    finally:
        env.close()  # type: ignore[no-untyped-call]


def build_initial_reward_prompt(
    *,
    objective_text: str,
    environment_id: str,
    environment_context: str,
    baseline_hint: str,
) -> str:
    """
    Build the initial synthesis prompt for a first executable reward program.
    """
    normalized_hint = baseline_hint.strip() or "No baseline reward was provided."
    return "\n\n".join(
        [
            _PROMPT_PREAMBLE,
            f"Task objective:\n{objective_text.strip()}",
            f"Gymnasium environment:\n{environment_id}",
            f"Environment context:\n{environment_context.strip()}",
            f"Baseline hint:\n{normalized_hint}",
            "Write the best initial reward program you can for this environment.",
        ]
    )


def build_revision_reward_prompt(
    *,
    objective_text: str,
    environment_id: str,
    environment_context: str,
    previous_reward_definition: str,
    reflection_summary: str,
    feedback_summary: str,
) -> str:
    """
    Build a mutation prompt using the best known reward and PPO reflection text.
    """
    normalized_feedback = feedback_summary.strip() or "No additional human or peer feedback."
    normalized_reward = normalize_reward_program_source(previous_reward_definition)
    return "\n\n".join(
        [
            _PROMPT_PREAMBLE,
            f"Task objective:\n{objective_text.strip()}",
            f"Gymnasium environment:\n{environment_id}",
            f"Environment context:\n{environment_context.strip()}",
            "Current best reward program:",
            f"```python\n{normalized_reward}\n```",
            "PPO reward reflection:",
            reflection_summary.strip(),
            f"Additional reviewer feedback:\n{normalized_feedback}",
            (
                "Revise the reward to improve environment return while keeping the reward "
                "components well-scaled and interpretable."
            ),
        ]
    )


def extract_reward_program_source(response_text: str) -> str:
    """
    Normalize raw LLM output into executable reward-program source text.
    """
    return normalize_reward_program_source(response_text)


def _safe_source(candidate: Any, *, max_chars: int) -> str:
    """
    Return a trimmed source snippet for one callable when available.
    """
    if candidate is None:
        return ""
    try:
        source = inspect.getsource(candidate)
    except (OSError, TypeError):
        return ""
    return textwrap.shorten(source, width=max_chars, placeholder="\n... [truncated]")
