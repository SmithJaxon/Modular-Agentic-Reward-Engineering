from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

from .experiment import ExperimentRunner
from .execution import PPORunDispatcher
from .manifest import ExperimentManifest
from .launch import LaunchTarget
from .project import load_project_context
from .registry import list_baseline_presets, validate_registry
from .runtime import load_experiment_spec, spec_from_preset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mare",
        description="Modular Agentic Reward Engineering research prototype",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest",
        help="Create and save a run manifest from a YAML config",
    )
    manifest_parser.add_argument("--config", required=True, type=Path)
    manifest_parser.add_argument("--run-dir", type=Path, default=None)

    dry_run_parser = subparsers.add_parser(
        "dry-run",
        help="Load config, save manifest, and execute the placeholder run",
    )
    dry_run_parser.add_argument("--config", required=True, type=Path)
    dry_run_parser.add_argument("--run-dir", type=Path, default=None)

    preset_parser = subparsers.add_parser(
        "preset",
        help="Create and save a run manifest from a built-in baseline preset",
    )
    preset_parser.add_argument(
        "--name",
        required=True,
        choices=["cartpole", "humanoid", "allegro_hand"],
        help="Preset key from the baseline registry",
    )
    preset_parser.add_argument("--run-dir", type=Path, default=None)

    list_parser = subparsers.add_parser(
        "list-presets",
        help="List built-in baseline presets for the target environments",
    )

    plan_parser = subparsers.add_parser(
        "plan",
        help="Render the PPO run contract for a config or preset",
    )
    plan_source = plan_parser.add_mutually_exclusive_group(required=True)
    plan_source.add_argument("--config", type=Path)
    plan_source.add_argument("--preset", choices=["cartpole", "humanoid", "allegro_hand"])
    plan_parser.add_argument("--run-dir", type=Path, default=None)
    plan_parser.add_argument("--kind", default="gpu_vm")
    plan_parser.add_argument("--python", dest="python_executable", default="python3")

    preview_parser = subparsers.add_parser(
        "preview",
        help="Render the launch receipt for a config or preset without executing it",
    )
    preview_source = preview_parser.add_mutually_exclusive_group(required=True)
    preview_source.add_argument("--config", type=Path)
    preview_source.add_argument("--preset", choices=["cartpole", "humanoid", "allegro_hand"])
    preview_parser.add_argument("--run-dir", type=Path, default=None)
    preview_parser.add_argument("--kind", default="gpu_vm")
    preview_parser.add_argument("--python", dest="python_executable", default="python3")

    script_parser = subparsers.add_parser(
        "script",
        help="Write a launch shell script for a config or preset",
    )
    script_source = script_parser.add_mutually_exclusive_group(required=True)
    script_source.add_argument("--config", type=Path)
    script_source.add_argument("--preset", choices=["cartpole", "humanoid", "allegro_hand"])
    script_parser.add_argument("--run-dir", type=Path, default=None)
    script_parser.add_argument("--kind", default="gpu_vm")
    script_parser.add_argument("--python", dest="python_executable", default="python3")

    return parser


def _create_manifest(spec_path: Path) -> ExperimentManifest:
    spec = load_experiment_spec(spec_path)
    return ExperimentManifest(
        name=spec.name,
        environment=spec.environment,
        baseline=spec.baseline,
        seed=spec.seed,
        notes=spec.notes,
        extra=spec.extra,
    )


def _create_manifest_from_preset(name: str) -> ExperimentManifest:
    spec = spec_from_preset(name)
    return ExperimentManifest(
        name=spec.name,
        environment=spec.environment,
        baseline=spec.baseline,
        seed=spec.seed,
        notes=spec.notes,
        extra=spec.extra,
    )


def _resolve_run_dir(default_root: Path, run_dir: Optional[Path], manifest: ExperimentManifest) -> Path:
    return run_dir if run_dir is not None else default_root / manifest.name


def _run_manifest(path: Path, run_dir: Optional[Path], do_dry_run: bool) -> int:
    context = load_project_context()
    runner = ExperimentRunner(context.paths)
    manifest = _create_manifest(path)
    resolved_run_dir = _resolve_run_dir(context.paths.runs, run_dir, manifest)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runner.save_manifest(resolved_run_dir, manifest)
    print(str(manifest_path))
    if do_dry_run:
        report = runner.placeholder_report(manifest, resolved_run_dir)
        result_path = runner.save_result(resolved_run_dir, report)
        print(str(result_path))
        print(report.status)
        for key, value in report.metrics.items():
            print(f"{key}={value}")
        for warning in report.warnings:
            print(f"warning={warning}")
    return 0


def _run_preset(name: str, run_dir: Optional[Path], do_dry_run: bool) -> int:
    context = load_project_context()
    runner = ExperimentRunner(context.paths)
    manifest = _create_manifest_from_preset(name)
    resolved_run_dir = _resolve_run_dir(context.paths.runs, run_dir, manifest)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runner.save_manifest(resolved_run_dir, manifest)
    print(str(manifest_path))
    if do_dry_run:
        report = runner.placeholder_report(manifest, resolved_run_dir)
        result_path = runner.save_result(resolved_run_dir, report)
        print(str(result_path))
        print(report.status)
        for key, value in report.metrics.items():
            print(f"{key}={value}")
        for warning in report.warnings:
            print(f"warning={warning}")
    return 0


def _list_presets() -> int:
    validate_registry()
    for preset in list_baseline_presets():
        print("{0}: {1} / {2} / seed={3}".format(
            preset.name, preset.environment, preset.baseline, preset.seed
        ))
    return 0


def _load_manifest_from_args(config: Optional[Path], preset: Optional[str]) -> ExperimentManifest:
    if config is not None:
        return _create_manifest(config)
    if preset is not None:
        return _create_manifest_from_preset(preset)
    raise ValueError("Either config or preset must be provided")


def _plan_run(
    config: Optional[Path],
    preset: Optional[str],
    run_dir: Optional[Path],
    kind: str,
    python_executable: str,
) -> int:
    context = load_project_context()
    runner = ExperimentRunner(context.paths)
    manifest = _load_manifest_from_args(config, preset)
    resolved_run_dir = _resolve_run_dir(context.paths.runs, run_dir, manifest)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = runner.save_manifest(resolved_run_dir, manifest)
    launch_target = LaunchTarget(
        kind=kind,
        python_executable=python_executable,
        working_directory=context.config.project_root,
        gpu_required=True,
        environment_variables={},
    )
    contract = runner.build_ppo_run_contract(manifest, resolved_run_dir, launch_target)
    print(str(manifest_path))
    print(json.dumps(contract.to_dict(), indent=2, sort_keys=True))
    print(" ".join(contract.render_command()))
    return 0


def _preview_run(
    config: Optional[Path],
    preset: Optional[str],
    run_dir: Optional[Path],
    kind: str,
    python_executable: str,
) -> int:
    context = load_project_context()
    runner = ExperimentRunner(context.paths)
    manifest = _load_manifest_from_args(config, preset)
    resolved_run_dir = _resolve_run_dir(context.paths.runs, run_dir, manifest)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    runner.save_manifest(resolved_run_dir, manifest)
    launch_target = LaunchTarget(
        kind=kind,
        python_executable=python_executable,
        working_directory=context.config.project_root,
        gpu_required=True,
        environment_variables={},
    )
    contract = runner.build_ppo_run_contract(manifest, resolved_run_dir, launch_target)
    receipt = PPORunDispatcher().preview(contract)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0


def _write_script(
    config: Optional[Path],
    preset: Optional[str],
    run_dir: Optional[Path],
    kind: str,
    python_executable: str,
) -> int:
    context = load_project_context()
    runner = ExperimentRunner(context.paths)
    manifest = _load_manifest_from_args(config, preset)
    resolved_run_dir = _resolve_run_dir(context.paths.runs, run_dir, manifest)
    resolved_run_dir.mkdir(parents=True, exist_ok=True)
    runner.save_manifest(resolved_run_dir, manifest)
    launch_target = LaunchTarget(
        kind=kind,
        python_executable=python_executable,
        working_directory=context.config.project_root,
        gpu_required=True,
        environment_variables={},
    )
    receipt = runner.write_ppo_launch_script(manifest, resolved_run_dir, launch_target)
    print(json.dumps(receipt.to_dict(), indent=2, sort_keys=True))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.command == "manifest":
        return _run_manifest(args.config, args.run_dir, do_dry_run=False)
    if args.command == "dry-run":
        return _run_manifest(args.config, args.run_dir, do_dry_run=True)
    if args.command == "preset":
        return _run_preset(args.name, args.run_dir, do_dry_run=False)
    if args.command == "list-presets":
        return _list_presets()
    if args.command == "plan":
        return _plan_run(args.config, args.preset, args.run_dir, args.kind, args.python_executable)
    if args.command == "preview":
        return _preview_run(args.config, args.preset, args.run_dir, args.kind, args.python_executable)
    if args.command == "script":
        return _write_script(args.config, args.preset, args.run_dir, args.kind, args.python_executable)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
