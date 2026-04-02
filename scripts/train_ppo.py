#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Placeholder PPO training entrypoint")
    parser.add_argument("--environment", required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-steps", type=int, required=True)
    parser.add_argument("--eval-episodes", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "environment": args.environment,
        "task_name": args.task_name,
        "algorithm": args.algorithm,
        "seed": args.seed,
        "train_steps": args.train_steps,
        "eval_episodes": args.eval_episodes,
        "device": args.device,
        "status": "placeholder",
    }
    (args.run_dir / "train_request.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

