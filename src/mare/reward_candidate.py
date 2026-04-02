from __future__ import annotations

from dataclasses import dataclass, field
import ast
import builtins as py_builtins
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional


SAFE_IMPORTS = {"math"}
SAFE_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "sum": sum,
    "zip": zip,
}
DENIED_CALLS = {
    "compile",
    "dir",
    "eval",
    "exec",
    "globals",
    "help",
    "input",
    "locals",
    "open",
    "setattr",
    "getattr",
    "vars",
    "__import__",
}
DENIED_ATTR_PREFIXES = ("__",)
REQUIRED_ENTRYPOINT = "compute_reward"


@dataclass(frozen=True)
class RewardCandidateSpec:
    """Metadata for a reward-function source module."""

    name: str
    path: Path
    entrypoint: str = REQUIRED_ENTRYPOINT
    allowed_imports: tuple[str, ...] = ("math",)


@dataclass(frozen=True)
class RewardValidationIssue:
    """Single validation finding for a reward candidate."""

    severity: str
    message: str


@dataclass(frozen=True)
class RewardValidationResult:
    """Validation output for a reward candidate source file."""

    spec: RewardCandidateSpec
    valid: bool
    issues: List[RewardValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spec": {
                "name": self.spec.name,
                "path": str(self.spec.path),
                "entrypoint": self.spec.entrypoint,
                "allowed_imports": list(self.spec.allowed_imports),
            },
            "valid": self.valid,
            "issues": [
                {"severity": issue.severity, "message": issue.message}
                for issue in self.issues
            ],
        }


@dataclass(frozen=True)
class LoadedRewardCandidate:
    """Validated and executable reward candidate."""

    spec: RewardCandidateSpec
    source: str
    entrypoint: Callable[[Any, Any, Dict[str, Any]], float]

    def reward(self, observation: Any, action: Any, info: Dict[str, Any]) -> float:
        return float(self.entrypoint(observation, action, info))


class RewardCandidateValidator:
    """Static checks for Python reward modules."""

    def validate_source(self, source: str, spec: RewardCandidateSpec) -> RewardValidationResult:
        issues: List[RewardValidationIssue] = []
        try:
            tree = ast.parse(source, filename=str(spec.path))
        except SyntaxError as exc:
            return RewardValidationResult(
                spec=spec,
                valid=False,
                issues=[
                    RewardValidationIssue(
                        severity="error",
                        message="SyntaxError: {0}".format(exc.msg),
                    )
                ],
            )

        self._validate_ast(tree, spec, issues)
        has_entrypoint = self._has_required_entrypoint(tree, spec.entrypoint)
        if not has_entrypoint:
            issues.append(
                RewardValidationIssue(
                    severity="error",
                    message="Missing required function: {0}".format(spec.entrypoint),
                )
            )

        valid = not any(issue.severity == "error" for issue in issues)
        return RewardValidationResult(spec=spec, valid=valid, issues=issues)

    def validate_file(self, path: Path, entrypoint: str = REQUIRED_ENTRYPOINT) -> RewardValidationResult:
        spec = RewardCandidateSpec(name=path.stem, path=path, entrypoint=entrypoint)
        return self.validate_source(path.read_text(encoding="utf-8"), spec)

    def _validate_ast(
        self,
        tree: ast.AST,
        spec: RewardCandidateSpec,
        issues: List[RewardValidationIssue],
    ) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in spec.allowed_imports:
                        issues.append(
                            RewardValidationIssue(
                                severity="error",
                                message="Import not allowed: {0}".format(alias.name),
                            )
                        )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module not in spec.allowed_imports:
                    issues.append(
                        RewardValidationIssue(
                            severity="error",
                            message="Import not allowed: {0}".format(module or "<relative>"),
                        )
                    )
            elif isinstance(node, ast.Call):
                call_name = self._call_name(node.func)
                if call_name in DENIED_CALLS:
                    issues.append(
                        RewardValidationIssue(
                            severity="error",
                            message="Call not allowed: {0}".format(call_name),
                        )
                    )
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith(DENIED_ATTR_PREFIXES):
                    issues.append(
                        RewardValidationIssue(
                            severity="error",
                            message="Attribute not allowed: {0}".format(node.attr),
                        )
                    )
            elif isinstance(node, (ast.ClassDef, ast.With, ast.Try, ast.Global, ast.Nonlocal)):
                issues.append(
                    RewardValidationIssue(
                        severity="error",
                        message="Statement not allowed: {0}".format(type(node).__name__),
                    )
                )

    def _call_name(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def _has_required_entrypoint(self, tree: ast.AST, entrypoint: str) -> bool:
        for node in tree.body if isinstance(tree, ast.Module) else []:
            if isinstance(node, ast.FunctionDef) and node.name == entrypoint:
                args = node.args
                if args.vararg or args.kwarg or args.kwonlyargs:
                    return False
                return len(args.args) == 3
        return False


class RewardCandidateLoader:
    """Validate and load a Python reward module under constrained globals."""

    def __init__(self, validator: Optional[RewardCandidateValidator] = None) -> None:
        self.validator = validator or RewardCandidateValidator()

    def load(self, path: Path, entrypoint: str = REQUIRED_ENTRYPOINT) -> LoadedRewardCandidate:
        source = path.read_text(encoding="utf-8")
        spec = RewardCandidateSpec(name=path.stem, path=path, entrypoint=entrypoint)
        result = self.validator.validate_source(source, spec)
        if not result.valid:
            messages = "; ".join(issue.message for issue in result.issues if issue.severity == "error")
            raise ValueError("Invalid reward candidate: {0}".format(messages))

        module = self._exec_module(source, path)
        fn = getattr(module, entrypoint, None)
        if fn is None or not callable(fn):
            raise ValueError("Reward candidate missing callable entrypoint: {0}".format(entrypoint))
        return LoadedRewardCandidate(spec=spec, source=source, entrypoint=fn)

    def _exec_module(self, source: str, path: Path) -> ModuleType:
        namespace: Dict[str, Any] = {
            "__builtins__": {**SAFE_BUILTINS, "__import__": self._restricted_import},
            "__name__": path.stem,
        }
        exec(compile(source, str(path), "exec"), namespace, namespace)
        module = ModuleType(path.stem)
        for key, value in namespace.items():
            setattr(module, key, value)
        return module

    def _restricted_import(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        module_name = name.split(".", 1)[0]
        if module_name not in SAFE_IMPORTS:
            raise ImportError("Import not allowed: {0}".format(name))
        return py_builtins.__import__(name, globals, locals, fromlist, level)
