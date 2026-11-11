#!/usr/bin/env python3
"""Integrity gate: your code must not detect or special-case the grader. Scans the files you may edit (the `# free:` header of
.checksums) and writes results/integrity.json. A hit sets the score to 0 until you remove it (fix, push, submit the new run).
Only real code is scanned (string constants equal to a flagged name, identifiers, imports): comments and docstrings are fine.
Plain python3, no dependencies."""
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FLAGGED_STRINGS = {"LLM_MODE", "EMBEDDER", "ASSESSMENT_DEBUG", "GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_RUN_ID", "GITHUB_REPOSITORY", "CI", "RUNNER_OS",
                   "BASE_URL", "ASSESSMENT_ID"}
FLAGGED_NAMES = {"fake_llm", "ScriptedChatModel", "DEBUG", "DEBUG_READER", "LOG_BUFFER", "metrics_snapshot", "debug_routes"}
ALLOWED = {("src/app/main.py", "debug_routes")}  # main.py installs the provided routes


def free_paths() -> list[str]:
    for line in (ROOT / ".checksums").read_text().splitlines():
        if line.startswith("# free:"):
            return line.split(":", 1)[1].split()
    return ["src/app/"]


def files() -> list[Path]:
    out = []
    for f in free_paths():
        p = ROOT / f
        out += sorted(p.rglob("*.py")) if p.is_dir() else ([p] if p.suffix == ".py" and p.exists() else [])
    return out


def scan(path: Path) -> list[str]:
    rel = str(path.relative_to(ROOT))
    problems = []
    for node in ast.walk(ast.parse(path.read_text())):
        hit = None
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in FLAGGED_STRINGS:
            hit = node.value
        elif isinstance(node, ast.Name) and node.id in FLAGGED_NAMES:
            hit = node.id
        elif isinstance(node, ast.Attribute) and node.attr in FLAGGED_NAMES:
            hit = node.attr
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] + [a.name.split(".")[-1] for a in node.names] + ([node.module or ""] if isinstance(node, ast.ImportFrom) else [])
            hit = next((n.split(".")[-1] for n in names if n.split(".")[-1] in FLAGGED_NAMES), None)
        if hit and (rel, hit) not in ALLOWED:
            problems.append(f"{rel}:{node.lineno}: '{hit}' (your code must not detect or special-case the grader)")
    return problems


if __name__ == "__main__":
    problems = [p for f in files() for p in scan(f)]
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "integrity.json").write_text(json.dumps({"ok": not problems, "problems": problems}))
    print("integrity check:", "ok" if not problems else "\n  " + "\n  ".join(problems))
    sys.exit(0)
