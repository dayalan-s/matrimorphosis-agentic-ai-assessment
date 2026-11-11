#!/usr/bin/env python3
"""Static half of the async check (task 2): scans src/app for blocking calls inside the four tools that must be async.
Writes results/static.json: {"async_static": true|false, "problems": [...]}. Plain python3, no dependencies."""
import ast
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
TOOLS = {"get_weather": "src/app/tools_s1.py", "web_fetch": "src/app/tools_s1.py", "sql_query": "src/app/tools_s1.py",
         "search_knowledge_base": "src/app/kb/tools.py"}
BLOCKING = {"time.sleep", "httpx.get", "httpx.post", "httpx.request", "httpx.Client", "httpx.stream", "requests.get", "requests.post",
            "requests.request", "requests.Session", "sqlite3.connect", "trafilatura.fetch_url", "urllib.request.urlopen"}
# calls handed to asyncio.to_thread(...) are fine: they are references, not calls, so they never show up here


def dotted(node: ast.AST) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def check() -> list[str]:
    problems = []
    for name, rel in TOOLS.items():
        tree = ast.parse((ROOT / rel).read_text())
        fn = next((n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
        if fn is None:
            problems.append(f"{name}: not found in {rel}")
        elif not isinstance(fn, ast.AsyncFunctionDef):
            problems.append(f"{name}: still a sync function (use async def)")
        else:
            for call in (c for c in ast.walk(fn) if isinstance(c, ast.Call)):
                if dotted(call.func) in BLOCKING:
                    problems.append(f"{name}: blocking call {dotted(call.func)}() inside an async tool (line {call.lineno})")
    return problems


if __name__ == "__main__":
    problems = check()
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    (out / "static.json").write_text(json.dumps({"async_static": not problems, "problems": problems}))
    print("async check:", "ok" if not problems else "\n  " + "\n  ".join(problems))
    sys.exit(0)
