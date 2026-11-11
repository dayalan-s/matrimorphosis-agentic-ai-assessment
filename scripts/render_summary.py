#!/usr/bin/env python3
"""Student-friendly score report from results/summary.json (+ results/details.json for the reasons).
In GitHub Actions it writes the run's summary page ($GITHUB_STEP_SUMMARY) and a notice annotation; always prints it too."""
import json
import os
import sys
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"
TASKS = {"1": "Define the agent", "2": "Async tools", "3": "Chat memory (SQLite)", "4": "Alembic migration", "5": "User accounts and auth",
         "6": "One knowledge base per user", "7": "Two tools and observability", "8": "Manage the knowledge base with tools",
         "9": "Notes and tasks MCP server"}


def main() -> int:
    path = RESULTS / "summary.json"
    if not path.exists():
        print("No score yet: the tests did not run (see the failed step above).")
        return 0
    s = json.loads(path.read_text())
    details = json.loads((RESULTS / "details.json").read_text()) if (RESULTS / "details.json").exists() else {}
    earned, total = s["earnedPoints"], s["totalPoints"]
    filled = round(20 * earned / total) if total else 0
    integrity = json.loads((RESULTS / "integrity.json").read_text()) if (RESULTS / "integrity.json").exists() else {"ok": True, "problems": []}
    out = []
    if not integrity["ok"]:
        out += ["## Score set to 0: your code detects or special-cases the grader", "", "Remove these, push, and submit the new run (your best score is kept):", ""]
        out += [f"- `{p}`" for p in integrity["problems"]] + [""]
    out += [f"## Score: {earned} / {total} points", "", f"`{'#' * filled}{'-' * (20 - filled)}` {round(100 * earned / total) if total else 0}%", ""]
    out.append("**All tests passed. You can submit this run.**" if earned == total else
               "**Not all points yet.** The run shows a red cross until every check passes; your score is recorded either way and your best attempt counts.")
    for num, title in TASKS.items():
        group = [c for c in s["checks"] if c["name"].startswith(num + ".")]
        if not group:
            continue
        got, of = sum(c["points"] for c in group if c["passed"]), sum(c["points"] for c in group)
        out += ["", f"### Task {num}: {title} ({got}/{of})", "", "| | Check | Points |", "|---|---|---|"]
        out += [f"| {'✅' if c['passed'] else '❌'} | {c['name']} | {c['points'] if c['passed'] else 0}/{c['points']} |" for c in group]
        for c in group:
            reason = details.get(c["id"], "")
            if not c["passed"] and reason:
                out += ["", f"> ❌ **{c['name'].split(' ', 1)[0]}** {reason}"]
    text = "\n".join(out) + "\n"
    print(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)
        print(f"::notice title=Score::{earned}/{total} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())
