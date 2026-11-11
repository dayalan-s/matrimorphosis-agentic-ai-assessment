# Agentic AI assessment: integrate the Document Agent - how to submit

## At a glance
- Build: integrate the Document Agent: nine tasks in order, 34 points (see `docs/TASK.md`)
- Window: **5 Nov 2026, 00:00 IST** to **12 Nov 2026, 23:59 IST** (judged on when your GitHub Actions run finishes)
- Pass: **all 34 points** (every task is required; partial scores are recorded too)
- You may submit as many times as you like; **your best score is kept**
- Result arrives by **email** (check spam)

## What you need
- GitHub account (free)
- Google account (the form asks you to sign in; your result goes to that email)
- Git, Docker, and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed (setup: `README.md`)
- You do **not** need your Groq key for the submission: GitHub runs your code with a scripted model

## Steps

### 1. Create your repo
- Open the template repository link your instructor shared (named `matrimorphosis-agentic-ai-assessment`; it is public) then **Use this template** then **Create a new repository**
- Owner: **your own account** (not an organisation)
- Visibility: **Private** (so other students cannot see your solution)
- Use **Use this template**, **not Fork**: a fork of a public repository is always public, and GitHub Actions is switched off in forks
- Clone it:
  ```bash
  git clone https://github.com/<you>/<your-repo>
  cd <your-repo>
  ```

### 2. Set up and turn on the local checks (once)
Follow `README.md` (Groq key, `bash scripts/check_setup.sh`, `docker compose up --build -d`), then:
```bash
uv run --only-group hooks pre-commit install
```
- Before each commit: blocks changes to locked files
- Before each push: also runs the autograder locally in Docker (a few minutes)
- In a hurry? `git push --no-verify` skips the local run; the GitHub Action is the real grader

### 3. Write your solution
- Edit **only the files listed in `docs/TASK.md`** (nine files under `src/app/`, plus new files in `src/app/migrations/` and `src/app/dashboards/`; `README.md` and `.gitignore` are also free)
- Everything else is locked and compared with the template, byte for byte:
  - `tests/`, `Dockerfile`, `pyproject.toml`, `uv.lock`, `docker-compose.yml`, `alembic.ini`, `alembic/`, `.env.example`, `.github/`, `scripts/`, `docs/`, `data/`, `samples/`, `ui/`, `observability/`, `.pre-commit-config.yaml`, `.checksums`, `.gitattributes`
- Adding, deleting or changing any locked file makes the submission invalid
- If a commit is blocked, the message lists the files; fix with:
  ```bash
  git restore --staged --worktree <file>      # changed or deleted file
  git rm -f --cached <file> && rm <file>      # file you should not have added
  ```

### 4. Check locally
```bash
bash scripts/local_eval.sh       # the full autograder: scripted model, restart, second phase. Prints PASS/FAIL per check
```

### 5. Push and wait for the run
```bash
git add -A && git commit -m "my solution" && git push origin main
```
- Push to **`main`** only
- Open the **Actions** tab, wait for the run named **evaluate** to finish (green tick or red cross are both fine; it takes about 5-10 minutes; see **Why is my run red?** below)
- The run's **Summary** tab shows your score, a table per task and what to fix for each failed check
- Copy the run URL: `https://github.com/<you>/<repo>/actions/runs/<number>`

### 6. Create a read-only token
GitHub, then **Settings, Developer settings, Personal access tokens, Fine-grained tokens, Generate new token**

| Field | Value |
|---|---|
| Expiration | 7 days |
| Resource owner | your own account |
| Repository access | **Only select repositories**, then your repo |
| Permissions | **Actions: Read**, **Contents: Read** (Metadata: Read is added automatically) |

- Copy the token (starts with `github_pat_`); GitHub shows it once

### 7. Submit the form
- Open the submission form link your instructor shared and sign in with Google
- Fill: name, run URL, token
- Submit; your response (including the token) is **deleted automatically** right after verification
- After you get your result email, **revoke the token** on GitHub (Settings, Fine-grained tokens, Delete)

## What gets checked
- The run is from the official **evaluate** workflow, a **push to `main`** of **your** repo
- Every locked file (everything except the files you may edit, see `docs/TASK.md`) equals the template at that commit
- The run finished inside the window
- Test results belong to this assessment and to the same commit
- One GitHub account per Google email (do not share repos or tokens)
- Solutions are compared automatically across students
- Your code must not detect or special-case the grader (`LLM_MODE`, `ASSESSMENT_DEBUG` and the like): `scripts/check_integrity.py` sets the score to 0 if it does
- **There is no manual review: your grade is the score of the CI run**

## Scoring
The full table of the 34 checks (ids, points, what each one proves) is in `docs/TASK.md`. The email lists every check as PASS or FAIL.

## Why is my run red?
The last step of the **evaluate** run is called **Fail the run if any test failed**. It fails **on purpose** whenever your score is below 34/34, so the run shows a **red cross** until every check passes. A red run is not an error:
- the tests still ran and your score was written (open the run's **Summary** tab: score, a table per task, and what to fix for each failed check)
- **partial points are recorded**: submit the run URL through the form, red or green, and you get an email with your score
- **your best score is kept**: fix a check, push, wait for the new run and submit its URL again; a higher score replaces the old one, a lower one never does
- the run only turns **green** at 34/34
A run that fails *before* the tests (for example the Docker build breaks) has no score: the email says "no test results". Fix the error shown in the failed step and push again.

## If something goes wrong
The email says what failed. Common ones:

| Email says | Check |
|---|---|
| Token was not accepted | copied fully, not expired, belongs to the account that owns the repo |
| Could not open your repository | token's repository access includes this repo; Actions, Contents, Metadata set to Read |
| Run does not exist | URL copied from your repo's Actions tab |
| Run not finished | wait for the green tick or red cross, then resubmit |
| Not from the official workflow | submit the **evaluate** run; do not add or edit `.github/workflows/` |
| Not a normal push to main | push to `main`; do not use manual or pull-request runs |
| Locked files changed | restore the listed files, push, submit the **new** run |
| Outside the submission window | contact the instructor |
| No test results | re-run the workflow, then submit the new run |
| Account already registered with another email | use your original Google account, or contact the instructor |

An invalid submission records **no score** and never lowers an earlier one. Fix, push, and submit the new run URL.

## Contact
Contact your instructor.
