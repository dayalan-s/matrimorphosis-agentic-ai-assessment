#!/usr/bin/env bash
# Check that your machine is ready: bash scripts/check_setup.sh
cd "$(dirname "$0")/.."
ok=0; bad=0
pass() { echo "  OK    $*"; ok=$((ok+1)); }
fail() { echo "  FAIL  $*"; bad=$((bad+1)); }
echo "Checking your setup"
command -v git >/dev/null && pass "git $(git --version | cut -d' ' -f3)" || fail "git is not installed"
command -v uv >/dev/null && pass "uv $(uv --version | cut -d' ' -f2)" || fail "uv is not installed: https://docs.astral.sh/uv/getting-started/installation/"
if docker info >/dev/null 2>&1; then pass "Docker is running ($(docker --version | cut -d' ' -f3 | tr -d ,))"; else fail "Docker is not running (start Docker Desktop, or on Linux: sudo systemctl start docker)"; fi
docker compose version >/dev/null 2>&1 && pass "docker compose available" || fail "docker compose is missing (update Docker Desktop)"
if [ -f .env ]; then
    key=$(grep -E '^GROQ_API_KEY=' .env | head -1 | cut -d= -f2- | tr -d '"'"'"' ')
    if [ -z "$key" ] || [ "$key" = "your-groq-key-here" ]; then
        fail ".env has no GROQ_API_KEY: get a free key at https://console.groq.com/keys and paste it in .env"
    else
        code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' https://api.groq.com/openai/v1/models -H "Authorization: Bearer $key")
        case "$code" in
            200) pass "Groq API key works" ;;
            401) fail "Groq rejected the key (401): copy it again from console.groq.com/keys" ;;
            000) fail "could not reach api.groq.com (network, VPN or proxy?)" ;;
            *)   fail "Groq answered HTTP $code" ;;
        esac
    fi
else
    fail "no .env file: run  cp .env.example .env  and add your GROQ_API_KEY"
fi
for port in 8000 7860 3000 9090; do
    if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then fail "port $port is already in use (stop the other program, or 'docker compose down' if it is this project)"; else pass "port $port is free"; fi
done
if (exec 3<>/dev/tcp/127.0.0.1/6333) 2>/dev/null; then echo "  WARN  port 6333 (Qdrant) is in use: start with  QDRANT_PORT=6335 docker compose up --build -d"; else pass "port 6333 is free"; fi
[ -f uv.lock ] && pass "uv.lock present" || fail "uv.lock is missing (do not delete locked files)"
git config core.hooksPath >/dev/null 2>&1; [ -f .git/hooks/pre-commit ] && pass "pre-commit hooks installed" || echo "  TODO  install the hooks:  uv run --only-group hooks pre-commit install"
echo; echo "$ok OK, $bad problem(s)"; [ "$bad" -eq 0 ]
