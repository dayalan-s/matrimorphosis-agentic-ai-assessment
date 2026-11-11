#!/usr/bin/env bash
# Run the autograder on your machine, exactly like CI: scripted model, phase 1, restart, phase 2.
#   bash scripts/local_eval.sh            full run (builds two Docker images; first time takes a few minutes)
set -uo pipefail
cd "$(dirname "$0")/.."
RED='\033[0;31m'; GREEN='\033[0;32m'; BLUE='\033[0;34m'; NC='\033[0m'
info() { echo -e "${BLUE}[eval]${NC} $*"; }
CONTAINER="eval-local-$$"; MCP_CONTAINER="eval-local-mcp-$$"; VOLUME="eval-local-data-$$"; PORT="${EVAL_PORT:-8000}"; MCP_PORT="${EVAL_MCP_PORT:-9000}"
cleanup() { docker rm -f "$CONTAINER" "$MCP_CONTAINER" >/dev/null 2>&1 || true; docker rmi "eval-app-local-$$" "eval-test-local-$$" >/dev/null 2>&1 || true; docker volume rm "$VOLUME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

info "building the app image (runner) and the test image"
docker build -q --target runner -t eval-app-local-$$ . >/dev/null || { echo -e "${RED}[eval]${NC} app image failed to build"; exit 1; }
docker build -q --target test -t eval-test-local-$$ . >/dev/null || { echo -e "${RED}[eval]${NC} test image failed to build"; exit 1; }

rm -rf results && mkdir -p results
python3 scripts/check_async.py
python3 scripts/check_integrity.py
docker volume create "$VOLUME" >/dev/null
info "alembic round trip (upgrade, downgrade, upgrade)"
docker run --rm -v "$VOLUME:/data" eval-app-local-$$ python scripts/check_migration.py > results/migration.json || echo '{}' > results/migration.json
wait_mcp() { for _ in $(seq 1 30); do (echo > "/dev/tcp/127.0.0.1/$MCP_PORT") 2>/dev/null && return 0; sleep 1; done; return 1; }
# host network: the app reaches the MCP server at localhost (ports come from EVAL_PORT / EVAL_MCP_PORT)
docker run -d --name "$MCP_CONTAINER" --network host -v "$VOLUME:/data" -e "MCP_PORT=$MCP_PORT" eval-app-local-$$ python src/app/mcp_server/server.py --http >/dev/null
wait_mcp || { echo -e "${RED}[eval]${NC} the MCP server did not start. Its log:"; docker logs --tail 40 "$MCP_CONTAINER"; exit 1; }
docker run -d --name "$CONTAINER" --network host --memory 1g --cpus 1 -v "$VOLUME:/data" -e "MCP_URL=http://localhost:$MCP_PORT/mcp" eval-app-local-$$ uvicorn app.main:app --app-dir src --host 0.0.0.0 --port "$PORT" >/dev/null

wait_health() {
    for _ in $(seq 1 60); do curl -sf "http://localhost:$PORT/health" >/dev/null && return 0; sleep 2; done
    echo -e "${RED}[eval]${NC} the app did not become healthy. Its log:"; docker logs --tail 40 "$CONTAINER"; return 1
}
run_tests() {  # $1 = phase marker
    docker run --rm --network host -e BASE_URL="http://localhost:$PORT" -e MCP_URL="http://localhost:$MCP_PORT/mcp" -e ASSESSMENT_ID=agentic-ai-extension \
        -v "$PWD/results:/app/results" eval-test-local-$$ pytest tests/ -m "$1" --tb=short -q
}
wait_health || exit 1
info "phase 1: before the restart"; run_tests phase1
info "restarting the container (same data volume)"; docker restart "$MCP_CONTAINER" >/dev/null; wait_mcp; docker restart "$CONTAINER" >/dev/null; wait_health || exit 1
info "phase 2: after the restart"; run_tests phase2

echo
python3 scripts/render_summary.py
echo -e "\n(the app's log is in 'docker logs'; it is removed with the container. Re-run after fixing.)"
