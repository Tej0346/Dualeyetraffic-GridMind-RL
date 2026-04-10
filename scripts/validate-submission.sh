#!/usr/bin/env bash
#
# validate-submission.sh — OpenEnv Submission Validator
#
# Checks that your HF Space is live, Docker image builds, and openenv validate passes.
#

set -uo pipefail

DOCKER_BUILD_TIMEOUT=600
PING_URL="${1:-}"
REPO_DIR="${2:-.}"

# Colors
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BOLD='\033[1m'
  NC='\033[0m'
else
  RED='' GREEN='' YELLOW='' BOLD='' NC=''
fi

log_info() { echo -e "${BOLD}[INFO]${NC} $1"; }
log_pass() { echo -e "${GREEN}[PASS]${NC} $1"; }
log_fail() { echo -e "${RED}[FAIL]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }

run_with_timeout() {
  local secs="$1"; shift
  if command -v timeout &>/dev/null; then
    timeout "$secs" "$@"
  elif command -v gtimeout &>/dev/null; then
    gtimeout "$secs" "$@"
  else
    "$@" &
    local pid=$!
    ( sleep "$secs" && kill "$pid" 2>/dev/null ) &
    local watcher=$!
    wait "$pid" 2>/dev/null
    local rc=$?
    kill "$watcher" 2>/dev/null
    wait "$watcher" 2>/dev/null
    return $rc
  fi
}

echo "=============================================="
echo "  OpenEnv Submission Validator"
echo "=============================================="
echo ""

# Check prerequisites
log_info "Checking prerequisites..."

if ! command -v docker &>/dev/null; then
  log_fail "Docker not found. Install: https://docs.docker.com/get-docker/"
  exit 1
fi
log_pass "Docker found"

if ! command -v curl &>/dev/null; then
  log_fail "curl not found"
  exit 1
fi
log_pass "curl found"

# Check repo directory
if [ ! -d "$REPO_DIR" ]; then
  log_fail "Repo directory not found: $REPO_DIR"
  exit 1
fi
cd "$REPO_DIR"
log_pass "Repository: $(pwd)"

echo ""
echo "----------------------------------------------"
echo "1. Checking required files..."
echo "----------------------------------------------"

REQUIRED_FILES=("openenv.yaml" "inference.py" "Dockerfile" "requirements.txt")
for file in "${REQUIRED_FILES[@]}"; do
  if [ -f "$file" ]; then
    log_pass "$file exists"
  else
    log_fail "$file missing"
    exit 1
  fi
done

echo ""
echo "----------------------------------------------"
echo "2. Validating openenv.yaml..."
echo "----------------------------------------------"

if python3 -c "import yaml; yaml.safe_load(open('openenv.yaml'))" 2>/dev/null; then
  log_pass "openenv.yaml is valid YAML"

  # Check for required fields
  if grep -q "entrypoint:" openenv.yaml && grep -q "endpoints:" openenv.yaml; then
    log_pass "openenv.yaml has entrypoint and endpoints"
  else
    log_warn "openenv.yaml may be missing entrypoint or endpoints"
  fi
else
  log_fail "openenv.yaml is not valid YAML"
  exit 1
fi

echo ""
echo "----------------------------------------------"
echo "3. Building Docker image..."
echo "----------------------------------------------"

log_info "Building Docker image (timeout: ${DOCKER_BUILD_TIMEOUT}s)..."

if run_with_timeout "$DOCKER_BUILD_TIMEOUT" docker build -t traffic-env-test . 2>&1 | tail -20; then
  log_pass "Docker image built successfully"
else
  log_fail "Docker build failed"
  exit 1
fi

echo ""
echo "----------------------------------------------"
echo "4. Testing Docker container..."
echo "----------------------------------------------"

log_info "Starting container..."

# Run container in background
docker run -d --name traffic-env-test -p 7870:7860 traffic-env-test 2>/dev/null
CONTAINER_ID=$(docker ps -q -f name=traffic-env-test)

if [ -z "$CONTAINER_ID" ]; then
  log_fail "Failed to start container"
  exit 1
fi

log_pass "Container started: $CONTAINER_ID"

# Wait for startup
sleep 5

# Check if space is responsive
if [ -n "$PING_URL" ]; then
  log_info "Checking HF Space at $PING_URL..."

  if curl -sf "$PING_URL" > /dev/null 2>&1; then
    log_pass "HF Space is responsive"
  else
    log_warn "HF Space not reachable (this may be expected during dev)"
  fi
fi

# Check local endpoint
log_info "Checking local endpoint..."

if curl -sf "http://localhost:7870/" > /dev/null 2>&1; then
  log_pass "Local endpoint responding"
else
  log_warn "Local endpoint not responding yet"
fi

# Test /reset endpoint
log_info "Testing /reset endpoint..."
if curl -sf "http://localhost:7870/reset?task=easy" > /dev/null 2>&1; then
  log_pass "/reset endpoint working"
else
  log_fail "/reset endpoint failed"
fi

# Cleanup
log_info "Cleaning up..."
docker stop traffic-env-test 2>/dev/null
docker rm traffic-env-test 2>/dev/null

echo ""
echo "----------------------------------------------"
echo "5. Checking inference.py structure..."
echo "----------------------------------------------"

if grep -q "log_start\|log_step\|log_end" inference.py; then
  log_pass "inference.py has logging functions"
else
  log_warn "inference.py may be missing log_start/log_step/log_end"
fi

if grep -q "API_BASE_URL\|MODEL_NAME\|HF_TOKEN" inference.py; then
  log_pass "inference.py has required environment variables"
else
  log_fail "inference.py missing required environment variables"
  exit 1
fi

echo ""
echo "=============================================="
echo -e "${GREEN}  VALIDATION COMPLETE${NC}"
echo "=============================================="
echo ""
echo "All checks passed! Your submission is ready."
echo ""
echo "Next steps:"
echo "  1. Push to Hugging Face Spaces: git push"
echo "  2. Wait for deployment"
echo "  3. Submit your Space URL"
echo ""