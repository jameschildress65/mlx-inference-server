#!/bin/bash
#
# MLX Inference Server v3.2.0 Feature Test Suite
#
# Tests all new features introduced in v3.2.0:
# - JSON Mode (structured output)
# - Priority Queue
# - Rate Limiting
# - Prometheus Metrics
#
# Usage: ./bin/test-v3.2-features.sh [host] [port]
#        Default: localhost 11440
#
# Exit codes:
#   0 = All tests passed
#   1 = One or more tests failed

set -e

HOST="${1:-localhost}"
PORT="${2:-11440}"
ADMIN_PORT="${3:-11441}"
BASE_URL="http://${HOST}:${PORT}"
ADMIN_URL="http://${HOST}:${ADMIN_PORT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

PASSED=0
FAILED=0
SKIPPED=0

# Test helper functions
log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAILED++))
}

log_skip() {
    echo -e "${YELLOW}[SKIP]${NC} $1"
    ((SKIPPED++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Check if server is running
check_server() {
    log_test "Checking server availability..."
    if curl -s --connect-timeout 5 "${BASE_URL}/health" > /dev/null 2>&1; then
        log_pass "Server is running at ${BASE_URL}"
        return 0
    else
        log_fail "Server not responding at ${BASE_URL}"
        echo "Please start the server first: ./bin/mlx-inference-server-daemon.sh"
        exit 1
    fi
}

# Check if model is loaded
check_model() {
    log_test "Checking if model is loaded..."
    local status=$(curl -s "${ADMIN_URL}/admin/status")
    local loaded=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin)['model']['loaded'])" 2>/dev/null || echo "false")

    if [ "$loaded" = "True" ] || [ "$loaded" = "true" ]; then
        local model=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin)['model']['name'])" 2>/dev/null)
        log_pass "Model loaded: $model"
        return 0
    else
        log_info "No model loaded - will trigger on-demand loading"
        return 0
    fi
}

# ============================================================
# TEST 1: Basic Health Endpoints
# ============================================================
test_health_endpoints() {
    echo ""
    echo "=========================================="
    echo "TEST 1: Health Endpoints"
    echo "=========================================="

    # /health endpoint
    log_test "Testing /health endpoint..."
    local health=$(curl -s -w "%{http_code}" -o /tmp/health_response.json "${BASE_URL}/health")
    if [ "$health" = "200" ]; then
        log_pass "/health returns 200"
    else
        log_fail "/health returned $health (expected 200)"
    fi

    # /ready endpoint
    log_test "Testing /ready endpoint..."
    local ready=$(curl -s -w "%{http_code}" -o /tmp/ready_response.json "${BASE_URL}/ready")
    if [ "$ready" = "200" ] || [ "$ready" = "503" ]; then
        log_pass "/ready returns $ready (valid response)"
    else
        log_fail "/ready returned unexpected code: $ready"
    fi

    # /admin/status endpoint
    log_test "Testing /admin/status endpoint..."
    local status=$(curl -s -w "%{http_code}" -o /tmp/status_response.json "${ADMIN_URL}/admin/status")
    if [ "$status" = "200" ]; then
        log_pass "/admin/status returns 200"
        # Check version
        local version=$(cat /tmp/status_response.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('version','unknown'))" 2>/dev/null)
        if [ "$version" = "3.2.0" ]; then
            log_pass "Server version is 3.2.0"
        else
            log_fail "Server version is $version (expected 3.2.0)"
        fi
    else
        log_fail "/admin/status returned $status"
    fi
}

# ============================================================
# TEST 2: JSON Mode (Structured Output)
# ============================================================
test_json_mode() {
    echo ""
    echo "=========================================="
    echo "TEST 2: JSON Mode (Structured Output)"
    echo "=========================================="

    # Test 2a: json_object mode
    log_test "Testing json_object mode..."
    local response=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Return a JSON object with keys: name (string), count (number). Just the JSON, nothing else."}],
            "max_tokens": 100,
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }' 2>/dev/null)

    if echo "$response" | python3 -c "import sys,json; data=json.load(sys.stdin); content=data['choices'][0]['message']['content']; json.loads(content); print('valid')" 2>/dev/null | grep -q "valid"; then
        log_pass "json_object mode returns valid JSON"
    else
        log_fail "json_object mode did not return valid JSON"
        echo "Response: $response"
    fi

    # Test 2b: json_schema mode
    log_test "Testing json_schema mode with schema..."
    local response=$(curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Generate a person with name and age."}],
            "max_tokens": 100,
            "temperature": 0.1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"}
                    },
                    "required": ["name", "age"]
                }
            }
        }' 2>/dev/null)

    local content=$(echo "$response" | python3 -c "import sys,json; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    if echo "$content" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'name' in d and 'age' in d; print('valid')" 2>/dev/null | grep -q "valid"; then
        log_pass "json_schema mode returns schema-compliant JSON"
        log_info "Generated: $content"
    else
        log_fail "json_schema mode did not return schema-compliant JSON"
        echo "Response: $response"
    fi

    # Test 2c: Schema validation (should reject oversized schema)
    log_test "Testing schema size validation..."
    # Create a schema > 64KB (will be rejected)
    local big_schema='{"type":"object","properties":{'
    for i in {1..1000}; do
        big_schema+="\"field_$i\":{\"type\":\"string\",\"description\":\"$(printf 'x%.0s' {1..100})\"},"
    done
    big_schema="${big_schema%,}}}"

    local response=$(curl -s -w "%{http_code}" -o /tmp/schema_error.json -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"mlx-community/Qwen2.5-7B-Instruct-4bit\",
            \"messages\": [{\"role\": \"user\", \"content\": \"test\"}],
            \"response_format\": {\"type\": \"json_schema\", \"json_schema\": $big_schema}
        }" 2>/dev/null)

    if [ "$response" = "422" ] || [ "$response" = "400" ]; then
        log_pass "Oversized schema rejected with $response"
    else
        log_fail "Oversized schema should be rejected (got $response)"
    fi
}

# ============================================================
# TEST 3: Priority Queue
# ============================================================
test_priority_queue() {
    echo ""
    echo "=========================================="
    echo "TEST 3: Priority Queue"
    echo "=========================================="

    # Test 3a: Check queue status endpoint
    log_test "Testing /admin/queue endpoint..."
    local queue_status=$(curl -s "${ADMIN_URL}/admin/queue")
    local enabled=$(echo "$queue_status" | python3 -c "import sys,json; print(json.load(sys.stdin).get('enabled', False))" 2>/dev/null)

    if [ "$enabled" = "True" ] || [ "$enabled" = "true" ]; then
        log_pass "Priority queue is enabled"
    else
        log_info "Priority queue is disabled (using semaphore fallback)"
    fi

    # Test 3b: X-Priority header acceptance
    log_test "Testing X-Priority: high header..."
    local response=$(curl -s -w "%{http_code}" -o /tmp/priority_response.json -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: high" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Say hi"}],
            "max_tokens": 10
        }' 2>/dev/null)

    if [ "$response" = "200" ]; then
        log_pass "X-Priority: high accepted"
    else
        log_fail "X-Priority: high request failed with $response"
    fi

    # Test 3c: X-Priority: low header
    log_test "Testing X-Priority: low header..."
    local response=$(curl -s -w "%{http_code}" -o /dev/null -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: low" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Say bye"}],
            "max_tokens": 10
        }' 2>/dev/null)

    if [ "$response" = "200" ]; then
        log_pass "X-Priority: low accepted"
    else
        log_fail "X-Priority: low request failed with $response"
    fi

    # Test 3d: Invalid priority defaults to normal
    log_test "Testing invalid X-Priority defaults to normal..."
    local response=$(curl -s -w "%{http_code}" -o /dev/null -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: INVALID" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Test"}],
            "max_tokens": 10
        }' 2>/dev/null)

    if [ "$response" = "200" ]; then
        log_pass "Invalid priority handled gracefully"
    else
        log_fail "Invalid priority caused error: $response"
    fi
}

# ============================================================
# TEST 4: Prometheus Metrics
# ============================================================
test_prometheus_metrics() {
    echo ""
    echo "=========================================="
    echo "TEST 4: Prometheus Metrics"
    echo "=========================================="

    # Test 4a: /metrics endpoint exists
    log_test "Testing /metrics endpoint..."
    local response=$(curl -s -w "%{http_code}" -o /tmp/metrics.txt "${ADMIN_URL}/metrics")

    if [ "$response" = "200" ]; then
        log_pass "/metrics endpoint available"

        # Check for expected metrics
        if grep -q "mlx_requests_total" /tmp/metrics.txt 2>/dev/null; then
            log_pass "mlx_requests_total metric present"
        else
            log_fail "mlx_requests_total metric missing"
        fi

        if grep -q "mlx_request_duration" /tmp/metrics.txt 2>/dev/null; then
            log_pass "mlx_request_duration metric present"
        else
            log_fail "mlx_request_duration metric missing"
        fi
    else
        log_fail "/metrics returned $response"
    fi

    # Test 4b: /admin/metrics endpoint
    log_test "Testing /admin/metrics endpoint..."
    local response=$(curl -s -w "%{http_code}" -o /tmp/admin_metrics.json "${ADMIN_URL}/admin/metrics")

    if [ "$response" = "200" ]; then
        log_pass "/admin/metrics endpoint available"
        local total=$(cat /tmp/admin_metrics.json | python3 -c "import sys,json; print(json.load(sys.stdin).get('total_requests', 0))" 2>/dev/null)
        log_info "Total requests recorded: $total"
    else
        log_fail "/admin/metrics returned $response"
    fi
}

# ============================================================
# TEST 5: Rate Limiting (if enabled)
# ============================================================
test_rate_limiting() {
    echo ""
    echo "=========================================="
    echo "TEST 5: Rate Limiting"
    echo "=========================================="

    # Check if rate limiting is enabled
    local metrics=$(curl -s "${ADMIN_URL}/admin/metrics")
    local enabled=$(echo "$metrics" | python3 -c "import sys,json; print(json.load(sys.stdin).get('rate_limiter',{}).get('enabled', False))" 2>/dev/null)

    if [ "$enabled" = "True" ] || [ "$enabled" = "true" ]; then
        log_pass "Rate limiting is enabled"

        # Get current tokens
        local tokens=$(echo "$metrics" | python3 -c "import sys,json; print(json.load(sys.stdin).get('rate_limiter',{}).get('current_tokens', 0))" 2>/dev/null)
        log_info "Current tokens: $tokens"

        # Test burst behavior would require many rapid requests
        log_skip "Burst test skipped (would require many rapid requests)"
    else
        log_info "Rate limiting is disabled (default for home lab)"
        log_skip "Rate limiting tests skipped - enable with MLX_RATE_LIMIT_ENABLED=1"
    fi
}

# ============================================================
# TEST 6: Streaming with New Features
# ============================================================
test_streaming() {
    echo ""
    echo "=========================================="
    echo "TEST 6: Streaming Requests"
    echo "=========================================="

    # Test 6a: Streaming chat completion
    log_test "Testing streaming chat completion..."
    local response=$(curl -s -N -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Count from 1 to 5"}],
            "max_tokens": 50,
            "stream": true
        }' 2>/dev/null | head -20)

    if echo "$response" | grep -q "data:"; then
        log_pass "Streaming returns SSE data chunks"
    else
        log_fail "Streaming did not return SSE format"
        echo "Response: $response"
    fi

    # Test 6b: Streaming with priority
    log_test "Testing streaming with X-Priority header..."
    local response=$(curl -s -N -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: high" \
        -d '{
            "model": "mlx-community/Qwen2.5-7B-Instruct-4bit",
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 20,
            "stream": true
        }' 2>/dev/null | head -10)

    if echo "$response" | grep -q "data:"; then
        log_pass "Streaming with priority works"
    else
        log_fail "Streaming with priority failed"
    fi
}

# ============================================================
# TEST 7: Concurrent Request Handling
# ============================================================
test_concurrent() {
    echo ""
    echo "=========================================="
    echo "TEST 7: Concurrent Requests"
    echo "=========================================="

    log_test "Sending 3 concurrent requests with different priorities..."

    # Launch 3 requests in background
    curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: low" \
        -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "messages": [{"role": "user", "content": "Say low"}], "max_tokens": 10}' \
        > /tmp/concurrent_low.json 2>&1 &
    local pid_low=$!

    curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -H "X-Priority: high" \
        -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "messages": [{"role": "user", "content": "Say high"}], "max_tokens": 10}' \
        > /tmp/concurrent_high.json 2>&1 &
    local pid_high=$!

    curl -s -X POST "${BASE_URL}/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model": "mlx-community/Qwen2.5-7B-Instruct-4bit", "messages": [{"role": "user", "content": "Say normal"}], "max_tokens": 10}' \
        > /tmp/concurrent_normal.json 2>&1 &
    local pid_normal=$!

    # Wait for all
    wait $pid_low $pid_high $pid_normal

    # Check results
    local success=0
    for f in /tmp/concurrent_low.json /tmp/concurrent_high.json /tmp/concurrent_normal.json; do
        if cat "$f" | python3 -c "import sys,json; json.load(sys.stdin)['choices'][0]" 2>/dev/null; then
            ((success++))
        fi
    done

    if [ "$success" -eq 3 ]; then
        log_pass "All 3 concurrent requests completed successfully"
    else
        log_fail "Only $success/3 concurrent requests succeeded"
    fi
}

# ============================================================
# MAIN
# ============================================================
main() {
    echo ""
    echo "=============================================="
    echo "  MLX Inference Server v3.2.0 Feature Tests"
    echo "=============================================="
    echo "Target: ${BASE_URL}"
    echo "Admin:  ${ADMIN_URL}"
    echo ""

    check_server
    check_model

    test_health_endpoints
    test_json_mode
    test_priority_queue
    test_prometheus_metrics
    test_rate_limiting
    test_streaming
    test_concurrent

    echo ""
    echo "=============================================="
    echo "  TEST SUMMARY"
    echo "=============================================="
    echo -e "${GREEN}Passed:${NC}  $PASSED"
    echo -e "${RED}Failed:${NC}  $FAILED"
    echo -e "${YELLOW}Skipped:${NC} $SKIPPED"
    echo ""

    if [ "$FAILED" -eq 0 ]; then
        echo -e "${GREEN}All tests passed!${NC}"
        exit 0
    else
        echo -e "${RED}Some tests failed.${NC}"
        exit 1
    fi
}

main "$@"
