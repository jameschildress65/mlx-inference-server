#!/bin/bash
# test_concurrency.sh
# Automated test suite for MLX server concurrency fix
# Tests that concurrent requests don't crash the server

set -e

# Configuration
SERVER_URL="http://localhost:11436"
MODEL="mlx-community/SmolLM2-1.7B-Instruct"
TEST_OUTPUT_DIR="/tmp/mlx_concurrency_tests"
TIMEOUT=300  # 5 minutes max per test

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Test counters
TESTS_RUN=0
TESTS_PASSED=0
TESTS_FAILED=0

# Cleanup function
cleanup() {
    echo -e "\n${CYAN}Cleaning up test files...${NC}"
    rm -rf "$TEST_OUTPUT_DIR"
}

# Trap cleanup on exit
trap cleanup EXIT

# Create test output directory
mkdir -p "$TEST_OUTPUT_DIR"

echo -e "${CYAN}=================================${NC}"
echo -e "${CYAN}MLX Server Concurrency Test Suite${NC}"
echo -e "${CYAN}=================================${NC}"
echo ""

# ==============================================================================
# Test 1: Server Health Check
# ==============================================================================
test_server_health() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "${YELLOW}[Test 1/5] Server Health Check${NC}"

    if curl -s -f "$SERVER_URL/v1/models" > /dev/null; then
        echo -e "${GREEN}✓ Server is running and responsive${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ Server is not running or not responsive${NC}"
        echo -e "${RED}  Please start server: python3 mlx_server_extended.py${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==============================================================================
# Test 2: Single Request Baseline
# ==============================================================================
test_single_request() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n${YELLOW}[Test 2/5] Single Request Baseline${NC}"

    local output="$TEST_OUTPUT_DIR/single_request.json"

    curl -s -X POST "$SERVER_URL/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$MODEL\",
            \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
            \"stream\": false,
            \"max_tokens\": 50
        }" > "$output" 2>&1

    # Check for valid response
    if jq -e '.choices[0].message.content' "$output" > /dev/null 2>&1; then
        local content=$(jq -r '.choices[0].message.content' "$output")
        echo -e "${GREEN}✓ Single request completed successfully${NC}"
        echo -e "  Response: ${content:0:50}..."
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}✗ Single request failed${NC}"
        echo -e "${RED}  Output: $(cat $output)${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==============================================================================
# Test 3: Concurrent Requests (Primary Test)
# ==============================================================================
test_concurrent_requests() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n${YELLOW}[Test 3/5] Concurrent Requests Stress Test${NC}"
    echo -e "  Sending 3 concurrent requests..."

    local start_time=$(date +%s)

    # Send 3 concurrent requests
    for i in {1..3}; do
        (curl -s -X POST "$SERVER_URL/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Count to 10\"}],
                \"stream\": false,
                \"max_tokens\": 100
            }" > "$TEST_OUTPUT_DIR/concurrent_$i.json" 2>&1) &
    done

    # Wait for all requests with timeout
    if timeout $TIMEOUT wait; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))

        # Check all responses
        local success_count=0
        local error_count=0

        for i in {1..3}; do
            if jq -e '.choices[0].message.content' "$TEST_OUTPUT_DIR/concurrent_$i.json" > /dev/null 2>&1; then
                success_count=$((success_count + 1))
            else
                error_count=$((error_count + 1))
                echo -e "${RED}  Request $i failed${NC}"
                cat "$TEST_OUTPUT_DIR/concurrent_$i.json"
            fi
        done

        if [ $success_count -eq 3 ]; then
            echo -e "${GREEN}✓ All 3 concurrent requests completed successfully${NC}"
            echo -e "  Duration: ${duration}s"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ Only $success_count/3 requests succeeded${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ Concurrent requests timed out after ${TIMEOUT}s${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==============================================================================
# Test 4: Server Stability Check
# ==============================================================================
test_server_stability() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n${YELLOW}[Test 4/5] Server Stability Check${NC}"

    # Get PID before test
    local pid_before=$(pgrep -f "mlx_server_extended.py" | head -1)

    if [ -z "$pid_before" ]; then
        echo -e "${RED}✗ Server process not found${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi

    # Check if server still responding after concurrent load
    if curl -s -f "$SERVER_URL/v1/models" > /dev/null; then
        local pid_after=$(pgrep -f "mlx_server_extended.py" | head -1)

        if [ "$pid_before" = "$pid_after" ]; then
            echo -e "${GREEN}✓ Server remained stable (same PID: $pid_before)${NC}"
            TESTS_PASSED=$((TESTS_PASSED + 1))
            return 0
        else
            echo -e "${RED}✗ Server PID changed (crashed and restarted?)${NC}"
            echo -e "${RED}  Before: $pid_before, After: $pid_after${NC}"
            TESTS_FAILED=$((TESTS_FAILED + 1))
            return 1
        fi
    else
        echo -e "${RED}✗ Server not responding after concurrent load${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# ==============================================================================
# Test 5: Error Log Analysis
# ==============================================================================
test_error_logs() {
    TESTS_RUN=$((TESTS_RUN + 1))
    echo -e "\n${YELLOW}[Test 5/5] Error Log Analysis${NC}"

    # Check for Metal GPU errors in test outputs
    if grep -i "error\|assertion\|crash\|encoder.*encoding" "$TEST_OUTPUT_DIR"/*.json > /dev/null 2>&1; then
        echo -e "${RED}✗ Found errors in response logs:${NC}"
        grep -i "error\|assertion\|crash\|encoder.*encoding" "$TEST_OUTPUT_DIR"/*.json
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    else
        echo -e "${GREEN}✓ No Metal GPU errors or crashes detected${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    fi
}

# ==============================================================================
# Run All Tests
# ==============================================================================

echo -e "${CYAN}Starting test execution...${NC}\n"

test_server_health || exit 1  # Exit if server not running
test_single_request
test_concurrent_requests
test_server_stability
test_error_logs

# ==============================================================================
# Test Summary
# ==============================================================================

echo -e "\n${CYAN}=================================${NC}"
echo -e "${CYAN}Test Summary${NC}"
echo -e "${CYAN}=================================${NC}"
echo -e "Total Tests:  $TESTS_RUN"
echo -e "${GREEN}Passed:       $TESTS_PASSED${NC}"

if [ $TESTS_FAILED -gt 0 ]; then
    echo -e "${RED}Failed:       $TESTS_FAILED${NC}"
    echo -e "\n${RED}CONCURRENCY FIX FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Failed:       0${NC}"
    echo -e "\n${GREEN}✓ ALL TESTS PASSED${NC}"
    echo -e "${GREEN}Concurrency fix is working correctly!${NC}"
    exit 0
fi
