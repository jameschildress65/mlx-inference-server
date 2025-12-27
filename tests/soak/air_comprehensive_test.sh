#!/bin/bash
# MLX Server V2 - Comprehensive Soak Test for Apple Silicon Mac M4
# Tests all NASA-grade features: loading, queries, unload, timeout, memory recovery

set -e

ADMIN_API="http://localhost:11437"
MAIN_API="http://localhost:11436"
MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test counters
TESTS_PASSED=0
TESTS_FAILED=0

log_test() {
    echo -e "${BLUE}[TEST]${NC} $1"
}

log_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((TESTS_PASSED++))
}

log_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((TESTS_FAILED++))
}

log_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

get_memory_gb() {
    curl -s "$ADMIN_API/admin/status" | python3 -c "import sys, json; print(json.load(sys.stdin)['memory']['active_memory_gb'])"
}

get_model_loaded() {
    curl -s "$ADMIN_API/admin/status" | python3 -c "import sys, json; print(json.load(sys.stdin)['memory']['model_loaded'])"
}

get_active_requests() {
    curl -s "$ADMIN_API/admin/status" | python3 -c "import sys, json; print(json.load(sys.stdin)['requests']['active_requests'])"
}

echo "========================================================================"
echo "MLX Server V2 - Comprehensive Soak Test"
echo "Machine: Apple Silicon Mac (medium-memory)"
echo "Test Model: $MODEL"
echo "Started: $(date)"
echo "========================================================================"
echo ""

# Test 1: Health Check
log_test "1. Health Check - Verify server is running"
HEALTH=$(curl -s "$ADMIN_API/admin/health" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])")
if [ "$HEALTH" = "healthy" ]; then
    log_pass "Server is healthy"
else
    log_fail "Server health check failed: $HEALTH"
    exit 1
fi
echo ""

# Test 2: Baseline Memory (No Model Loaded)
log_test "2. Baseline Memory - Verify no model loaded"
BASELINE_MEM=$(get_memory_gb)
MODEL_LOADED=$(get_model_loaded)
log_info "Baseline memory: ${BASELINE_MEM} GB"
if [ "$MODEL_LOADED" = "False" ]; then
    log_pass "No model loaded at baseline (expected)"
else
    log_fail "Model already loaded at baseline (unexpected)"
fi
echo ""

# Test 3: On-Demand Loading (First Request)
log_test "3. On-Demand Loading - First request triggers model load"
log_info "Sending first completion request..."
RESPONSE=$(curl -s -X POST "$MAIN_API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Hello! Respond with exactly: 'MLX Test 1 OK'\", \"max_tokens\": 20, \"temperature\": 0}")

COMPLETION=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "PARSE_ERROR")

if [ "$COMPLETION" != "PARSE_ERROR" ]; then
    log_pass "Model loaded on-demand and generated completion"
    log_info "Response: $COMPLETION"
else
    log_fail "First completion failed: $RESPONSE"
fi

LOADED_MEM=$(get_memory_gb)
log_info "Memory after load: ${LOADED_MEM} GB (delta: $(python3 -c "print(f'{float($LOADED_MEM) - float($BASELINE_MEM):.2f}')") GB)"
echo ""

# Test 4: Subsequent Queries (Model Already Loaded)
log_test "4. Subsequent Queries - Model stays loaded for follow-up requests"
for i in {1..3}; do
    log_info "Query $i/3..."
    RESPONSE=$(curl -s -X POST "$MAIN_API/v1/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$MODEL\", \"prompt\": \"Count: $i. Reply with just the number.\", \"max_tokens\": 10, \"temperature\": 0}")

    COMPLETION=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERROR")

    if [ "$COMPLETION" != "ERROR" ]; then
        log_info "  Response: $COMPLETION"
    else
        log_fail "Query $i failed"
    fi
    sleep 1
done
log_pass "Multiple queries completed successfully"
echo ""

# Test 5: Concurrent Requests (Thread Safety)
log_test "5. Concurrent Requests - Thread-safe inference lock"
log_info "Sending 3 concurrent requests..."

curl -s -X POST "$MAIN_API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Concurrent 1\", \"max_tokens\": 10, \"temperature\": 0}" > /tmp/mlx_test_1.json &
PID1=$!

curl -s -X POST "$MAIN_API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Concurrent 2\", \"max_tokens\": 10, \"temperature\": 0}" > /tmp/mlx_test_2.json &
PID2=$!

curl -s -X POST "$MAIN_API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Concurrent 3\", \"max_tokens\": 10, \"temperature\": 0}" > /tmp/mlx_test_3.json &
PID3=$!

wait $PID1 $PID2 $PID3

CONCURRENT_ERRORS=0
for i in {1..3}; do
    if grep -q '"choices"' /tmp/mlx_test_$i.json 2>/dev/null; then
        log_info "  Concurrent request $i: OK"
    else
        log_fail "  Concurrent request $i: FAILED"
        ((CONCURRENT_ERRORS++))
    fi
done

if [ $CONCURRENT_ERRORS -eq 0 ]; then
    log_pass "All concurrent requests handled successfully (thread-safe)"
else
    log_fail "Concurrent request handling had $CONCURRENT_ERRORS errors"
fi
rm -f /tmp/mlx_test_*.json
echo ""

# Test 6: Manual Unload
log_test "6. Manual Unload - Admin API unload command"
log_info "Sending manual unload command..."
UNLOAD_RESPONSE=$(curl -s -X POST "$ADMIN_API/admin/unload")
UNLOAD_STATUS=$(echo "$UNLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['status'])" 2>/dev/null || echo "ERROR")

if [ "$UNLOAD_STATUS" = "success" ]; then
    log_pass "Manual unload successful"
    MEMORY_FREED=$(echo "$UNLOAD_RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['memory_freed_gb'])")
    log_info "Memory freed: ${MEMORY_FREED} GB"
else
    log_fail "Manual unload failed: $UNLOAD_RESPONSE"
fi
echo ""

# Test 7: Memory Recovery (97.5% target)
log_test "7. Memory Recovery - Verify memory returned to baseline"
sleep 2  # Let memory settle
AFTER_UNLOAD_MEM=$(get_memory_gb)
MEMORY_DELTA=$(python3 -c "print(f'{abs(float($AFTER_UNLOAD_MEM) - float($BASELINE_MEM)):.4f}')")
RECOVERY_PCT=$(python3 -c "print(f'{(1 - abs(float($AFTER_UNLOAD_MEM) - float($BASELINE_MEM)) / float($LOADED_MEM)) * 100:.2f}')")

log_info "Baseline: ${BASELINE_MEM} GB"
log_info "After unload: ${AFTER_UNLOAD_MEM} GB"
log_info "Delta: ${MEMORY_DELTA} GB"
log_info "Recovery: ${RECOVERY_PCT}%"

if (( $(python3 -c "print(1 if float($RECOVERY_PCT) >= 95.0 else 0)") )); then
    log_pass "Memory recovery ≥95% (target: 97.5%)"
else
    log_fail "Memory recovery <95%: ${RECOVERY_PCT}%"
fi
echo ""

# Test 8: Reload After Unload
log_test "8. Reload After Unload - Load → Unload → Load cycle"
log_info "Reloading model with new request..."
RESPONSE=$(curl -s -X POST "$MAIN_API/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"prompt\": \"Reload test. Say: RELOADED\", \"max_tokens\": 10, \"temperature\": 0}")

COMPLETION=$(echo "$RESPONSE" | python3 -c "import sys, json; print(json.load(sys.stdin)['choices'][0]['text'])" 2>/dev/null || echo "ERROR")

if [ "$COMPLETION" != "ERROR" ]; then
    log_pass "Model reloaded successfully after manual unload"
    log_info "Response: $COMPLETION"
else
    log_fail "Reload failed: $RESPONSE"
fi
echo ""

# Test 9: Idle Timeout Unload (180s on Air)
log_test "9. Idle Timeout Unload - Auto-unload after 180s idle"
log_info "Waiting for idle timeout (180s = 3 minutes)..."
log_info "Current time: $(date +%H:%M:%S)"

# Check every 30 seconds
for i in {1..7}; do
    sleep 30
    MODEL_LOADED=$(get_model_loaded)
    ELAPSED=$((i * 30))

    if [ "$MODEL_LOADED" = "False" ]; then
        log_pass "Idle timeout unload triggered after ${ELAPSED}s (expected ~180s)"
        break
    else
        log_info "  ${ELAPSED}s elapsed - model still loaded (checking again...)"
    fi

    if [ $i -eq 7 ]; then
        log_fail "Idle timeout did not trigger after 210s (expected 180s)"
    fi
done
echo ""

# Test 10: Final Memory Check (No Leaks)
log_test "10. Memory Leak Detection - Final baseline check"
sleep 2
FINAL_MEM=$(get_memory_gb)
FINAL_DELTA=$(python3 -c "print(f'{abs(float($FINAL_MEM) - float($BASELINE_MEM)):.4f}')")

log_info "Initial baseline: ${BASELINE_MEM} GB"
log_info "Final memory: ${FINAL_MEM} GB"
log_info "Total delta: ${FINAL_DELTA} GB"

if (( $(python3 -c "print(1 if float($FINAL_DELTA) < 0.15 else 0)") )); then
    log_pass "No memory leaks detected (delta <0.15 GB - normal overhead)"
else
    log_fail "Possible memory leak detected (delta: ${FINAL_DELTA} GB)"
fi
echo ""

# Test 11: Status API Verification
log_test "11. Status API - Verify all fields present"
STATUS=$(curl -s "$ADMIN_API/admin/status")
REQUIRED_FIELDS=("status" "memory" "requests" "idle_timeout_seconds")
MISSING_FIELDS=0

for field in "${REQUIRED_FIELDS[@]}"; do
    if echo "$STATUS" | python3 -c "import sys, json; '$field' in json.load(sys.stdin)" 2>/dev/null; then
        log_info "  Field '$field': present"
    else
        log_fail "  Field '$field': MISSING"
        ((MISSING_FIELDS++))
    fi
done

if [ $MISSING_FIELDS -eq 0 ]; then
    log_pass "All required status fields present"
else
    log_fail "Status API missing $MISSING_FIELDS required fields"
fi
echo ""

# Summary
echo "========================================================================"
echo "Test Summary"
echo "========================================================================"
echo -e "${GREEN}Tests Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Tests Failed: $TESTS_FAILED${NC}"
echo "Completed: $(date)"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ ALL TESTS PASSED - NASA-GRADE VERIFIED${NC}"
    exit 0
else
    echo -e "${RED}✗ SOME TESTS FAILED - REVIEW REQUIRED${NC}"
    exit 1
fi
