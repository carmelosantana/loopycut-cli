#!/bin/bash
# LoopyCut Test Runner
# Quick test runner for different test scenarios

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo -e "${YELLOW}Activating virtual environment...${NC}"
    source .venv/bin/activate
fi

echo -e "${GREEN}LoopyCut Test Runner${NC}"
echo -e "${GREEN}===================${NC}"

# Function to run a specific test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo -e "\n${BLUE}Running: $test_name${NC}"
    echo -e "${YELLOW}Command: $test_command${NC}"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ $test_name passed${NC}"
        return 0
    else
        echo -e "${RED}✗ $test_name failed${NC}"
        return 1
    fi
}

# Quick functionality tests
echo -e "\n${BLUE}=== QUICK TESTS ===${NC}"

run_test "System Info" "python cli.py --info"
run_test "Help Command" "python cli.py --help"

# Check if test video exists
if [[ -f "test_speed_3x.mp4" ]]; then
    run_test "Basic Loop Creation" "python cli.py test_speed_3x.mp4 test_outputs/quick_test.mp4 --length 2 --similarity 85 --method fast_hash --gpu"
    
    run_test "Multiple Loops Test" "python cli.py test_speed_3x.mp4 test_outputs/quick_multi.mp4 --loop-count 2 --similarity 80 --method fast_hash --gpu"
else
    echo -e "${YELLOW}⚠ test_speed_3x.mp4 not found, skipping video tests${NC}"
fi

# Python unit tests
echo -e "\n${BLUE}=== PYTHON UNIT TESTS ===${NC}"
run_test "Unit Tests" "python test_suite.py --unit --verbose"

echo -e "\n${BLUE}=== INTEGRATION TESTS ===${NC}"
run_test "Integration Tests" "python test_suite.py --integration --verbose"

echo -e "\n${BLUE}=== PERFORMANCE TESTS ===${NC}"
run_test "Performance Tests" "python test_suite.py --performance --verbose"

echo -e "\n${GREEN}All tests completed!${NC}"
