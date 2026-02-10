#!/bin/bash
# Convenience script for running OSWorld benchmark with vLLM models

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}OSWorld Benchmark Runner for vLLM Models${NC}"
echo "========================================="

# Default values
API_URL="${API_URL:-http://localhost:9991}"
MAX_TASKS="${MAX_TASKS:-10}"
MAX_STEPS="${MAX_STEPS:-100}"
DOMAIN="${DOMAIN:-all}"

# Function to show usage
show_usage() {
    echo "Usage: $0 [MODEL_NAME] [OPTIONS]"
    echo ""
    echo "Examples:"
    echo "  $0 meta-llama/Meta-Llama-3.1-70B-Instruct"
    echo "  $0 mistralai/Mistral-7B-Instruct-v0.3 --max-tasks 5"
    echo "  $0 meta-llama/Meta-Llama-3.1-8B-Instruct --domain chrome --resume"
    echo ""
    echo "Available options:"
    echo "  --max-tasks N     Maximum number of tasks to run (default: 10)"
    echo "  --max-steps N     Maximum steps per task (default: 100)"
    echo "  --domain NAME     Filter to specific domain (default: all)"
    echo "  --resume          Resume from previous run"
    echo "  --test            Run only 1 task as a test"
    echo "  --help            Show this help message"
    echo ""
    echo "Environment variables:"
    echo "  API_URL          Bytebot API URL (default: http://localhost:9991)"
    echo "  MAX_TASKS        Default max tasks (default: 10)"
    echo "  MAX_STEPS        Default max steps (default: 100)"
}

# Check if help is requested
if [ "$1" = "--help" ] || [ "$1" = "-h" ] || [ -z "$1" ]; then
    show_usage
    exit 0
fi

# Get model name from first argument
MODEL_NAME="$1"
shift

# Parse additional arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            MAX_TASKS=1
            echo -e "${YELLOW}Test mode: Running only 1 task${NC}"
            shift
            ;;
        --max-tasks)
            MAX_TASKS="$2"
            shift 2
            ;;
        --max-steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --resume)
            EXTRA_ARGS="$EXTRA_ARGS --resume"
            echo -e "${YELLOW}Resume mode: Skipping completed tasks${NC}"
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            show_usage
            exit 1
            ;;
    esac
done

# Check if dataset exists
if [ ! -f "osworld_dataset.json" ]; then
    echo -e "${YELLOW}Dataset not found. Extracting from OSWorld...${NC}"
    python3 extract_data.py
    if [ $? -ne 0 ]; then
        echo -e "${RED}Failed to extract dataset${NC}"
        exit 1
    fi
fi

# Test vLLM integration first
echo -e "\n${YELLOW}Testing vLLM integration...${NC}"
python3 test_vllm.py "$API_URL" > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo -e "${RED}vLLM integration test failed!${NC}"
    echo "Please ensure:"
    echo "1. vLLM server is running"
    echo "2. VLLM_BASE_URL and VLLM_MODEL_NAMES are configured"
    echo "3. bytebot-agent has been restarted"
    echo ""
    echo "Run 'python3 test_vllm.py' for detailed diagnostics"
    exit 1
fi
echo -e "${GREEN}✓ vLLM integration test passed${NC}"

# Display configuration
echo ""
echo "Configuration:"
echo "  Model:     $MODEL_NAME"
echo "  API URL:   $API_URL"
echo "  Max tasks: $MAX_TASKS"
echo "  Max steps: $MAX_STEPS"
echo "  Domain:    $DOMAIN"
echo ""

# Run the benchmark
echo -e "${GREEN}Starting benchmark...${NC}"
echo "----------------------------------------"

python3 run_benchmark.py \
    --vllm-model "$MODEL_NAME" \
    --api-url "$API_URL" \
    --max-tasks "$MAX_TASKS" \
    --max-steps "$MAX_STEPS" \
    --domain "$DOMAIN" \
    $EXTRA_ARGS

EXIT_CODE=$?

# Show completion message
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✓ Benchmark completed successfully!${NC}"
    echo "Results are in: osworld-logs/"
    echo ""
    echo "View results with:"
    echo "  cat osworld-logs/final.json | jq '.summary'"
else
    echo -e "${RED}✗ Benchmark failed with exit code: $EXIT_CODE${NC}"
fi

exit $EXIT_CODE