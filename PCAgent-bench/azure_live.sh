#!/bin/bash

# Live monitoring script for PC-Eval with Azure Claude

echo "======================================"
echo "🚀 PC-Eval Live Monitor - Azure Claude"
echo "======================================"
echo ""

# Check if services are running
if ! curl -s http://localhost:9991/tasks?limit=1 > /dev/null 2>&1; then
    echo "⚠️  Bytebot agent not running at http://localhost:9991"
    echo "Please start services: docker-compose -f docker/docker-compose.yml up -d"
    exit 1
fi

# Get Azure model from API
MODELS_JSON=$(curl -s http://localhost:9991/tasks/models)
if echo "$MODELS_JSON" | grep -q '"title":"Azure:'; then
    AZURE_DEPLOYMENT=$(echo "$MODELS_JSON" | grep -o '"name":"[^"]*","title":"Azure:' | sed 's/.*"name":"\([^"]*\)".*/\1/' | head -1)
    echo "✅ Found Azure deployment: $AZURE_DEPLOYMENT"
    MODEL_CONFIG="{\"provider\":\"anthropic\",\"name\":\"$AZURE_DEPLOYMENT\",\"title\":\"Azure: $AZURE_DEPLOYMENT\"}"
else
    echo "❌ No Azure models found. Please configure Azure Anthropic in docker/.env"
    exit 1
fi

echo ""

# Parse command
case "$1" in
    "test")
        TASK_NUM=${2:-1}
        echo "🧪 Testing with task #$TASK_NUM"
        python3 PCAgent-bench/run_pc_eval_live.py \
            --tasks "$TASK_NUM" \
            --model "$MODEL_CONFIG" \
            --verbose
        ;;

    "quick")
        echo "⚡ Quick test (tasks 1-3)"
        python3 PCAgent-bench/run_pc_eval_live.py \
            --tasks "1,2,3" \
            --model "$MODEL_CONFIG" \
            --verbose
        ;;

    "full")
        echo "📊 Full benchmark (all 27 tasks)"
        python3 PCAgent-bench/run_pc_eval_live.py \
            --model "$MODEL_CONFIG"
        ;;

    "resume")
        START=${2:-1}
        echo "⏯️ Resuming from task #$START"
        python3 PCAgent-bench/run_pc_eval_live.py \
            --start-from "$START" \
            --model "$MODEL_CONFIG"
        ;;

    *)
        echo "Usage: $0 [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test [N]    - Test single task N (default: 1) with verbose output"
        echo "  quick       - Run tasks 1-3 with verbose output"
        echo "  full        - Run all 27 tasks"
        echo "  resume [N]  - Resume from task N"
        echo ""
        echo "Examples:"
        echo "  $0 test 5     # Test task 5 with detailed output"
        echo "  $0 quick      # Quick test with first 3 tasks"
        echo "  $0 full       # Complete benchmark"
        echo ""
        echo "The live monitor shows:"
        echo "  • Real-time step-by-step progress"
        echo "  • Tool calls (clicks, typing, screenshots)"
        echo "  • Status changes and timeouts"
        echo "  • Performance statistics"
        ;;
esac