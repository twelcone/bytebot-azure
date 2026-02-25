#!/bin/bash

# Script to run PC-Eval benchmark with Azure Claude Sonnet 4.5

echo "========================================"
echo "PC-Eval Benchmark with Azure Claude 4.5"
echo "========================================"
echo ""

# Check if services are running
if ! curl -s http://localhost:9991/tasks?limit=1 > /dev/null 2>&1; then
    echo "⚠️  Bytebot agent not running at http://localhost:9991"
    echo ""
    echo "Please start the services:"
    echo "  docker-compose -f docker/docker-compose.yml up -d"
    echo ""
    exit 1
fi

echo "✅ Bytebot agent is running"
echo ""

# Check for Azure models in the running service
echo "Checking for Azure models..."
MODELS_JSON=$(curl -s http://localhost:9991/tasks/models)

# Check if any Azure models are available
if echo "$MODELS_JSON" | grep -q '"title":"Azure:'; then
    # Extract the Azure deployment name
    AZURE_DEPLOYMENT=$(echo "$MODELS_JSON" | grep -o '"name":"[^"]*","title":"Azure:' | sed 's/.*"name":"\([^"]*\)".*/\1/' | head -1)
    echo "✅ Found Azure deployment: $AZURE_DEPLOYMENT"
    echo ""
else
    echo "⚠️  No Azure models found in Bytebot!"
    echo ""
    echo "To enable Azure Claude Sonnet 4.5:"
    echo ""
    echo "1. Edit docker/.env and add:"
    echo "   AZURE_ANTHROPIC_ENDPOINT=https://your-resource.cognitiveservices.azure.com"
    echo "   AZURE_ANTHROPIC_API_KEY=your-api-key"
    echo "   AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-45"
    echo ""
    echo "2. Restart Docker services:"
    echo "   docker-compose -f docker/docker-compose.yml down"
    echo "   docker-compose -f docker/docker-compose.yml up -d"
    echo ""
    echo "3. Run this script again"
    echo ""
    echo "Available models:"
    python PCAgent-bench/check_models.py
    exit 1
fi

# Build the model configuration for Azure Claude
MODEL_CONFIG="{\"provider\":\"anthropic\",\"name\":\"$AZURE_DEPLOYMENT\",\"title\":\"Azure: $AZURE_DEPLOYMENT\"}"

# Parse command line arguments
case "$1" in
    "test")
        # Test with a single task
        TASK_NUM=${2:-1}
        echo "Running test with task #$TASK_NUM"
        echo "Model: Azure $AZURE_DEPLOYMENT"
        echo "============================="
        python PCAgent-bench/test_pc_eval.py --task-num $TASK_NUM --model "$MODEL_CONFIG"
        ;;

    "quick")
        # Quick benchmark with first 3 tasks
        echo "Running quick benchmark (tasks 1-3)"
        echo "Model: Azure $AZURE_DEPLOYMENT"
        echo "===================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --tasks 1,2,3 \
            --model "$MODEL_CONFIG"
        ;;

    "full")
        # Full benchmark
        echo "Running full PC-Eval benchmark (27 tasks)"
        echo "Model: Azure $AZURE_DEPLOYMENT"
        echo "=========================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --model "$MODEL_CONFIG"
        ;;

    "resume")
        # Resume from specific task
        START_FROM=${2:-1}
        echo "Resuming benchmark from task #$START_FROM"
        echo "Model: Azure $AZURE_DEPLOYMENT"
        echo "=========================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --start-from $START_FROM \
            --model "$MODEL_CONFIG"
        ;;

    "list")
        # List all tasks
        echo "PC-Eval Tasks:"
        echo "=============="
        python PCAgent-bench/test_pc_eval.py --list
        ;;

    *)
        echo "Usage: ./run_azure_benchmark.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test [N]     - Test with single task N (default: 1)"
        echo "  quick        - Run quick benchmark (tasks 1-3)"
        echo "  full         - Run full benchmark (all 27 tasks)"
        echo "  resume [N]   - Resume from task N"
        echo "  list         - List all available tasks"
        echo ""
        echo "Examples:"
        echo "  ./run_azure_benchmark.sh test 5      # Run task #5"
        echo "  ./run_azure_benchmark.sh quick       # Run tasks 1-3"
        echo "  ./run_azure_benchmark.sh full        # Run all tasks"
        echo "  ./run_azure_benchmark.sh resume 10   # Resume from task #10"
        echo ""
        echo "Current configuration:"
        echo "  Azure deployment: $AZURE_DEPLOYMENT"
        ;;
esac