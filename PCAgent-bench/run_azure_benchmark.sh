#!/bin/bash

# Script to run PC-Eval benchmark with Azure Claude Sonnet 4.5

echo "========================================"
echo "PC-Eval Benchmark with Azure Claude 4.5"
echo "========================================"
echo ""

# Check if Azure environment variables are set
if [ -z "$AZURE_ANTHROPIC_DEPLOYMENT" ]; then
    echo "⚠️  Azure Anthropic environment variables not configured!"
    echo ""
    echo "Please set in docker/.env or export directly:"
    echo "  AZURE_ANTHROPIC_ENDPOINT=https://your-resource.cognitiveservices.azure.com"
    echo "  AZURE_ANTHROPIC_API_KEY=your-api-key"
    echo "  AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-45"
    echo ""
    echo "Then restart Docker services:"
    echo "  docker-compose -f docker/docker-compose.yml down"
    echo "  docker-compose -f docker/docker-compose.yml up -d"
    echo ""
    exit 1
fi

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
echo "✅ Azure deployment: $AZURE_ANTHROPIC_DEPLOYMENT"
echo ""

# Build the model configuration for Azure Claude
MODEL_CONFIG="{\"provider\":\"anthropic\",\"name\":\"$AZURE_ANTHROPIC_DEPLOYMENT\",\"title\":\"Azure: $AZURE_ANTHROPIC_DEPLOYMENT\"}"

# Parse command line arguments
case "$1" in
    "test")
        # Test with a single task
        TASK_NUM=${2:-1}
        echo "Running test with task #$TASK_NUM"
        echo "============================="
        python PCAgent-bench/test_pc_eval.py --task-num $TASK_NUM --model "$MODEL_CONFIG"
        ;;

    "quick")
        # Quick benchmark with first 3 tasks
        echo "Running quick benchmark (tasks 1-3)"
        echo "===================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --tasks 1,2,3 \
            --model "$MODEL_CONFIG"
        ;;

    "full")
        # Full benchmark
        echo "Running full PC-Eval benchmark (27 tasks)"
        echo "=========================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --model "$MODEL_CONFIG"
        ;;

    "resume")
        # Resume from specific task
        START_FROM=${2:-1}
        echo "Resuming benchmark from task #$START_FROM"
        echo "=========================================="
        python PCAgent-bench/run_pc_eval_benchmark.py \
            --start-from $START_FROM \
            --model "$MODEL_CONFIG"
        ;;

    *)
        echo "Usage: ./run_azure_benchmark.sh [command] [options]"
        echo ""
        echo "Commands:"
        echo "  test [N]     - Test with single task N (default: 1)"
        echo "  quick        - Run quick benchmark (tasks 1-3)"
        echo "  full         - Run full benchmark (all 27 tasks)"
        echo "  resume [N]   - Resume from task N"
        echo ""
        echo "Examples:"
        echo "  ./run_azure_benchmark.sh test 5      # Run task #5"
        echo "  ./run_azure_benchmark.sh quick       # Run tasks 1-3"
        echo "  ./run_azure_benchmark.sh full        # Run all tasks"
        echo "  ./run_azure_benchmark.sh resume 10   # Resume from task #10"
        ;;
esac