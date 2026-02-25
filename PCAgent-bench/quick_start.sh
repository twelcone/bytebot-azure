#!/bin/bash

# Quick start script for PC-Eval benchmark
echo "===================="
echo "PC-Eval Quick Start"
echo "===================="

# Check if services are running
if ! curl -s http://localhost:9991/tasks?limit=1 > /dev/null 2>&1; then
    echo "⚠️  Bytebot agent not running at http://localhost:9991"
    echo ""
    echo "Please start the services first:"
    echo "  docker-compose -f docker/docker-compose.yml up -d"
    echo ""
    exit 1
fi

echo "✅ Bytebot agent is running"
echo ""

# Check for model configuration
if [ -z "$1" ]; then
    echo "Usage: ./quick_start.sh [task_number]"
    echo ""
    echo "Examples:"
    echo "  ./quick_start.sh 1        # Run task #1"
    echo "  ./quick_start.sh list     # List all tasks"
    echo ""
    echo "Available tasks:"
    python test_pc_eval.py --list | head -20
    echo "..."
    exit 0
fi

if [ "$1" = "list" ]; then
    python test_pc_eval.py --list
else
    echo "Running PC-Eval Task #$1"
    echo "========================"
    python test_pc_eval.py --task-num $1
fi