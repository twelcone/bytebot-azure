#!/usr/bin/env python3
"""Quick test script for running individual PC-Eval tasks.

Usage:
    # Run a specific task by number
    python test_pc_eval.py --task-num 1

    # Run with a custom task description
    python test_pc_eval.py --custom "Open Notepad and write 'Hello World'"

    # List all available tasks
    python test_pc_eval.py --list
"""

import argparse
import json
import re
import sys
from pathlib import Path

import requests


def parse_pc_eval_tasks(file_path: str) -> dict:
    """Parse PC-Eval.txt and extract task descriptions."""
    tasks = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse tasks using regex
    pattern = r'(\d+)→"([^"]+)"'
    matches = re.findall(pattern, content)

    for match in matches:
        task_num = int(match[0])
        task_desc = match[1]
        tasks[task_num] = task_desc

    return tasks


def list_tasks(tasks: dict):
    """Print all available tasks."""
    print("\n" + "="*80)
    print("PC-EVAL TASKS")
    print("="*80)

    for num in sorted(tasks.keys()):
        desc = tasks[num]
        # Truncate long descriptions
        if len(desc) > 70:
            desc = desc[:67] + "..."
        print(f"{num:3d}: {desc}")

    print("="*80)
    print(f"Total: {len(tasks)} tasks\n")


def create_task(base_url: str, description: str, model: dict = None) -> str:
    """Create a task via the Bytebot API."""
    if model is None:
        # Default to Claude Sonnet
        model = {
            "provider": "anthropic",
            "name": "claude-3-5-sonnet-20241022",
            "title": "Claude Sonnet 3.5"
        }

    url = f"{base_url}/tasks"
    payload = {
        "description": description,
        "model": model
    }

    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        task_data = response.json()
        return task_data["id"]
    except Exception as e:
        print(f"Error creating task: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Test PC-Eval tasks with Bytebot")
    parser.add_argument(
        "--task-num",
        type=int,
        help="Run a specific task by number (1-27)"
    )
    parser.add_argument(
        "--custom",
        type=str,
        help="Run a custom task description"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available PC-Eval tasks"
    )
    parser.add_argument(
        "--model",
        type=str,
        help='Model configuration JSON (optional)',
        default=None
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default="http://localhost:9991",
        help="Bytebot agent API URL"
    )

    args = parser.parse_args()

    # Parse PC-Eval tasks
    pc_eval_file = Path(__file__).parent / "PC-Eval.txt"
    if not pc_eval_file.exists():
        print(f"Error: PC-Eval.txt not found at {pc_eval_file}")
        sys.exit(1)

    tasks = parse_pc_eval_tasks(str(pc_eval_file))

    # Handle list command
    if args.list:
        list_tasks(tasks)
        return

    # Determine task description
    if args.task_num:
        if args.task_num not in tasks:
            print(f"Error: Task #{args.task_num} not found. Valid range: 1-{max(tasks.keys())}")
            sys.exit(1)
        description = tasks[args.task_num]
        print(f"\nRunning PC-Eval Task #{args.task_num}:")
    elif args.custom:
        description = args.custom
        print(f"\nRunning custom task:")
    else:
        print("Error: Specify either --task-num, --custom, or --list")
        sys.exit(1)

    print(f"Description: {description}\n")

    # Parse model if provided
    model = None
    if args.model:
        try:
            model = json.loads(args.model)
        except json.JSONDecodeError:
            print("Error: Invalid model JSON")
            sys.exit(1)

    # Create the task
    print(f"Creating task on Bytebot agent at {args.agent_url}...")
    task_id = create_task(args.agent_url, description, model)

    if task_id:
        print(f"✅ Task created successfully!")
        print(f"Task ID: {task_id}")
        print(f"\nYou can monitor the task at:")
        print(f"  Web UI: http://localhost:9992")
        print(f"  API: {args.agent_url}/tasks/{task_id}")
        print(f"\nTo view messages:")
        print(f"  curl {args.agent_url}/tasks/{task_id}/messages")
    else:
        print("❌ Failed to create task")
        sys.exit(1)


if __name__ == "__main__":
    main()