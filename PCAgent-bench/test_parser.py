#!/usr/bin/env python3
"""Test the PC-Eval parser to verify it works correctly."""

import re
from pathlib import Path

def parse_pc_eval_tasks(file_path: str) -> dict:
    """Parse PC-Eval.txt and extract task descriptions."""
    tasks = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Parse tasks - they are quoted strings separated by commas
    # Remove trailing comma and newline if present
    content = content.strip()
    if content.endswith(','):
        content = content[:-1]

    # Split by pattern that matches quoted strings
    # This handles multi-line task descriptions
    task_strings = re.findall(r'"([^"]+)"', content)

    # Assign numbers starting from 1
    for i, task_desc in enumerate(task_strings, start=1):
        if task_desc.strip():  # Skip empty tasks
            tasks[i] = task_desc.strip()

    return tasks

# Test the parser
pc_eval_file = Path(__file__).parent / "PC-Eval.txt"
tasks = parse_pc_eval_tasks(str(pc_eval_file))

print(f"Found {len(tasks)} tasks")
print("\nFirst 3 tasks:")
for i in range(1, min(4, len(tasks) + 1)):
    if i in tasks:
        desc = tasks[i]
        if len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"{i}: {desc}")

print(f"\nTask numbers: {sorted(tasks.keys())[:10]}..." if len(tasks) > 10 else f"\nTask numbers: {sorted(tasks.keys())}")