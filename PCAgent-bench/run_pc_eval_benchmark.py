#!/usr/bin/env python3
"""Run PC-Eval benchmark tasks against the bytebot agent API.

This script parses PC-Eval.txt tasks and submits them to Bytebot for execution,
monitoring progress and collecting results for benchmark analysis.

Usage:
    # Run all tasks with Claude Sonnet
    python run_pc_eval_benchmark.py --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'

    # Run specific tasks (by number)
    python run_pc_eval_benchmark.py --tasks 1,2,3 --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'

    # Run with vLLM model
    python run_pc_eval_benchmark.py --vllm-model "microsoft/UI-TARS-1.5-7B"

    # Resume from a specific task
    python run_pc_eval_benchmark.py --start-from 5 --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_HELP", "NEEDS_REVIEW"}
NO_PROGRESS_TIMEOUT = 300  # 5 minutes with no new messages -> cancel
MAX_TASK_DURATION = 600  # 10 minutes max per task


class BytebotClient:
    """HTTP client for the bytebot agent REST API."""

    def __init__(self, base_url: str, max_retries: int = 3):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.max_retries = max_retries

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}{path}"
        backoff = 2
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.request(method, url, timeout=30, **kwargs)
                resp.raise_for_status()
                return resp
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                if attempt < self.max_retries - 1:
                    logger.warning(
                        "Request %s %s failed (attempt %d/%d): %s. Retrying in %ds...",
                        method, path, attempt + 1, self.max_retries, exc, backoff,
                    )
                    time.sleep(backoff)
                    backoff *= 2
            except requests.HTTPError:
                raise
        raise last_exc

    def health_check(self) -> bool:
        try:
            self._request("GET", "/tasks?limit=1")
            return True
        except Exception:
            return False

    def get_models(self) -> list[dict]:
        """GET /tasks/models - returns available models."""
        resp = self._request("GET", "/tasks/models")
        return resp.json()

    def create_task(self, description: str, model: dict) -> dict:
        resp = self._request(
            "POST",
            "/tasks",
            json={"description": description, "model": model},
        )
        return resp.json()

    def get_task(self, task_id: str) -> dict:
        resp = self._request("GET", f"/tasks/{task_id}")
        return resp.json()

    def get_all_messages(self, task_id: str) -> list[dict]:
        all_messages: list[dict] = []
        page = 1
        while True:
            resp = self._request(
                "GET",
                f"/tasks/{task_id}/messages",
                params={"page": page, "limit": 100},
            )
            batch = resp.json()
            if not batch:
                break
            all_messages.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return all_messages

    def cancel_task(self, task_id: str) -> dict:
        try:
            resp = self._request("POST", f"/tasks/{task_id}/cancel")
            return resp.json()
        except requests.HTTPError as exc:
            logger.warning("Failed to cancel task %s: %s", task_id, exc)
            return {}


def parse_pc_eval_tasks(file_path: str) -> Dict[int, str]:
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


def run_single_task(
    client: BytebotClient,
    task_num: int,
    task_desc: str,
    model: dict,
    output_dir: Path,
) -> dict:
    """Run a single PC-Eval task and return results."""

    logger.info(f"Starting task #{task_num}: {task_desc[:100]}...")

    start_time = datetime.now()

    # Create the task
    try:
        task_data = client.create_task(task_desc, model)
        task_id = task_data["id"]
        logger.info(f"Created Bytebot task {task_id} for PC-Eval task #{task_num}")
    except Exception as e:
        logger.error(f"Failed to create task: {e}")
        return {
            "task_num": task_num,
            "description": task_desc,
            "status": "ERROR",
            "error": str(e),
            "duration_seconds": 0,
        }

    # Monitor task progress
    last_message_count = 0
    last_progress_time = time.time()

    while True:
        try:
            task = client.get_task(task_id)
            status = task["status"]

            # Check if task is complete
            if status in TERMINAL_STATUSES:
                logger.info(f"Task #{task_num} completed with status: {status}")
                break

            # Get messages to check progress
            messages = client.get_all_messages(task_id)
            current_message_count = len(messages)

            if current_message_count > last_message_count:
                logger.info(f"Task #{task_num} progress: {current_message_count} messages")
                last_message_count = current_message_count
                last_progress_time = time.time()
            else:
                # Check for timeout
                if time.time() - last_progress_time > NO_PROGRESS_TIMEOUT:
                    logger.warning(f"Task #{task_num} timed out - no progress for {NO_PROGRESS_TIMEOUT}s")
                    client.cancel_task(task_id)
                    status = "TIMEOUT"
                    break

            # Check overall task duration
            if (datetime.now() - start_time).total_seconds() > MAX_TASK_DURATION:
                logger.warning(f"Task #{task_num} exceeded max duration of {MAX_TASK_DURATION}s")
                client.cancel_task(task_id)
                status = "TIMEOUT"
                break

            time.sleep(5)  # Poll every 5 seconds

        except Exception as e:
            logger.error(f"Error monitoring task #{task_num}: {e}")
            status = "ERROR"
            break

    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()

    # Collect final messages for analysis
    try:
        messages = client.get_all_messages(task_id)
        message_count = len(messages)
    except:
        messages = []
        message_count = 0

    # Save task results
    result = {
        "task_num": task_num,
        "description": task_desc,
        "bytebot_task_id": task_id,
        "status": status,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration,
        "message_count": message_count,
        "model": model,
    }

    # Save individual task result
    task_file = output_dir / f"task_{task_num:03d}.json"
    with open(task_file, "w") as f:
        json.dump({
            "result": result,
            "messages": messages[-10:] if messages else [],  # Save last 10 messages
        }, f, indent=2, default=str)

    return result


def main():
    parser = argparse.ArgumentParser(description="Run PC-Eval benchmark with Bytebot")
    parser.add_argument(
        "--model",
        type=str,
        help='Model configuration JSON (e.g., \'{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}\')',
    )
    parser.add_argument(
        "--vllm-model",
        type=str,
        help="Convenience flag for vLLM models (e.g., 'microsoft/UI-TARS-1.5-7B')",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        help="Comma-separated list of task numbers to run (e.g., '1,2,3'). If not specified, runs all tasks.",
    )
    parser.add_argument(
        "--start-from",
        type=int,
        default=1,
        help="Start from a specific task number (useful for resuming)",
    )
    parser.add_argument(
        "--agent-url",
        type=str,
        default=os.getenv("BYTEBOT_AGENT_BASE_URL", "http://localhost:9991"),
        help="Bytebot agent API URL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pc_eval_results",
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Determine model configuration
    if args.vllm_model:
        model = {
            "provider": "proxy",
            "name": args.vllm_model,
            "title": f"vLLM: {args.vllm_model}",
        }
    elif args.model:
        try:
            model = json.loads(args.model)
        except json.JSONDecodeError:
            logger.error("Invalid model JSON")
            sys.exit(1)
    else:
        logger.error("Either --model or --vllm-model must be specified")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}_{model.get('name', 'unknown').replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = BytebotClient(args.agent_url)

    # Check connection
    if not client.health_check():
        logger.error(f"Cannot connect to Bytebot agent at {args.agent_url}")
        sys.exit(1)

    logger.info(f"Connected to Bytebot agent at {args.agent_url}")

    # Parse PC-Eval tasks
    pc_eval_file = Path(__file__).parent / "PC-Eval.txt"
    if not pc_eval_file.exists():
        logger.error(f"PC-Eval.txt not found at {pc_eval_file}")
        sys.exit(1)

    all_tasks = parse_pc_eval_tasks(str(pc_eval_file))
    logger.info(f"Loaded {len(all_tasks)} tasks from PC-Eval.txt")

    # Determine which tasks to run
    if args.tasks:
        task_nums = [int(n.strip()) for n in args.tasks.split(",")]
        tasks_to_run = {n: all_tasks[n] for n in task_nums if n in all_tasks}
    else:
        tasks_to_run = {n: desc for n, desc in all_tasks.items() if n >= args.start_from}

    logger.info(f"Will run {len(tasks_to_run)} tasks")

    # Run tasks
    results = []
    for task_num in sorted(tasks_to_run.keys()):
        task_desc = tasks_to_run[task_num]
        logger.info(f"\n{'='*60}")
        logger.info(f"Task #{task_num}/{max(tasks_to_run.keys())}")
        logger.info(f"{'='*60}")

        result = run_single_task(client, task_num, task_desc, model, run_dir)
        results.append(result)

        # Save cumulative results after each task
        summary_file = run_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "model": model,
                "total_tasks": len(tasks_to_run),
                "completed_tasks": len(results),
                "results": results,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*60)
    print("PC-EVAL BENCHMARK RESULTS")
    print("="*60)
    print(f"Model: {model.get('title', model.get('name'))}")
    print(f"Total tasks: {len(results)}")

    # Count statuses
    status_counts = {}
    for r in results:
        status = r["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    print("\nStatus breakdown:")
    for status, count in sorted(status_counts.items()):
        percentage = (count / len(results)) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")

    # Success rate
    successful = status_counts.get("COMPLETED", 0)
    success_rate = (successful / len(results)) * 100 if results else 0
    print(f"\nSuccess rate: {successful}/{len(results)} ({success_rate:.1f}%)")

    # Average duration
    durations = [r["duration_seconds"] for r in results if r["duration_seconds"] > 0]
    if durations:
        avg_duration = sum(durations) / len(durations)
        print(f"Average task duration: {avg_duration:.1f}s")

    print(f"\nResults saved to: {run_dir}")

    # Generate detailed report
    report_file = run_dir / "report.txt"
    with open(report_file, "w") as f:
        f.write("PC-EVAL BENCHMARK DETAILED REPORT\n")
        f.write("="*60 + "\n")
        f.write(f"Model: {model.get('title', model.get('name'))}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Total tasks: {len(results)}\n")
        f.write(f"Success rate: {success_rate:.1f}%\n")
        f.write("\n" + "="*60 + "\n")
        f.write("INDIVIDUAL TASK RESULTS\n")
        f.write("="*60 + "\n\n")

        for r in results:
            f.write(f"Task #{r['task_num']}: {r['status']}\n")
            f.write(f"Description: {r['description'][:100]}...\n")
            f.write(f"Duration: {r['duration_seconds']:.1f}s\n")
            f.write(f"Messages: {r.get('message_count', 0)}\n")
            f.write("-"*40 + "\n")

    logger.info(f"Detailed report saved to: {report_file}")


if __name__ == "__main__":
    main()