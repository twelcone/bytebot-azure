#!/usr/bin/env python3
"""Enhanced PC-Eval benchmark runner with real-time monitoring and detailed logging.

This provides OSWorld-style real-time status updates, progress tracking, and detailed logs.

Usage:
    # Run with live monitoring
    python run_pc_eval_live.py --model '{"provider":"anthropic","name":"claude-sonnet-45","title":"Azure: Claude Sonnet 4.5"}'

    # Run specific tasks with verbose output
    python run_pc_eval_live.py --tasks 1,2,3 --verbose --model '...'
"""

import argparse
import json
import logging
import os
import re
import sys
import time
import base64
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import requests
from dataclasses import dataclass, asdict

# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for different log levels."""

    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        # Add color to the level name
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{self.BOLD}{record.levelname}{self.RESET}"

        # Format the message
        message = super().format(record)

        # Special formatting for certain message types
        if "Step" in record.getMessage():
            message = f"{self.BOLD}{message}{self.RESET}"
        elif "Task #" in record.getMessage() and "completed" in record.getMessage():
            message = f"\n{self.BOLD}{'='*60}{self.RESET}\n{message}\n{self.BOLD}{'='*60}{self.RESET}"

        return message

# Set up logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_HELP", "NEEDS_REVIEW"}
NO_PROGRESS_TIMEOUT = 300  # 5 minutes
MAX_TASK_DURATION = 600   # 10 minutes

@dataclass
class StepData:
    """Data for a single agent step."""
    step_number: int
    timestamp: str
    message_id: str
    tool_calls: list
    text_content: str
    has_screenshot: bool = False
    screenshot_file: Optional[str] = None

@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_num: int
    description: str
    bytebot_task_id: str
    status: str
    final_api_status: str
    total_steps: int
    start_time: str
    end_time: str
    duration_seconds: float
    error_message: Optional[str] = None

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
                    time.sleep(backoff)
                    backoff *= 2
            except requests.HTTPError:
                raise
        raise last_exc

    def create_task(self, description: str, model: dict) -> dict:
        resp = self._request("POST", "/tasks", json={"description": description, "model": model})
        return resp.json()

    def get_task(self, task_id: str) -> dict:
        resp = self._request("GET", f"/tasks/{task_id}")
        return resp.json()

    def get_all_messages(self, task_id: str) -> list:
        all_messages = []
        page = 1
        while True:
            resp = self._request("GET", f"/tasks/{task_id}/messages", params={"page": page, "limit": 100})
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
        except:
            return {}


def parse_pc_eval_tasks(file_path: str) -> Dict[int, str]:
    """Parse PC-Eval.txt and extract task descriptions."""
    tasks = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    content = content.strip()
    if content.endswith(','):
        content = content[:-1]

    task_strings = re.findall(r'"([^"]+)"', content)
    for i, task_desc in enumerate(task_strings, start=1):
        if task_desc.strip():
            tasks[i] = task_desc.strip()

    return tasks


def extract_tool_calls(content_blocks: list) -> list:
    """Extract tool calls from message content blocks."""
    tool_calls = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            tool_calls.append({
                "name": block.get("name", "unknown"),
                "input": block.get("input", {})
            })
    return tool_calls


def extract_text_content(content_blocks: list) -> str:
    """Extract text from message content blocks."""
    texts = []
    for block in content_blocks:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)


def find_screenshot_in_messages(messages: list, assistant_msg: dict) -> Optional[str]:
    """Find screenshot from USER message following the assistant message."""
    assistant_created = assistant_msg.get("createdAt", "")

    # Find the next USER message after this assistant message
    for msg in messages:
        if msg.get("role") == "USER" and msg.get("createdAt", "") > assistant_created:
            content = msg.get("content", [])
            for block in content:
                if block.get("type") == "image":
                    source = block.get("source", {})
                    if source.get("type") == "base64":
                        return source.get("data")
    return None


def save_screenshot(output_dir: Path, task_num: int, step_num: int, base64_data: str) -> str:
    """Save screenshot to file."""
    screenshots_dir = output_dir / f"task_{task_num:03d}" / "screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    filepath = screenshots_dir / f"step_{step_num:03d}.png"

    image_bytes = base64.b64decode(base64_data)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    return str(filepath)


def format_tool_call(tool_call: dict) -> str:
    """Format a tool call for display."""
    name = tool_call.get("name", "unknown")
    input_data = tool_call.get("input", {})

    # Special formatting for common tools
    if name == "computer":
        action = input_data.get("action", "")
        if action == "screenshot":
            return "📸 Screenshot"
        elif action == "click":
            coords = input_data.get("coordinate", [])
            return f"🖱️ Click at ({coords[0] if len(coords) > 0 else '?'}, {coords[1] if len(coords) > 1 else '?'})"
        elif action == "type":
            text = input_data.get("text", "")[:30]
            return f"⌨️ Type: '{text}...'" if len(text) == 30 else f"⌨️ Type: '{text}'"
        elif action == "key":
            key = input_data.get("text", "")
            return f"⌨️ Key: {key}"

    return f"🔧 {name}"


def run_single_task_live(
    client: BytebotClient,
    task_num: int,
    task_desc: str,
    model: dict,
    output_dir: Path,
    verbose: bool = False,
    poll_interval: float = 2.0,
) -> TaskResult:
    """Run a single PC-Eval task with live monitoring."""

    logger.info(f"\n{'='*70}")
    logger.info(f"📋 Task #{task_num}/{27}")
    logger.info(f"{'='*70}")
    logger.info(f"Description: {task_desc[:100]}{'...' if len(task_desc) > 100 else ''}")

    start_time = datetime.now(timezone.utc)
    start_time_str = start_time.isoformat()

    # Create task
    try:
        logger.info("🚀 Creating task...")
        task_data = client.create_task(task_desc, model)
        task_id = task_data["id"]
        logger.info(f"✅ Task created: {task_id}")
    except Exception as e:
        logger.error(f"❌ Failed to create task: {e}")
        return TaskResult(
            task_num=task_num,
            description=task_desc,
            bytebot_task_id="",
            status="ERROR",
            final_api_status="CREATE_FAILED",
            total_steps=0,
            start_time=start_time_str,
            end_time=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0,
            error_message=str(e)
        )

    # Monitor task progress
    processed_msg_ids = set()
    step_count = 0
    last_progress_time = time.time()
    last_status = ""
    spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spinner_idx = 0

    # Create output directory for this task
    task_output_dir = output_dir / f"task_{task_num:03d}"
    task_output_dir.mkdir(parents=True, exist_ok=True)

    while True:
        # Show spinner
        print(f"\r{spinner_chars[spinner_idx % len(spinner_chars)]} Monitoring task... (Step {step_count})", end='', flush=True)
        spinner_idx += 1

        # Get task status
        try:
            task = client.get_task(task_id)
            status = task["status"]

            if status != last_status:
                print(f"\r✨ Status changed: {last_status} → {status}                    ")
                last_status = status

        except Exception as e:
            logger.warning(f"Failed to get task status: {e}")
            status = "UNKNOWN"

        # Check if task is complete
        if status in TERMINAL_STATUSES:
            print(f"\r                                                      ", end='\r')
            logger.info(f"🏁 Task completed with status: {status}")
            break

        # Get and process new messages
        try:
            messages = client.get_all_messages(task_id)
            messages.sort(key=lambda m: m.get("createdAt", ""))

            # Process assistant messages
            assistant_msgs = [m for m in messages if m.get("role") == "ASSISTANT"]
            new_msgs = [m for m in assistant_msgs if m.get("id") not in processed_msg_ids]

            if new_msgs:
                last_progress_time = time.time()
                print(f"\r                                                      ", end='\r')

                for msg in new_msgs:
                    step_count += 1
                    msg_id = msg.get("id", "")
                    processed_msg_ids.add(msg_id)
                    content = msg.get("content", [])

                    # Extract information
                    tool_calls = extract_tool_calls(content)
                    text = extract_text_content(content)
                    screenshot_b64 = find_screenshot_in_messages(messages, msg)

                    # Display step information
                    logger.info(f"\n  📍 Step {step_count}:")

                    if text and verbose:
                        # Show first 150 chars of text
                        text_preview = text[:150].replace('\n', ' ')
                        if len(text) > 150:
                            text_preview += "..."
                        logger.info(f"     💬 {text_preview}")

                    if tool_calls:
                        for tc in tool_calls:
                            logger.info(f"     {format_tool_call(tc)}")

                    # Save screenshot if available
                    if screenshot_b64:
                        screenshot_path = save_screenshot(output_dir, task_num, step_count, screenshot_b64)
                        logger.info(f"     📸 Screenshot saved: {Path(screenshot_path).name}")

                    # Save step data
                    step_data = StepData(
                        step_number=step_count,
                        timestamp=msg.get("createdAt", ""),
                        message_id=msg_id,
                        tool_calls=tool_calls,
                        text_content=text,
                        has_screenshot=bool(screenshot_b64),
                        screenshot_file=screenshot_path if screenshot_b64 else None
                    )

                    # Save step JSON
                    step_file = task_output_dir / f"step_{step_count:03d}.json"
                    with open(step_file, "w") as f:
                        json.dump(asdict(step_data), f, indent=2)

        except Exception as e:
            logger.debug(f"Error processing messages: {e}")

        # Check for timeout
        elapsed = time.time() - start_time.timestamp()
        if elapsed > MAX_TASK_DURATION:
            print(f"\r                                                      ", end='\r')
            logger.warning(f"⏱️ Task exceeded {MAX_TASK_DURATION}s limit. Cancelling...")
            client.cancel_task(task_id)
            status = "TIMEOUT"
            break

        if time.time() - last_progress_time > NO_PROGRESS_TIMEOUT:
            print(f"\r                                                      ", end='\r')
            logger.warning(f"⏱️ No progress for {NO_PROGRESS_TIMEOUT}s. Cancelling...")
            client.cancel_task(task_id)
            status = "TIMEOUT"
            break

        time.sleep(poll_interval)

    # Calculate final metrics
    end_time = datetime.now(timezone.utc)
    duration = (end_time - start_time).total_seconds()

    # Determine final status
    if status == "COMPLETED":
        result_status = "SUCCESS"
        logger.info(f"✅ Task #{task_num} completed successfully in {duration:.1f}s with {step_count} steps")
    elif status == "TIMEOUT":
        result_status = "TIMEOUT"
        logger.warning(f"⏱️ Task #{task_num} timed out after {duration:.1f}s with {step_count} steps")
    elif status in ["FAILED", "CANCELLED"]:
        result_status = status
        logger.warning(f"❌ Task #{task_num} {status.lower()} after {duration:.1f}s with {step_count} steps")
    else:
        result_status = "UNKNOWN"
        logger.error(f"❓ Task #{task_num} ended with unknown status after {duration:.1f}s")

    # Create result
    result = TaskResult(
        task_num=task_num,
        description=task_desc,
        bytebot_task_id=task_id,
        status=result_status,
        final_api_status=status,
        total_steps=step_count,
        start_time=start_time_str,
        end_time=end_time.isoformat(),
        duration_seconds=round(duration, 1),
    )

    # Save result
    result_file = task_output_dir / "result.json"
    with open(result_file, "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def print_final_summary(results: List[TaskResult], output_dir: Path):
    """Print final benchmark summary with statistics."""

    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)

    total = len(results)
    successful = sum(1 for r in results if r.status == "SUCCESS")
    failed = sum(1 for r in results if r.status == "FAILED")
    timeout = sum(1 for r in results if r.status == "TIMEOUT")
    other = total - successful - failed - timeout

    # Overall stats
    success_rate = (successful / total * 100) if total > 0 else 0
    print(f"\n📈 Overall Performance:")
    print(f"   Total Tasks: {total}")
    print(f"   ✅ Successful: {successful} ({success_rate:.1f}%)")
    print(f"   ❌ Failed: {failed}")
    print(f"   ⏱️ Timeout: {timeout}")
    if other > 0:
        print(f"   ❓ Other: {other}")

    # Time stats
    durations = [r.duration_seconds for r in results]
    if durations:
        avg_duration = sum(durations) / len(durations)
        max_duration = max(durations)
        min_duration = min(durations)
        print(f"\n⏱️ Time Statistics:")
        print(f"   Average: {avg_duration:.1f}s")
        print(f"   Fastest: {min_duration:.1f}s")
        print(f"   Slowest: {max_duration:.1f}s")

    # Step stats
    step_counts = [r.total_steps for r in results]
    if step_counts:
        avg_steps = sum(step_counts) / len(step_counts)
        max_steps = max(step_counts)
        min_steps = min(step_counts)
        print(f"\n👣 Step Statistics:")
        print(f"   Average: {avg_steps:.1f} steps")
        print(f"   Min: {min_steps} steps")
        print(f"   Max: {max_steps} steps")

    # Task details
    print(f"\n📝 Task Details:")
    for r in results:
        status_icon = "✅" if r.status == "SUCCESS" else "❌" if r.status == "FAILED" else "⏱️" if r.status == "TIMEOUT" else "❓"
        print(f"   Task #{r.task_num}: {status_icon} {r.status} ({r.duration_seconds:.1f}s, {r.total_steps} steps)")

    print(f"\n💾 Results saved to: {output_dir.absolute()}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run PC-Eval benchmark with live monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--model", type=str, required=True, help="Model configuration JSON")
    parser.add_argument("--tasks", type=str, help="Comma-separated task numbers (e.g., '1,2,3')")
    parser.add_argument("--start-from", type=int, default=1, help="Start from task number")
    parser.add_argument("--agent-url", type=str, default="http://localhost:9991", help="Bytebot agent URL")
    parser.add_argument("--output-dir", type=str, default="pc_eval_live_results", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Show detailed agent output")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="Polling interval in seconds")

    args = parser.parse_args()

    # Parse model
    try:
        model = json.loads(args.model)
    except json.JSONDecodeError:
        logger.error("Invalid model JSON")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = model.get("name", "unknown").replace("/", "_")
    run_dir = output_dir / f"run_{timestamp}_{model_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Initialize client
    client = BytebotClient(args.agent_url)

    # Parse tasks
    pc_eval_file = Path(__file__).parent / "PC-Eval.txt"
    all_tasks = parse_pc_eval_tasks(str(pc_eval_file))

    # Determine which tasks to run
    if args.tasks:
        task_nums = [int(n.strip()) for n in args.tasks.split(",")]
        tasks_to_run = {n: all_tasks[n] for n in task_nums if n in all_tasks}
    else:
        tasks_to_run = {n: desc for n, desc in all_tasks.items() if n >= args.start_from}

    logger.info(f"🎯 Running {len(tasks_to_run)} tasks with model: {model.get('title', model.get('name'))}")
    logger.info(f"💾 Output directory: {run_dir}")

    # Run tasks
    results = []
    for task_num in sorted(tasks_to_run.keys()):
        task_desc = tasks_to_run[task_num]
        result = run_single_task_live(
            client, task_num, task_desc, model, run_dir,
            verbose=args.verbose, poll_interval=args.poll_interval
        )
        results.append(result)

        # Save cumulative results
        summary_file = run_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "model": model,
                "total_tasks": len(tasks_to_run),
                "completed_tasks": len(results),
                "results": [asdict(r) for r in results],
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

    # Print final summary
    print_final_summary(results, run_dir)


if __name__ == "__main__":
    main()