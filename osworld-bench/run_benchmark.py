#!/usr/bin/env python3
"""Run OSWorld benchmark tasks against the bytebot agent API.

Submits tasks sequentially to the bytebot API, monitors progress via polling,
captures screenshots and step data, and produces structured logs with aggregate
statistics.

Usage:
    # Using Anthropic models
    python run_benchmark.py --model '{"provider":"anthropic","name":"claude-sonnet-4-20250514","title":"Claude Sonnet 4"}'

    # Using vLLM models (requires VLLM_BASE_URL configured)
    python run_benchmark.py --model '{"provider":"proxy","name":"meta-llama/Meta-Llama-3.1-70B-Instruct","title":"vLLM: Llama 3.1 70B"}'
    python run_benchmark.py --vllm-model "meta-llama/Meta-Llama-3.1-70B-Instruct"

    # Other examples
    python run_benchmark.py --max-tasks 5 --domain chrome --model '...'
    python run_benchmark.py --resume --model '...'
"""

import argparse
import base64
import dataclasses
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

TERMINAL_STATUSES = {"COMPLETED", "FAILED", "CANCELLED", "NEEDS_HELP", "NEEDS_REVIEW"}
NO_PROGRESS_TIMEOUT = 300  # 5 minutes with no new messages -> cancel


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class StepData:
    step_number: int
    timestamp: str
    message_id: str
    tool_calls: list[dict]
    text_content: str
    has_screenshot: bool
    screenshot_file: str | None


@dataclasses.dataclass
class TaskResult:
    task_id: str
    domain: str
    instruction: str
    bytebot_task_id: str
    status: str  # "success" / "failed" / "cancelled" / "error"
    final_api_status: str
    total_steps: int
    start_time: str
    end_time: str
    duration_seconds: float
    error_message: str | None = None


# ---------------------------------------------------------------------------
# API Client
# ---------------------------------------------------------------------------


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
        raise last_exc  # type: ignore[misc]

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

    def wait_for_idle(self, poll_interval: float = 3.0, timeout: float = 300):
        """Wait until no tasks are RUNNING or PENDING."""
        start = time.time()
        while time.time() - start < timeout:
            try:
                resp = self._request(
                    "GET",
                    "/tasks",
                    params={"statuses": "RUNNING,PENDING", "limit": 1},
                )
                data = resp.json()
                tasks = data.get("tasks", data) if isinstance(data, dict) else data
                if not tasks or (isinstance(data, dict) and data.get("total", 0) == 0):
                    return
            except Exception:
                pass
            time.sleep(poll_interval)
        logger.warning("Timed out waiting for idle state after %.0fs", timeout)


# ---------------------------------------------------------------------------
# Message parsing
# ---------------------------------------------------------------------------


def extract_assistant_steps(messages: list[dict]) -> list[dict]:
    """Return assistant messages in chronological order."""
    assistant_msgs = [m for m in messages if m.get("role") == "ASSISTANT"]
    # Sort by createdAt to ensure chronological order
    assistant_msgs.sort(key=lambda m: m.get("createdAt", ""))
    return assistant_msgs


def find_screenshot_for_step(
    messages: list[dict], assistant_msg: dict
) -> str | None:
    """Find the screenshot (base64 PNG) from the USER message following the assistant msg.

    Returns the base64 data string or None.
    """
    # Find index of assistant message
    msg_time = assistant_msg.get("createdAt", "")
    msg_id = assistant_msg.get("id", "")

    # Look for the next USER message after this assistant message
    found_assistant = False
    for m in messages:
        if m.get("id") == msg_id:
            found_assistant = True
            continue
        if found_assistant and m.get("role") == "USER":
            return _extract_last_image(m.get("content", []))
        if found_assistant and m.get("role") == "ASSISTANT":
            # Hit another assistant message without finding a user message
            break
    return None


def _extract_last_image(content_blocks: list[dict]) -> str | None:
    """Walk content blocks and return the last base64 image data found.

    Screenshots can be nested inside tool_result content arrays or at the top level.
    """
    last_image = None

    for block in content_blocks:
        if block.get("type") == "image":
            source = block.get("source", {})
            if source.get("type") == "base64":
                last_image = source.get("data")

        # tool_result blocks contain nested content arrays
        if block.get("type") == "tool_result":
            nested = block.get("content", [])
            if isinstance(nested, list):
                img = _extract_last_image(nested)
                if img:
                    last_image = img

    return last_image


def parse_tool_calls(content_blocks: list[dict]) -> list[dict]:
    """Extract tool_use blocks from message content."""
    calls = []
    for block in content_blocks:
        if block.get("type") == "tool_use":
            calls.append({
                "name": block.get("name", ""),
                "input": block.get("input", {}),
            })
    return calls


def extract_text_content(content_blocks: list[dict]) -> str:
    """Extract text from message content blocks."""
    texts = []
    for block in content_blocks:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
    return "\n".join(texts)


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_screenshot(output_dir: str, task_id: str, step_number: int, base64_data: str):
    screenshots_dir = os.path.join(output_dir, task_id, "screenshots")
    ensure_dir(screenshots_dir)
    filepath = os.path.join(screenshots_dir, f"step_{step_number:03d}.png")
    image_bytes = base64.b64decode(base64_data)
    with open(filepath, "wb") as f:
        f.write(image_bytes)
    return filepath


def save_step_json(output_dir: str, task_id: str, step: StepData):
    task_dir = os.path.join(output_dir, task_id)
    ensure_dir(task_dir)
    filepath = os.path.join(task_dir, f"step_{step.step_number:03d}.json")
    data = {
        "step_number": step.step_number,
        "timestamp": step.timestamp,
        "message_id": step.message_id,
        "tool_calls": step.tool_calls,
        "text_content": step.text_content,
        "has_screenshot": step.has_screenshot,
        "screenshot_file": step.screenshot_file,
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def save_result_json(output_dir: str, result: TaskResult):
    task_dir = os.path.join(output_dir, result.task_id)
    ensure_dir(task_dir)
    filepath = os.path.join(task_dir, "result.json")
    data = dataclasses.asdict(result)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def compute_final_stats(
    results: list[TaskResult],
    config: dict,
    benchmark_start: str,
    benchmark_end: str,
) -> dict:
    total = len(results)
    completed = sum(1 for r in results if r.status == "success")
    failed = sum(1 for r in results if r.status == "failed")
    cancelled = sum(1 for r in results if r.status == "cancelled")
    errors = sum(1 for r in results if r.status == "error")

    all_steps = [r.total_steps for r in results if r.status != "error"]
    all_durations = [r.duration_seconds for r in results if r.status != "error"]

    # Per-domain stats
    per_domain: dict[str, dict[str, Any]] = {}
    for r in results:
        d = per_domain.setdefault(r.domain, {"total": 0, "completed": 0, "steps": []})
        d["total"] += 1
        if r.status == "success":
            d["completed"] += 1
        if r.status != "error":
            d["steps"].append(r.total_steps)

    per_domain_final = {}
    for domain, stats in sorted(per_domain.items()):
        per_domain_final[domain] = {
            "total": stats["total"],
            "completed": stats["completed"],
            "success_rate": round(stats["completed"] / stats["total"], 4) if stats["total"] else 0,
            "avg_steps": round(sum(stats["steps"]) / len(stats["steps"]), 1) if stats["steps"] else 0,
        }

    # Find extremes
    max_steps_task = max(results, key=lambda r: r.total_steps) if results else None
    min_steps_task = min(
        (r for r in results if r.status != "error"),
        key=lambda r: r.total_steps,
        default=None,
    )
    longest_task = max(
        (r for r in results if r.status != "error"),
        key=lambda r: r.duration_seconds,
        default=None,
    )

    start_dt = datetime.fromisoformat(benchmark_start)
    end_dt = datetime.fromisoformat(benchmark_end)

    return {
        "benchmark_config": config,
        "summary": {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "errors": errors,
            "success_rate": round(completed / total, 4) if total else 0,
            "avg_steps": round(sum(all_steps) / len(all_steps), 1) if all_steps else 0,
            "avg_duration_seconds": round(sum(all_durations) / len(all_durations), 1) if all_durations else 0,
            "max_steps_task": (
                {"task_id": max_steps_task.task_id, "steps": max_steps_task.total_steps}
                if max_steps_task
                else None
            ),
            "min_steps_task": (
                {"task_id": min_steps_task.task_id, "steps": min_steps_task.total_steps}
                if min_steps_task
                else None
            ),
            "longest_task": (
                {"task_id": longest_task.task_id, "duration_seconds": longest_task.duration_seconds}
                if longest_task
                else None
            ),
        },
        "per_domain": per_domain_final,
        "started_at": benchmark_start,
        "finished_at": benchmark_end,
        "total_duration_seconds": round((end_dt - start_dt).total_seconds(), 1),
        "tasks": [
            {
                "task_id": r.task_id,
                "domain": r.domain,
                "status": r.status,
                "total_steps": r.total_steps,
                "duration_seconds": r.duration_seconds,
            }
            for r in results
        ],
    }


def save_final_json(output_dir: str, final_data: dict):
    filepath = os.path.join(output_dir, "final.json")
    with open(filepath, "w") as f:
        json.dump(final_data, f, indent=2)
    logger.info("Final results written to %s", filepath)


def task_already_completed(output_dir: str, task_id: str) -> bool:
    return os.path.exists(os.path.join(output_dir, task_id, "result.json"))


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------


def run_single_task(
    client: BytebotClient,
    task: dict,
    model: dict,
    output_dir: str,
    max_steps: int,
    poll_interval: float,
) -> TaskResult:
    """Run a single OSWorld task through bytebot and collect results."""

    task_id = task["task_id"]
    domain = task["domain"]
    instruction = task["instruction"]

    # Prepare output directory
    task_dir = os.path.join(output_dir, task_id)
    ensure_dir(os.path.join(task_dir, "screenshots"))

    start_time = datetime.now(timezone.utc).isoformat()

    # Create the task via API
    try:
        api_task = client.create_task(instruction, model)
        bytebot_task_id = api_task["id"]
        logger.info("Created bytebot task %s for OSWorld task %s", bytebot_task_id, task_id)
    except Exception as exc:
        end_time = datetime.now(timezone.utc).isoformat()
        logger.error("Failed to create task %s: %s", task_id, exc)
        return TaskResult(
            task_id=task_id,
            domain=domain,
            instruction=instruction,
            bytebot_task_id="",
            status="error",
            final_api_status="CREATE_FAILED",
            total_steps=0,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=0,
            error_message=str(exc),
        )

    # Polling loop
    processed_msg_ids: set[str] = set()
    step_count = 0
    last_progress_time = time.time()
    final_api_status = "UNKNOWN"

    while True:
        time.sleep(poll_interval)

        # Check task status
        try:
            api_task = client.get_task(bytebot_task_id)
            final_api_status = api_task.get("status", "UNKNOWN")
        except Exception as exc:
            logger.warning("Failed to poll task %s: %s", bytebot_task_id, exc)
            continue

        # Fetch messages and process new steps
        try:
            messages = client.get_all_messages(bytebot_task_id)
            # Sort chronologically
            messages.sort(key=lambda m: m.get("createdAt", ""))

            assistant_msgs = extract_assistant_steps(messages)
            new_steps = [m for m in assistant_msgs if m.get("id") not in processed_msg_ids]

            if new_steps:
                last_progress_time = time.time()

            for msg in new_steps:
                step_count += 1
                msg_id = msg.get("id", "")
                processed_msg_ids.add(msg_id)
                content = msg.get("content", [])

                # Extract data
                tool_calls = parse_tool_calls(content)
                text = extract_text_content(content)
                screenshot_b64 = find_screenshot_for_step(messages, msg)

                # Save screenshot
                screenshot_file = None
                if screenshot_b64:
                    save_screenshot(output_dir, task_id, step_count, screenshot_b64)
                    screenshot_file = f"screenshots/step_{step_count:03d}.png"

                # Save step JSON
                step_data = StepData(
                    step_number=step_count,
                    timestamp=msg.get("createdAt", ""),
                    message_id=msg_id,
                    tool_calls=tool_calls,
                    text_content=text,
                    has_screenshot=screenshot_b64 is not None,
                    screenshot_file=screenshot_file,
                )
                save_step_json(output_dir, task_id, step_data)

                logger.info(
                    "  Step %d: %d tool calls, screenshot=%s",
                    step_count,
                    len(tool_calls),
                    "yes" if screenshot_b64 else "no",
                )
        except Exception as exc:
            logger.warning("Failed to fetch messages for task %s: %s", bytebot_task_id, exc)

        # Check termination conditions
        if final_api_status in TERMINAL_STATUSES:
            logger.info("Task %s reached terminal status: %s", task_id, final_api_status)
            break

        if step_count >= max_steps:
            logger.info("Task %s exceeded max steps (%d). Cancelling.", task_id, max_steps)
            client.cancel_task(bytebot_task_id)
            final_api_status = "CANCELLED"
            break

        if time.time() - last_progress_time > NO_PROGRESS_TIMEOUT:
            logger.info(
                "Task %s had no progress for %ds. Cancelling.",
                task_id, NO_PROGRESS_TIMEOUT,
            )
            client.cancel_task(bytebot_task_id)
            final_api_status = "CANCELLED"
            break

    end_time = datetime.now(timezone.utc).isoformat()
    start_dt = datetime.fromisoformat(start_time)
    end_dt = datetime.fromisoformat(end_time)
    duration = (end_dt - start_dt).total_seconds()

    # Determine result status
    if final_api_status == "COMPLETED":
        status = "success"
    elif final_api_status == "CANCELLED":
        status = "cancelled"
    else:
        status = "failed"

    result = TaskResult(
        task_id=task_id,
        domain=domain,
        instruction=instruction,
        bytebot_task_id=bytebot_task_id,
        status=status,
        final_api_status=final_api_status,
        total_steps=step_count,
        start_time=start_time,
        end_time=end_time,
        duration_seconds=round(duration, 1),
    )
    save_result_json(output_dir, result)
    return result


def run_benchmark(
    client: BytebotClient,
    tasks: list[dict],
    model: dict,
    output_dir: str,
    max_tasks: int,
    max_steps: int,
    poll_interval: float,
    resume: bool,
    config: dict,
) -> list[TaskResult]:
    """Run the full benchmark, iterating through tasks sequentially."""

    ensure_dir(output_dir)
    results: list[TaskResult] = []
    benchmark_start = datetime.now(timezone.utc).isoformat()

    # If resuming, load existing results
    if resume:
        for task in tasks[:max_tasks]:
            tid = task["task_id"]
            result_path = os.path.join(output_dir, tid, "result.json")
            if os.path.exists(result_path):
                with open(result_path, "r") as f:
                    data = json.load(f)
                results.append(TaskResult(**data))

    tasks_to_run = tasks[:max_tasks]
    total = len(tasks_to_run)

    for i, task in enumerate(tasks_to_run):
        task_id = task["task_id"]
        domain = task["domain"]

        if resume and task_already_completed(output_dir, task_id):
            logger.info(
                "[%d/%d] Skipping %s (%s) - already completed", i + 1, total, task_id, domain,
            )
            continue

        logger.info(
            "[%d/%d] Running task %s (%s): %.80s...",
            i + 1, total, task_id, domain, task["instruction"],
        )

        # Wait for agent to be idle before submitting
        client.wait_for_idle(poll_interval=poll_interval)

        result = run_single_task(client, task, model, output_dir, max_steps, poll_interval)
        results.append(result)

        logger.info(
            "[%d/%d] Result: %s (%d steps, %.1fs)",
            i + 1, total, result.status, result.total_steps, result.duration_seconds,
        )

        # Update final.json after each task
        benchmark_now = datetime.now(timezone.utc).isoformat()
        final_data = compute_final_stats(results, config, benchmark_start, benchmark_now)
        save_final_json(output_dir, final_data)

        # Brief pause between tasks
        time.sleep(2)

    benchmark_end = datetime.now(timezone.utc).isoformat()
    final_data = compute_final_stats(results, config, benchmark_start, benchmark_end)
    save_final_json(output_dir, final_data)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OSWorld benchmark tasks against the bytebot agent API.",
    )
    parser.add_argument(
        "--dataset",
        default="./osworld_dataset.json",
        help="Path to the extracted dataset JSON (default: ./osworld_dataset.json)",
    )
    parser.add_argument(
        "--api-url",
        default="http://localhost:9991",
        help="Bytebot agent API base URL (default: http://localhost:9991)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=100,
        help="Maximum number of tasks to run (default: 100)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum steps per task before cancelling (default: 100)",
    )
    parser.add_argument(
        "--output-dir",
        default="./osworld-logs",
        help="Output directory for logs (default: ./osworld-logs)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=3.0,
        help="Seconds between status polls (default: 3.0)",
    )
    parser.add_argument(
        "--domain",
        default="all",
        help="Filter to a specific domain (default: all)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help='Model config as JSON string. If omitted, auto-detects from API (uses first available model, same as web UI).',
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help='Shorthand for vLLM model name (e.g., "meta-llama/Meta-Llama-3.1-70B-Instruct"). Alternative to --model.',
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip tasks that already have a result.json in output-dir",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate mutually exclusive arguments
    if args.model and args.vllm_model:
        logger.error("Cannot use both --model and --vllm-model. Choose one.")
        sys.exit(1)

    # Initialize client and check connectivity first (needed for model auto-detection)
    client = BytebotClient(args.api_url)
    if not client.health_check():
        logger.error("Cannot reach bytebot API at %s", args.api_url)
        sys.exit(1)
    logger.info("Connected to bytebot API at %s", args.api_url)

    # Resolve model: vllm shorthand, explicit JSON, or auto-detect from API
    if args.vllm_model:
        # Convert vLLM model name to full model config
        model = {
            "provider": "proxy",
            "name": args.vllm_model,
            "title": f"vLLM: {args.vllm_model.split('/')[-1]}",
        }
        logger.info("Using vLLM model: %s", args.vllm_model)
    elif args.model:
        try:
            model = json.loads(args.model)
        except json.JSONDecodeError as exc:
            logger.error("Invalid --model JSON: %s", exc)
            sys.exit(1)
        for field in ("provider", "name", "title"):
            if field not in model:
                logger.error("Model JSON missing required field: %s", field)
                sys.exit(1)
    else:
        # Auto-detect: fetch models from API and pick the first one (same as web UI)
        try:
            available_models = client.get_models()
            if not available_models:
                logger.error("No models available from API. Provide --model explicitly.")
                sys.exit(1)
            model = available_models[0]
            logger.info("Auto-detected model: %s (%s)", model.get("title"), model.get("name"))
        except Exception as exc:
            logger.error("Failed to auto-detect model from API: %s", exc)
            logger.error("Provide --model explicitly.")
            sys.exit(1)

    # Load dataset
    if not os.path.exists(args.dataset):
        logger.error("Dataset file not found: %s", args.dataset)
        logger.error("Run extract_data.py first to generate the dataset.")
        sys.exit(1)

    with open(args.dataset, "r") as f:
        dataset = json.load(f)

    tasks = dataset.get("tasks", [])
    if not tasks:
        logger.error("No tasks found in dataset.")
        sys.exit(1)

    # Filter by domain if specified
    if args.domain != "all":
        tasks = [t for t in tasks if t["domain"] == args.domain]
        if not tasks:
            logger.error("No tasks found for domain: %s", args.domain)
            sys.exit(1)

    logger.info(
        "Loaded %d tasks (domain=%s), will run up to %d",
        len(tasks), args.domain, args.max_tasks,
    )

    config = {
        "max_tasks": args.max_tasks,
        "max_steps": args.max_steps,
        "poll_interval": args.poll_interval,
        "api_url": args.api_url,
        "model": model,
        "dataset": args.dataset,
        "domain": args.domain,
    }

    results = run_benchmark(
        client=client,
        tasks=tasks,
        model=model,
        output_dir=args.output_dir,
        max_tasks=args.max_tasks,
        max_steps=args.max_steps,
        poll_interval=args.poll_interval,
        resume=args.resume,
        config=config,
    )

    # Print summary
    success = sum(1 for r in results if r.status == "success")
    total = len(results)
    print(f"\n{'='*60}")
    print(f"Benchmark complete: {success}/{total} tasks succeeded ({success/total*100:.1f}%)" if total else "No tasks ran.")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
