# OSWorld Benchmark for Bytebot

Benchmark suite that evaluates Bytebot against [OSWorld](https://github.com/xlang-ai/OSWorld) tasks -- a collection of 368 real-world desktop tasks spanning 10 application domains.

## Overview

The benchmark consists of two scripts:

| Script | Purpose |
|---|---|
| `extract_data.py` | Parses the OSWorld repo and produces a consolidated `osworld_dataset.json` |
| `run_benchmark.py` | Submits tasks to the Bytebot API, monitors execution, and collects results |

### How it works

1. **extract_data.py** reads `OSWorld/evaluation_examples/test_all.json` (the canonical task list) and loads each task JSON from `examples/{domain}/{task_id}.json`. It outputs a single JSON file containing all 368 tasks with their metadata.

2. **run_benchmark.py** iterates through the dataset and for each task:
   - Sends the task instruction to the Bytebot agent via `POST /tasks`
   - Polls the API for status updates and new messages
   - Counts each ASSISTANT message as one step (1 step = 1 AI iteration = 1 screenshot)
   - Saves screenshots and step data to disk
   - Cancels the task if it exceeds the step limit (default 100) or stalls for 5 minutes
   - Records success/failure based on the agent's self-reported status (`COMPLETED` = success)

### Task domains (368 total)

| Domain | Tasks | Description |
|---|---|---|
| `chrome` | 46 | Browser tasks (settings, bookmarks, extensions) |
| `gimp` | 26 | Image editing tasks |
| `libreoffice_calc` | 47 | Spreadsheet tasks |
| `libreoffice_impress` | 47 | Presentation tasks |
| `libreoffice_writer` | 23 | Document editing tasks |
| `multi_apps` | 101 | Cross-application workflows |
| `os` | 24 | System/terminal tasks |
| `thunderbird` | 15 | Email client tasks |
| `vlc` | 17 | Media player tasks |
| `vs_code` | 22 | Code editor tasks |

## Prerequisites

- Python 3.10+
- A running Bytebot stack (agent on port 9991, bytebotd on port 9990, PostgreSQL)
- The OSWorld repo cloned at `./OSWorld`

## Setup

```bash
cd osworld-bench
pip install -r requirements.txt
```

## Usage

### Step 1: Extract the dataset

```bash
python3 extract_data.py
```

This produces `osworld_dataset.json` with all 368 tasks. You only need to run this once.

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--examples-dir` | `./OSWorld/evaluation_examples/examples` | Path to the examples directory |
| `--test-all` | `./OSWorld/evaluation_examples/test_all.json` | Path to test_all.json |
| `--output` | `./osworld_dataset.json` | Output file path |
| `--domain` | `all` | Extract only one domain (e.g. `chrome`) |

**Example:** extract only Chrome tasks:
```bash
python3 extract_data.py --domain chrome --output chrome_dataset.json
```

### Step 2: Run the benchmark

```bash
# Run with auto-detected model (picks first from API, same as web UI)
python3 run_benchmark.py

# Run just 1 task to test
python3 run_benchmark.py --max-tasks 1

# Run only Chrome tasks
python3 run_benchmark.py --domain chrome --max-tasks 10

# Run with explicit model
python3 run_benchmark.py --model '{"provider":"anthropic","name":"claude-sonnet-4-20250514","title":"Claude Sonnet 4"}'

# Resume a previous run (skip tasks with existing results)
python3 run_benchmark.py --resume
```

**Options:**

| Flag | Default | Description |
|---|---|---|
| `--dataset` | `./osworld_dataset.json` | Path to extracted dataset |
| `--api-url` | `http://localhost:9991` | Bytebot agent API URL |
| `--max-tasks` | `100` | Max number of tasks to run |
| `--max-steps` | `100` | Max steps per task before cancelling |
| `--output-dir` | `./osworld-logs` | Output directory for logs |
| `--poll-interval` | `3.0` | Seconds between API polls |
| `--domain` | `all` | Filter to one domain |
| `--model` | auto-detect | Model config as JSON string |
| `--resume` | off | Skip tasks that already have a `result.json` |

## Output structure

```
osworld-logs/
|-- <task_id>/
|   |-- screenshots/
|   |   |-- step_001.png
|   |   |-- step_002.png
|   |   |-- ...
|   |-- step_001.json
|   |-- step_002.json
|   |-- ...
|   |-- result.json
|-- final.json
```

### step_NNN.json

Per-step data for each AI iteration:

```json
{
  "step_number": 1,
  "timestamp": "2026-02-08T12:01:23.456Z",
  "message_id": "uuid",
  "tool_calls": [
    {"name": "computer_click_mouse", "input": {"coordinates": {"x": 100, "y": 200}}},
    {"name": "computer_screenshot", "input": {}}
  ],
  "text_content": "I'll click on the settings icon...",
  "has_screenshot": true,
  "screenshot_file": "screenshots/step_001.png"
}
```

### result.json

Per-task result:

```json
{
  "task_id": "030eeff7-b492-4218-b312-701ec99ee0cc",
  "domain": "chrome",
  "instruction": "Can you enable the 'Do Not Track' feature...",
  "bytebot_task_id": "uuid-from-api",
  "status": "success",
  "final_api_status": "COMPLETED",
  "total_steps": 12,
  "start_time": "2026-02-08T10:00:00Z",
  "end_time": "2026-02-08T10:01:35Z",
  "duration_seconds": 95.2
}
```

Status values:
- `success` -- agent marked the task as COMPLETED
- `failed` -- agent marked as FAILED, or entered NEEDS_HELP/NEEDS_REVIEW
- `cancelled` -- exceeded step limit or stalled (no progress for 5 min)
- `error` -- API/network error during task creation

### final.json

Aggregate statistics, updated after each task completes:

```json
{
  "benchmark_config": { "max_tasks": 100, "max_steps": 100, "model": {...}, ... },
  "summary": {
    "total_tasks": 100,
    "completed": 45,
    "failed": 40,
    "cancelled": 10,
    "errors": 5,
    "success_rate": 0.45,
    "avg_steps": 23.4,
    "avg_duration_seconds": 187.3,
    "max_steps_task": { "task_id": "...", "steps": 98 },
    "min_steps_task": { "task_id": "...", "steps": 2 },
    "longest_task": { "task_id": "...", "duration_seconds": 1245.6 }
  },
  "per_domain": {
    "chrome": { "total": 15, "completed": 8, "success_rate": 0.533, "avg_steps": 18.2 },
    "gimp": { "total": 10, "completed": 3, "success_rate": 0.3, "avg_steps": 35.1 }
  },
  "started_at": "2026-02-08T10:00:00Z",
  "finished_at": "2026-02-08T15:30:00Z",
  "total_duration_seconds": 19800.0,
  "tasks": [
    { "task_id": "...", "domain": "chrome", "status": "success", "total_steps": 12, "duration_seconds": 95.2 }
  ]
}
```

## Evaluation method

This benchmark uses **agent self-report** evaluation:

- **Success**: The Bytebot agent marks the task as `COMPLETED`
- **Failed**: The agent marks it as `FAILED`, or enters `NEEDS_HELP`/`NEEDS_REVIEW` (no human available)
- **Cancelled**: The task exceeded the step limit or produced no new messages for 5 minutes

This differs from OSWorld's native evaluation which uses file comparison, app state checks, and 50+ metric functions. The self-report approach measures the agent's ability to attempt and complete real desktop tasks end-to-end without external validation.

## Error handling

- **Network errors**: Retries up to 3 times with exponential backoff (2s, 4s, 8s)
- **Stalled tasks**: Auto-cancelled after 5 minutes with no new messages
- **Step limit**: Tasks exceeding `--max-steps` are cancelled
- **Queue management**: Waits for the agent to be idle before submitting each task
- **Partial runs**: Use `--resume` to continue from where a previous run left off
