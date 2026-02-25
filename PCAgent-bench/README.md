# PC-Eval Benchmark for Bytebot

This directory contains scripts and tools for running the PC-Eval benchmark tasks on Bytebot, testing the agent's ability to complete desktop automation tasks.

## PC-Eval Tasks

PC-Eval consists of 27 tasks that test various desktop automation capabilities including:
- File operations (Notepad, Word, Excel)
- Web browsing and search (Chrome)
- Email operations (Outlook)
- Application interactions (Calculator, Clock)
- Programming tasks (Visual Studio Code)

Tasks are defined in `PC-Eval.txt` with the format:
```
1→"Task description"
2→"Another task description"
```

## Prerequisites

1. **Bytebot Services Running**:
   ```bash
   # Start all services with Docker
   cd /Users/dangvqm/Project/bytebot-azure
   docker-compose -f docker/docker-compose.yml up -d
   ```

2. **Required Environment Variables** (if not using Docker):
   ```bash
   export BYTEBOT_AGENT_BASE_URL=http://localhost:9991
   export ANTHROPIC_API_KEY=your_key_here  # or other provider keys
   ```

3. **Python Dependencies**:
   ```bash
   pip install requests
   ```

## Quick Start

### 1. List Available Tasks

```bash
python test_pc_eval.py --list
```

### 2. Run a Single Task

```bash
# Run task #1
python test_pc_eval.py --task-num 1

# Run a custom task
python test_pc_eval.py --custom "Open Chrome and search for weather"
```

### 3. Run Full Benchmark

```bash
# Run all tasks with Claude Sonnet 3.5
python run_pc_eval_benchmark.py \
  --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'

# Run specific tasks (1, 2, and 3)
python run_pc_eval_benchmark.py \
  --tasks 1,2,3 \
  --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'

# Resume from task 5
python run_pc_eval_benchmark.py \
  --start-from 5 \
  --model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'
```

## Model Configuration

### Anthropic Claude Models
```bash
--model '{"provider":"anthropic","name":"claude-3-5-sonnet-20241022","title":"Claude Sonnet 3.5"}'
--model '{"provider":"anthropic","name":"claude-3-opus-20240229","title":"Claude Opus 3"}'
--model '{"provider":"anthropic","name":"claude-3-5-haiku-20241022","title":"Claude Haiku 3.5"}'
```

### OpenAI Models
```bash
--model '{"provider":"openai","name":"gpt-4o","title":"GPT-4o"}'
--model '{"provider":"openai","name":"gpt-4o-mini","title":"GPT-4o Mini"}'
```

### Google Models
```bash
--model '{"provider":"google","name":"gemini-2.0-flash-exp","title":"Gemini 2.0 Flash"}'
```

### vLLM Models (if configured)
```bash
# Using the convenience flag
python run_pc_eval_benchmark.py --vllm-model "microsoft/UI-TARS-1.5-7B"

# Or with full model config
--model '{"provider":"proxy","name":"microsoft/UI-TARS-1.5-7B","title":"vLLM: UI-TARS 1.5 7B"}'
```

## Benchmark Output

The benchmark runner creates structured output in the `pc_eval_results/` directory:

```
pc_eval_results/
└── run_20240225_143022_claude-3-5-sonnet/
    ├── summary.json          # Cumulative results
    ├── report.txt            # Human-readable report
    ├── task_001.json         # Individual task result
    ├── task_002.json
    └── ...
```

### Output Files

- **summary.json**: Overall benchmark statistics and all task results
- **report.txt**: Detailed human-readable report with success rates
- **task_XXX.json**: Individual task details including:
  - Task status (COMPLETED, FAILED, TIMEOUT, etc.)
  - Execution duration
  - Message count
  - Last 10 messages from the conversation

## Monitoring Tasks

While tasks are running, you can monitor them through:

1. **Web UI**: http://localhost:9992
2. **VNC Viewer**: http://localhost:9990 (to see the desktop)
3. **API**:
   ```bash
   # Get task status
   curl http://localhost:9991/tasks/{task_id}

   # Get task messages
   curl http://localhost:9991/tasks/{task_id}/messages
   ```

## Task Timeouts

- **Progress Timeout**: 5 minutes without new messages → task cancelled
- **Maximum Duration**: 10 minutes per task → task cancelled

You can adjust these in `run_pc_eval_benchmark.py`:
```python
NO_PROGRESS_TIMEOUT = 300  # seconds
MAX_TASK_DURATION = 600    # seconds
```

## Troubleshooting

### Connection Issues
```bash
# Check if services are running
docker ps

# Check agent health
curl http://localhost:9991/tasks?limit=1

# View logs
docker logs bytebot-agent
```

### Task Failures

1. **TIMEOUT**: Task took too long or made no progress
   - Some tasks may require specific applications installed
   - Check if the desktop environment has required apps

2. **FAILED**: Task encountered an error
   - Check task messages for error details
   - Verify model has sufficient capabilities

3. **NEEDS_HELP**: Agent couldn't complete autonomously
   - Some tasks may be too complex for certain models
   - Try with a more capable model

### Missing Applications

PC-Eval tasks expect a Windows-like environment with:
- Notepad
- Chrome browser
- Microsoft Office (Word, Excel, Outlook)
- Calculator
- Clock
- Visual Studio Code

For Bytebot's Ubuntu environment, you may need to:
1. Install equivalent applications
2. Modify task descriptions to use available apps
3. Create custom tasks suited to your environment

## Custom Tasks

You can create custom benchmark suites by:

1. Creating a new task file:
```python
# my_tasks.txt
1→"Open Firefox and navigate to example.com"
2→"Create a text file with today's date"
```

2. Modifying the parser in the scripts to read your file

3. Running with your custom tasks:
```bash
python test_pc_eval.py --custom "Your custom task description"
```

## Results Analysis

After running the benchmark, analyze results with:

```python
import json
from pathlib import Path

# Load summary
with open("pc_eval_results/run_*/summary.json") as f:
    data = json.load(f)

# Calculate metrics
total = len(data["results"])
completed = sum(1 for r in data["results"] if r["status"] == "COMPLETED")
success_rate = (completed / total) * 100

print(f"Success Rate: {success_rate:.1f}%")
```

## Contributing

To add new tasks or improve the benchmark:

1. Add tasks to `PC-Eval.txt` following the existing format
2. Test with `test_pc_eval.py` first
3. Run full benchmark with `run_pc_eval_benchmark.py`
4. Share results and insights

## Known Limitations

1. **Application Availability**: Tasks expect Windows applications that may not be available in Ubuntu
2. **File Paths**: Some tasks reference Windows-style paths (e.g., 'Documents')
3. **UI Elements**: Tasks assume Windows UI conventions that may differ in Linux/XFCE

Consider adapting task descriptions for your specific environment.

## Support

For issues or questions:
- Check Bytebot logs: `docker logs bytebot-agent`
- Review task messages in output JSON files
- Ensure all required services are running
- Verify API keys are configured correctly