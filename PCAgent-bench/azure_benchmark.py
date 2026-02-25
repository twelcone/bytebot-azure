#!/usr/bin/env python3
"""Run PC-Eval benchmark with Azure Claude models."""

import argparse
import json
import sys
import subprocess
import requests


def get_azure_model():
    """Fetch Azure model configuration from the Bytebot API."""
    try:
        response = requests.get("http://localhost:9991/tasks/models", timeout=5)
        response.raise_for_status()
        models = response.json()

        # Find Azure models
        azure_models = [m for m in models if "Azure:" in m.get("title", "")]

        if azure_models:
            return azure_models[0]
        else:
            return None
    except requests.ConnectionError:
        print("❌ Cannot connect to Bytebot agent at http://localhost:9991")
        print("   Make sure services are running:")
        print("   docker-compose -f docker/docker-compose.yml up -d")
        sys.exit(1)
    except Exception as e:
        print(f"Error fetching models: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run PC-Eval benchmark with Azure Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s test 5        # Test task #5
  %(prog)s quick         # Run tasks 1-3
  %(prog)s full          # Run all 27 tasks
  %(prog)s resume 10     # Resume from task #10
  %(prog)s list          # List all tasks
        """
    )

    parser.add_argument(
        "command",
        choices=["test", "quick", "full", "resume", "list", "check"],
        help="Command to run"
    )
    parser.add_argument(
        "option",
        nargs="?",
        type=int,
        help="Task number for 'test' or 'resume' commands"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("PC-Eval Benchmark with Azure Claude")
    print("=" * 60)
    print()

    # Check for Azure model (except for 'check' command)
    if args.command != "check":
        azure_model = get_azure_model()

        if not azure_model:
            print("⚠️  No Azure models found!")
            print()
            print("To enable Azure Claude:")
            print()
            print("1. Edit docker/.env and add:")
            print("   AZURE_ANTHROPIC_ENDPOINT=https://your-resource.cognitiveservices.azure.com")
            print("   AZURE_ANTHROPIC_API_KEY=your-api-key")
            print("   AZURE_ANTHROPIC_DEPLOYMENT=claude-sonnet-45")
            print()
            print("2. Restart Docker services:")
            print("   docker-compose -f docker/docker-compose.yml down")
            print("   docker-compose -f docker/docker-compose.yml up -d")
            print()
            print("Available models:")
            subprocess.run(["python", "PCAgent-bench/check_models.py"])
            sys.exit(1)

        model_name = azure_model["name"]
        model_title = azure_model["title"]
        model_config = json.dumps(azure_model)

        print(f"✅ Using model: {model_title}")
        print(f"   Deployment: {model_name}")
        print()

    # Execute commands
    if args.command == "test":
        task_num = args.option or 1
        print(f"Running test with task #{task_num}")
        print("=" * 40)
        subprocess.run([
            "python", "PCAgent-bench/test_pc_eval.py",
            "--task-num", str(task_num),
            "--model", model_config
        ])

    elif args.command == "quick":
        print("Running quick benchmark (tasks 1-3)")
        print("=" * 40)
        subprocess.run([
            "python", "PCAgent-bench/run_pc_eval_benchmark.py",
            "--tasks", "1,2,3",
            "--model", model_config
        ])

    elif args.command == "full":
        print("Running full PC-Eval benchmark (27 tasks)")
        print("=" * 40)
        subprocess.run([
            "python", "PCAgent-bench/run_pc_eval_benchmark.py",
            "--model", model_config
        ])

    elif args.command == "resume":
        start_from = args.option or 1
        print(f"Resuming benchmark from task #{start_from}")
        print("=" * 40)
        subprocess.run([
            "python", "PCAgent-bench/run_pc_eval_benchmark.py",
            "--start-from", str(start_from),
            "--model", model_config
        ])

    elif args.command == "list":
        print("PC-Eval Tasks:")
        print("=" * 40)
        subprocess.run(["python", "PCAgent-bench/test_pc_eval.py", "--list"])

    elif args.command == "check":
        print("Checking available models:")
        print("=" * 40)
        subprocess.run(["python", "PCAgent-bench/check_models.py"])


if __name__ == "__main__":
    main()