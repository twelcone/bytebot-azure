#!/usr/bin/env python3
"""Test script to verify vLLM integration with Bytebot.

This script tests the vLLM model availability and runs a simple task.
"""

import json
import requests
import sys
import time


def test_vllm_models(api_url="http://localhost:9991"):
    """Test if vLLM models are available in the API."""

    print(f"Testing vLLM integration at {api_url}")
    print("-" * 60)

    # 1. Check API connectivity
    try:
        response = requests.get(f"{api_url}/tasks?limit=1", timeout=5)
        response.raise_for_status()
        print("✓ API is reachable")
    except requests.RequestException as e:
        print(f"✗ Cannot reach API: {e}")
        return False

    # 2. Get available models
    try:
        response = requests.get(f"{api_url}/tasks/models", timeout=5)
        response.raise_for_status()
        models = response.json()

        print(f"\n✓ Found {len(models)} models total")

        # Filter vLLM models (they use proxy provider)
        vllm_models = [m for m in models if m.get("title", "").startswith("vLLM:")]

        if vllm_models:
            print(f"✓ Found {len(vllm_models)} vLLM models:")
            for model in vllm_models:
                print(f"  - {model['title']} (name: {model['name']})")
        else:
            print("✗ No vLLM models found. Check your VLLM_* environment variables.")
            print("\nExpected environment variables:")
            print("  VLLM_BASE_URL=http://localhost:8000")
            print("  VLLM_MODEL_NAMES=meta-llama/Meta-Llama-3.1-70B-Instruct,...")
            return False

    except requests.RequestException as e:
        print(f"✗ Failed to get models: {e}")
        return False

    # 3. Test creating a task with vLLM model
    if vllm_models:
        test_model = vllm_models[0]
        print(f"\n Testing task creation with: {test_model['title']}")

        try:
            task_data = {
                "description": "Take a screenshot to test vLLM integration",
                "model": test_model
            }

            response = requests.post(
                f"{api_url}/tasks",
                json=task_data,
                timeout=10
            )
            response.raise_for_status()
            task = response.json()

            print(f"✓ Created task {task['id']}")

            # Wait a moment and check status
            time.sleep(3)
            response = requests.get(f"{api_url}/tasks/{task['id']}", timeout=5)
            response.raise_for_status()
            task_status = response.json()

            print(f"✓ Task status: {task_status['status']}")

            # Cancel the test task
            try:
                requests.post(f"{api_url}/tasks/{task['id']}/cancel", timeout=5)
                print("✓ Test task cancelled")
            except:
                pass

            return True

        except requests.RequestException as e:
            print(f"✗ Failed to create/check task: {e}")
            return False

    return True


def main():
    """Main test function."""

    # Check command line arguments
    api_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:9991"

    print("vLLM Integration Test for Bytebot")
    print("=" * 60)

    success = test_vllm_models(api_url)

    print("\n" + "=" * 60)
    if success:
        print("✓ vLLM integration test PASSED")
        print("\nYou can now run the benchmark with vLLM models:")
        print("  python run_benchmark.py --vllm-model 'meta-llama/Meta-Llama-3.1-70B-Instruct'")
    else:
        print("✗ vLLM integration test FAILED")
        print("\nPlease ensure:")
        print("1. vLLM server is running (e.g., at http://localhost:8000)")
        print("2. Environment variables are set in .env file:")
        print("   - VLLM_BASE_URL")
        print("   - VLLM_MODEL_NAMES")
        print("3. bytebot-agent service has been restarted after .env changes")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()